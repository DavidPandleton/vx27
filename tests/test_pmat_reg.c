#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) { return (double)clock()*1000.0/CLOCKS_PER_SEC; }
#endif
#include "../src/vx27_pmat.h"
#include "../src/vx27_bp3.h"
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // 1) sintetik: reg vs skalar harus identik (toleransi akumulasi FMA)
    float w[256], x[256]; unsigned s = 11;
    for (int i = 0; i < 256; i++) { s = s*1664525u+1013904223u; w[i] = ((float)(s>>9)/8388608.0f-0.5f)*2.0f; }
    for (int i = 0; i < 256; i++) x[i] = (float)(i%11)*0.07f-0.35f;
    vx27_bp3_128 H[2];
    for (int k = 0; k < 2; k++) {
        float sc = 0; { float am=0; for(int i=0;i<128;i++){float a=fabsf(w[k*128+i]); if(a>am)am=a;} sc = am>0?am/3.5f:1.0f; }
        uint16_t hh; { uint32_t u; memcpy(&u,&sc,4); int e=((u>>23)&0xFF)-127+15; if(e<=0)hh=(uint16_t)((u>>16)&0x8000); else if(e>=31)hh=(uint16_t)(((u>>16)&0x8000)|0x7C00); else hh=(uint16_t)(((u>>16)&0x8000)|((uint32_t)e<<10)|((u>>13)&0x3FF)); }
        H[k].scale_h = hh;
        for (int b = 0; b < 12; b++) H[k].p[b] = 0;
        for (int i = 0; i < 128; i++) {
            int q = (int)roundf(w[k*128+i]/sc); if (q>3)q=3; if(q<-4)q=-4;
            uint32_t uu = (uint32_t)(q&7);
            if (uu&1) H[k].p[0*4+i/32] |= (1u<<(i%32));
            if (uu&2) H[k].p[1*4+i/32] |= (1u<<(i%32));
            if (uu&4) H[k].p[2*4+i/32] |= (1u<<(i%32));
        }
    }
    float r1 = vx27_pmat_bp3_scalar(H, x, 256);
    float r2 = vx27_pmat_bp3_reg(H, x, 256);
    printf("  sintetik scalar=%.6f reg=%.6f diff=%.3e\n", r1, r2, fabsf(r1-r2));
    CHECK(fabsf(r1-r2) <= 1e-3f*(fabsf(r1)+1.0f), "reg == scalar sintetik");
    // bench 200k dot 256-wide
    double t0 = now_ms();
    for (int it = 0; it < 200000; it++) { volatile float r = vx27_pmat_bp3_scalar(H, x, 256); (void)r; }
    double t1 = now_ms();
    double t2 = now_ms();
    for (int it = 0; it < 200000; it++) { volatile float r = vx27_pmat_bp3_reg(H, x, 256); (void)r; }
    double t3 = now_ms();
    printf("  bench scalar %.2f ms, reg %.2f ms (%.2fx)\n", t1-t0, t3-t2, (t1-t0)/((t3-t2)+1e-9));

    // 2) baris REAL dari vx27_core (FFN gate layer-0, 5120 kolom)
    char err[512] = {0};
    vx27_entry ec[4096]; int nc = 0;
    if (vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof(err)) != 0) { printf("SKIP: %s\n", err); return 0; }
    vx27_map mc = {0};
    if (vx27_mmap_open("build/vx27_core.vx27", &mc, err, sizeof(err)) != 0) { printf("SKIP: %s\n", err); return 0; }
    int bi = -1; for (int i = 0; i < nc; i++) if (!strcmp(ec[i].name, "model.language_model.layers.0.mlp.gate_proj.weight")) { bi = i; break; }
    CHECK(bi >= 0, "gate_proj ketemu");
    uint64_t bb = 0; const void *row0 = vx27_mmap_blob(&mc, bi, &bb);
    CHECK(row0 && bb > 0, "mmap baris-0");
    float *xr = malloc(5120*4); for (int i = 0; i < 5120; i++) xr[i] = (float)(i%13)*0.05f-0.3f;
    float q1 = vx27_pmat_bp3_scalar(row0, xr, 5120);
    float q2 = vx27_pmat_bp3_reg(row0, xr, 5120);
    printf("  real scalar=%.6f reg=%.6f diff=%.3e\n", q1, q2, fabsf(q1-q2));
    CHECK(fabsf(q1-q2) <= 2e-3f*(fabsf(q1)+1.0f), "reg == scalar bobot nyata");
    t0 = now_ms(); for (int it = 0; it < 2000; it++) { volatile float r = vx27_pmat_bp3_scalar(row0, xr, 5120); (void)r; } t1 = now_ms();
    t2 = now_ms(); for (int it = 0; it < 2000; it++) { volatile float r = vx27_pmat_bp3_reg(row0, xr, 5120); (void)r; } double t4 = now_ms();
    printf("  bench real 5120-wide scalar %.2f ms, reg %.2f ms (%.2fx)\n", t1-t0, t4-t2, (t1-t0)/((t4-t2)+1e-9));
    free(xr); vx27_mmap_close(&mc);
    if (fails) { printf("test_pmat_reg: %d FAIL\n", fails); return 1; }
    printf("test_pmat_reg: PASS\n");
    return 0;
}
