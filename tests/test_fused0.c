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
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
#include "../src/vx27_fused.h"
#include "../src/vx27_bp3.h"
#include "../src/quant.h"
#include "../src/quant_avx2.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

typedef struct { const char *nm; float *f; } slot;
static float *findf(const slot *A, int N, const char *nm) {
    for (int i = 0; i < N; i++) if (!strcmp(A[i].nm, nm)) return A[i].f;
    return NULL;
}

// dequant via BYTES (tahan tensor 3D spt conv [10240,1,4])
static float *deq(const vx27_entry *e, const void *blob, uint64_t bytes) {
    uint64_t n;
    if (e->dtype == 0) n = bytes / 4;
    else if (e->dtype == 2) n = bytes / sizeof(block_q8_k) * 256;
    else if (e->dtype == 4) n = bytes / sizeof(vx27_bp3_128) * 128;
    else return NULL;
    float *f = malloc(n * 4 + 64);
    if (e->dtype == 0) memcpy(f, blob, (size_t)bytes);
    else if (e->dtype == 2) {
        const block_q8_k *q = blob; size_t nb = bytes / sizeof(block_q8_k);
        for (size_t b = 0; b < nb; b++) for (int i = 0; i < 256; i++) f[b*256+i] = (float)q[b].qs[i]*q[b].d;
    } else {
        const vx27_bp3_128 *g = blob; size_t ng = bytes / sizeof(vx27_bp3_128);
        for (size_t k = 0; k < ng; k++) {
            uint16_t h = g[k].scale_h; uint32_t u = h;
            uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
            if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
            else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
            else bits=sign|((exp+112u)<<23)|(mant<<13);
            float sc; memcpy(&sc,&bits,4);
            for (int i = 0; i < 128; i++) {
                uint32_t uu=((g[k].p[0*4+i/32]>>(i%32))&1)|(((g[k].p[1*4+i/32]>>(i%32))&1)<<1)|(((g[k].p[2*4+i/32]>>(i%32))&1)<<2);
                int q=(uu>=4)?(int)uu-8:(int)uu;
                f[k*128+i]=q*sc;
            }
        }
    }
    return f;
}

int main(void) {
    char err[512] = {0};
    vx27_entry ec[4096]; int nc = 0;
    CHECK(vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof(err)) == 0, "list real core");
    printf("  core tensors=%d\n", nc);
    vx27_map mc = {0};
    CHECK(vx27_mmap_open("build/vx27_core.vx27", &mc, err, sizeof(err)) == 0, "mmap real core");

    // kumpulkan blob per nama
    static slot S[64]; int ns = 0;
    const char *want[] = {
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        NULL };
    for (int w = 0; want[w]; w++) {
        int found = -1;
        for (int i = 0; i < nc; i++) if (!strcmp(ec[i].name, want[w])) { found = i; break; }
        if (found < 0) { printf("FAIL: missing %s\n", want[w]); fails++; continue; }
        uint64_t bb = 0; const void *p = vx27_mmap_blob(&mc, found, &bb);
        if (!p) { printf("FAIL: blob %s\n", want[w]); fails++; continue; }
        float *f = deq(&ec[found], p, bb);
        if (!f) { printf("FAIL: dequant %s\n", want[w]); fails++; continue; }
        S[ns].nm = want[w]; S[ns].f = f; ns++;
        if (ns == 0 || ns == 7 || ns == 14) printf("  dequant %d/%d %s\n", ns, 14, want[w]);
    }
    CHECK(ns == 14, "14 tensor layer-0 ketemu + terdequant");
#define FINDF(nm) findf(S, ns, nm)
    if (ns != 14) { printf("test_fused0: SKIP (missing tensors)\n"); return 1; }

    vx27_fused_w W = {5120, 16, 48, 128, 17408,
        FINDF("model.language_model.layers.0.input_layernorm.weight"),
        FINDF("model.language_model.layers.0.post_attention_layernorm.weight"),
        FINDF("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
        FINDF("model.language_model.layers.0.linear_attn.in_proj_z.weight"),
        FINDF("model.language_model.layers.0.linear_attn.in_proj_a.weight"),
        FINDF("model.language_model.layers.0.linear_attn.in_proj_b.weight"),
        FINDF("model.language_model.layers.0.linear_attn.A_log"),
        FINDF("model.language_model.layers.0.linear_attn.dt_bias"),
        FINDF("model.language_model.layers.0.linear_attn.conv1d.weight"),
        FINDF("model.language_model.layers.0.linear_attn.norm.weight"),
        FINDF("model.language_model.layers.0.linear_attn.out_proj.weight"),
        FINDF("model.language_model.layers.0.mlp.gate_proj.weight"),
        FINDF("model.language_model.layers.0.mlp.up_proj.weight"),
        FINDF("model.language_model.layers.0.mlp.down_proj.weight")};

    float *Sst = calloc((size_t)16*128*128, 4);
    float *Cst = calloc((size_t)(16*128*2+48*128)*3, 4);
    vx27_fused_state st = {Sst, Cst};
    float x[5120], o1[5120], o2[5120];
    unsigned seed = 7;
    for (int i = 0; i < 5120; i++) { seed = seed*1664525u+1013904223u; x[i] = ((float)(seed>>9)/8388608.0f-0.5f)*0.5f; }

    double t0 = now_ms();
    vx27_fused_delta_forward(&W, &st, x, o1, NULL);
    double t1 = now_ms();
    double chk1 = 0; int fin = 1;
    for (int i = 0; i < 5120; i++) { chk1 += o1[i]; if (!(o1[i] > -1e30 && o1[i] < 1e30)) fin = 0; }
    printf("  fwd L0 ref checksum=%.6f ms=%.1f finite=%d\n", chk1, t1-t0, fin);
    CHECK(fin, "output finite (no NaN/Inf)");

    memset(Sst, 0, (size_t)16*128*128*4);
    threadpool *tp = tp_create(8, err, sizeof(err));
    double t2 = now_ms();
    vx27_fused_delta_forward(&W, &st, x, o2, tp);
    double t3 = now_ms();
    double md = 0, chk2 = 0;
    for (int i = 0; i < 5120; i++) { double d = fabs((double)o1[i]-o2[i]); if (d > md) md = d; chk2 += o2[i]; }
    printf("  fwd L0 tp checksum=%.6f ms=%.1f maxdiff=%.3e\n", chk2, t3-t2, md);
    CHECK(md < 1e-3f, "tp == ref (8 worker)");
    tp_free(tp);
    for (int i = 0; i < ns; i++) free(S[i].f);
    free(Sst); free(Cst);
    vx27_mmap_close(&mc);
    if (fails) { printf("test_fused0: %d FAIL\n", fails); return 1; }
    printf("test_fused0: PASS\n");
    return 0;
}
