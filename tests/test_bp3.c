#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "../src/vx27_bp3.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // pack/unpack roundtrip SNR on 256 weights (8 groups)
    float w[256], u[256];
    unsigned s = 123;
    for (int i = 0; i < 256; i++) { s = s*1664525u+1013904223u; w[i] = ((float)(s>>9)/8388608.0f-0.5f)*4.0f; }
    vx27_bp3_grp G[8];
    for (int g = 0; g < 8; g++) vx27_bp3_pack(w + g*32, &G[g]);
    for (int g = 0; g < 8; g++) vx27_bp3_unpack_ref(&G[g], u + g*32);
    double sig = 0, noi = 0;
    for (int i = 0; i < 256; i++) { sig += (double)w[i]*w[i]; double d = w[i]-u[i]; noi += d*d; }
    double snr = 10*log10(sig/(noi+1e-12));
    printf("  bp3 snr %.2f dB size %.2f bits/param\n", snr, 3.0 + 32.0/32);
    CHECK(snr > 12.0, "bp3 snr >12dB (uniform 3-bit)");
    // dot ref vs avx2
    float x[256];
    for (int i = 0; i < 256; i++) x[i] = (float)(i%11)*0.07f-0.35f;
    float r1 = vx27_bp3_dot_ref(G, x, 8);
    float r2 = vx27_bp3_dot_avx2(G, x, 8);
    printf("  dot ref=%.6f avx2=%.6f diff=%.3e\n", r1, r2, fabsf(r1-r2));
    CHECK(fabsf(r1-r2) < 1e-3f, "bp3 dot avx2 == ref");
    // dot vs float (quant error bound)
    double sf = 0; for (int i = 0; i < 256; i++) sf += (double)w[i]*x[i];
    printf("  float=%.6f q=%.6f relerr=%.3f%%\n", (float)sf, r1, fabsf(r1-(float)sf)/(fabsf((float)sf)+1.0f)*100.0f);
    CHECK(fabsf(r1-(float)sf)/(fabsf((float)sf)+1.0f) < 0.08f, "bp3 quant err <8%");
    // bench 200k dots 256-wide
#ifdef _WIN32
    LARGE_INTEGER f, c0, c1; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c0);
    for (int it = 0; it < 200000; it++) { volatile float r = vx27_bp3_dot_avx2(G, x, 8); (void)r; }
    QueryPerformanceCounter(&c1);
    double ms = (double)(c1.QuadPart-c0.QuadPart)*1000.0/(double)f.QuadPart;
#else
    double ms = 1.0;
#endif
    printf("  bench 200k x 256-dot: %.2f ms (%.3f us/dot)\n", ms, ms*1000.0/200000);
    // v1 final: 128-grup + FP16 skala = 50B/128 = 3.125 bit
    float w1[128], x1[128];
    for (int i = 0; i < 128; i++) { w1[i] = (float)(i%29)*0.09f-1.2f; x1[i] = (float)(i%11)*0.07f-0.35f; }
    vx27_bp3_128 G1;
    vx27_bp3_128_pack(w1, &G1);
    float e1 = vx27_bp3_128_dot_ref(&G1, x1, 1);
    float e2 = vx27_bp3_128_dot_avx2(&G1, x1, 1);
    printf("  v1 dot ref=%.6f avx2=%.6f diff=%.3e bits=%.3f\n", e1, e2, fabsf(e1-e2), 50.0*8/128);
    CHECK(fabsf(e1-e2) < 1e-3f, "v1 dot avx2 == ref");
    double sg = 0; for (int i = 0; i < 128; i++) sg += (double)w1[i]*x1[i];
    CHECK(fabsf(e1-(float)sg)/(fabsf((float)sg)+1.0f) < 0.10f, "v1 quant err <10%");
    if (fails) { printf("test_bp3: %d FAIL\n", fails); return 1; }
    printf("test_bp3: PASS\n");
    return 0;
}
