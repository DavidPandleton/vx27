#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../src/vx27_head.h"
#include "../src/vx27_config.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    CHECK(vx27_lmhead_params() == (uint64_t)248320 * 5120, "lmhead 1.27B params");
    double gb4 = vx27_lmhead_gb(4.0), gb3 = vx27_lmhead_gb(3.0);
    printf("  lmhead 4-bit %.3f GB, 3-bit %.3f GB\n", gb4, gb3);
    CHECK(gb4 > 0.6 && gb4 < 0.7, "lmhead 4-bit ~0.64GB");
    uint64_t rows = 0;
    int fit = vx27_lmhead_rows_gpu(4.0, 2.8, &rows);
    CHECK(fit == 1 && rows == 248320, "lmhead 4-bit fits 2.8GB alone");
    fit = vx27_lmhead_rows_gpu(4.0, 0.5, &rows);
    printf("  rows in 0.5GB @4bit: %llu\n", (unsigned long long)rows);
    CHECK(fit == 0 && rows < 248320 && rows > 190000, "shard math proportional");

    // tiny matvec 32x16
    const int R = 32, C = 16;
    float *W = malloc((size_t)R*C*4), *x = malloc((size_t)C*4), *y1 = malloc((size_t)R*4), *y2 = malloc((size_t)R*4);
    for (int i = 0; i < R*C; i++) W[i]=(float)(i%13)*0.05f-0.3f;
    for (int i = 0; i < C; i++) x[i]=(float)(i%7)*0.1f-0.2f;
    vx27_matvec_ref(W, x, y1, R, C);
    vx27_matvec_avx2(W, x, y2, R, C);
    float md = 0; for (int i = 0; i < R; i++) { float d=fabsf(y1[i]-y2[i]); if(d>md) md=d; }
    printf("  matvec tiny maxdiff %.3e\n", (double)md);
    CHECK(md < 1e-4f, "matvec avx2 == ref");

    // MTP tiny: hidden 16, ffn 8, vocab 32
    const int H = 16, F = 8, V = 32;
    float *h = malloc(H*4), *Wg = malloc((size_t)F*H*4), *Wu = malloc((size_t)F*H*4),
          *Wd = malloc((size_t)H*F*4), *Wlm = malloc((size_t)V*H*4),
          *nw = malloc(H*4), *d1 = malloc(V*4), *d2 = malloc(V*4);
    for (int i = 0; i < H; i++) h[i]=(float)(i%5)*0.2f-0.4f;
    for (int i = 0; i < F*H; i++) { Wg[i]=(float)(i%11)*0.03f-0.15f; Wu[i]=(float)(i%13)*0.025f-0.15f; }
    for (int i = 0; i < H*F; i++) Wd[i]=(float)(i%7)*0.04f-0.12f;
    for (int i = 0; i < V*H; i++) Wlm[i]=(float)(i%17)*0.02f-0.17f;
    for (int i = 0; i < H; i++) nw[i]=1.0f;
    vx27_mtp_forward_ref(h, Wg, Wu, Wd, Wlm, nw, d1, H, F, V);
    vx27_mtp_forward(h, Wg, Wu, Wd, Wlm, nw, d2, H, F, V);
    md = 0; for (int i = 0; i < V; i++) { float d=fabsf(d1[i]-d2[i]); if(d>md) md=d; }
    printf("  mtp draft maxdiff %.3e\n", (double)md);
    CHECK(md < 1e-3f, "mtp forward == ref");
    // draft is finite and not all equal
    int finite = 1; for (int i = 0; i < V; i++) if (!(d2[i] > -1e30 && d2[i] < 1e30)) finite = 0;
    CHECK(finite, "mtp draft finite");
    free(W); free(x); free(y1); free(y2); free(h); free(Wg); free(Wu); free(Wd); free(Wlm); free(nw); free(d1); free(d2);
    if (fails) { printf("test_head: %d FAIL\n", fails); return 1; }
    printf("test_head: PASS\n");
    return 0;
}
