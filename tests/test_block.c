#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
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
#include "../src/vx27_block.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

static float frand(unsigned *s) {
    *s = *s * 1664525u + 1013904223u;
    return ((int)(*s >> 9) & 0xffff) / 32768.0f - 1.0f;
}

int main(void) {
    vx27_block_cfg bc = {32, 64, 4, 2, 8, 8, 2, 2, 8, 10000.0f, 1.0f};
    int H = bc.hidden, F = bc.ffn;
    unsigned seed = 12345;
    // alloc weights
    size_t nWq = (size_t)H*bc.n_q*bc.hd, nWk = (size_t)H*bc.n_kv*bc.hd, nWv = nWk;
    size_t nWo = (size_t)bc.n_q*bc.hd*H, nWg = (size_t)H*F, nWu = nWg, nWd = (size_t)F*H;
    float *attn_n = malloc(H*4), *ffn_n = malloc(H*4), *qkn = malloc((size_t)bc.hd*4);
    float *Wq = malloc(nWq*4), *Wk = malloc(nWk*4), *Wv = malloc(nWv*4), *Wo = malloc(nWo*4);
    float *Wg = malloc(nWg*4), *Wu = malloc(nWu*4), *Wd = malloc(nWd*4);
    for (int i = 0; i < H; i++) { attn_n[i] = 1.0f; ffn_n[i] = 1.0f; }
    for (int i = 0; i < bc.hd; i++) qkn[i] = 1.0f;
    for (size_t i = 0; i < nWq; i++) Wq[i] = frand(&seed)*0.1f;
    for (size_t i = 0; i < nWk; i++) Wk[i] = frand(&seed)*0.1f;
    for (size_t i = 0; i < nWv; i++) Wv[i] = frand(&seed)*0.1f;
    for (size_t i = 0; i < nWo; i++) Wo[i] = frand(&seed)*0.1f;
    for (size_t i = 0; i < nWg; i++) Wg[i] = frand(&seed)*0.1f;
    for (size_t i = 0; i < nWu; i++) Wu[i] = frand(&seed)*0.1f;
    for (size_t i = 0; i < nWd; i++) Wd[i] = frand(&seed)*0.1f;
    vx27_block_weights w = {attn_n, ffn_n, Wq, Wk, Wv, Wo, Wg, Wu, Wd, qkn};

    float *x = malloc(H*4), *out1 = malloc(H*4), *out2 = malloc(H*4);
    for (int i = 0; i < H; i++) x[i] = frand(&seed);
    float *kc1 = calloc((size_t)4*bc.n_kv*bc.hd, 4), *vc1 = calloc((size_t)4*bc.n_kv*bc.hd, 4);
    float *kc2 = calloc((size_t)4*bc.n_kv*bc.hd, 4), *vc2 = calloc((size_t)4*bc.n_kv*bc.hd, 4);

    char err[256] = {0};
    threadpool *tp = tp_create(8, err, sizeof(err));
    CHECK(tp != NULL, "threadpool 8 created");

    // full block: ref (tp=NULL) vs tp
    vx27_full_block_forward(&bc, NULL, x, kc1, vc1, 0, out1, &w);
    vx27_full_block_forward(&bc, tp, x, kc2, vc2, 0, out2, &w);
    float md = 0; for (int i = 0; i < H; i++) { float d = fabsf(out1[i]-out2[i]); if (d>md) md=d; }
    printf("  full block tp vs ref maxdiff %.3e\n", (double)md);
    CHECK(md < 1e-4f, "full block tp == ref");
    // determinism: second run same
    float *out3 = malloc(H*4); float *kc3 = calloc((size_t)4*bc.n_kv*bc.hd,4), *vc3 = calloc((size_t)4*bc.n_kv*bc.hd,4);
    vx27_full_block_forward(&bc, tp, x, kc3, vc3, 0, out3, &w);
    md = 0; for (int i = 0; i < H; i++) { float d=fabsf(out2[i]-out3[i]); if(d>md) md=d; }
    CHECK(md == 0.0f, "full block deterministic");

    // linear block
    size_t nWlq = (size_t)H*bc.lin_qk*bc.lin_hd, nWlv = (size_t)H*bc.lin_v*bc.lin_hd;
    size_t nWlo = (size_t)bc.lin_v*bc.lin_hd*H;
    float *Lq = malloc(nWlq*4), *Lk = malloc(nWlq*4), *Lv = malloc(nWlv*4), *Lo = malloc(nWlo*4);
    for (size_t i = 0; i < nWlq; i++) { Lq[i]=frand(&seed)*0.1f; Lk[i]=frand(&seed)*0.1f; }
    for (size_t i = 0; i < nWlv; i++) Lv[i]=frand(&seed)*0.1f;
    for (size_t i = 0; i < nWlo; i++) Lo[i]=frand(&seed)*0.1f;
    vx27_block_weights wl = {attn_n, ffn_n, Lq, Lk, Lv, Lo, Wg, Wu, Wd, NULL};
    float *S1 = calloc((size_t)bc.lin_qk*bc.lin_hd*bc.lin_hd,4);
    float *S2 = calloc((size_t)bc.lin_qk*bc.lin_hd*bc.lin_hd,4);
    float *lo1 = malloc(H*4), *lo2 = malloc(H*4);
    vx27_linear_block_forward(&bc, NULL, x, S1, lo1, &wl);
    vx27_linear_block_forward(&bc, tp, x, S2, lo2, &wl);
    md = 0; for (int i = 0; i < H; i++) { float d=fabsf(lo1[i]-lo2[i]); if(d>md) md=d; }
    printf("  linear block tp vs ref maxdiff %.3e\n", (double)md);
    CHECK(md < 1e-4f, "linear block tp == ref");

    // bench: 200 full + 200 linear proxy blocks
    const int iters = 200;
    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        float *kc = calloc((size_t)4*bc.n_kv*bc.hd,4), *vc = calloc((size_t)4*bc.n_kv*bc.hd,4);
        float *o = malloc(H*4);
        vx27_full_block_forward(&bc, tp, x, kc, vc, 0, o, &w);
        free(kc); free(vc); free(o);
    }
    double t1 = now_ms();
    printf("  bench full-proxy x%d: %.2f ms (%.3f ms/block)\n", iters, t1-t0, (t1-t0)/iters);
    t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        float *S = calloc((size_t)bc.lin_qk*bc.lin_hd*bc.lin_hd,4);
        float *o = malloc(H*4);
        vx27_linear_block_forward(&bc, tp, x, S, o, &wl);
        free(S); free(o);
    }
    t1 = now_ms();
    printf("  bench linear-proxy x%d: %.2f ms (%.3f ms/block)\n", iters, t1-t0, (t1-t0)/iters);

    // real-dim estimate
    double flops_f, bytes_f, flops_l, bytes_l;
    vx27_block_estimate(1, &flops_f, &bytes_f);
    vx27_block_estimate(0, &flops_l, &bytes_l);
    double tot_flops = 16*flops_f + 48*flops_l;
    (void)bytes_f; (void)bytes_l;
    printf("  estimate real-dim per token: flops=%.3f GFLOP\n", tot_flops/1e9);
    printf("  estimate weight bytes Q4 per full pass: full=%.2fGB linear=%.2fGB\n",
        (16*bytes_f+48*0)/1e9, (48*bytes_l)/1e9);

    tp_free(tp);
    free(attn_n); free(ffn_n); free(qkn); free(Wq); free(Wk); free(Wv); free(Wo);
    free(Wg); free(Wu); free(Wd); free(x); free(out1); free(out2); free(out3);
    free(kc1); free(vc1); free(kc2); free(vc2); free(kc3); free(vc3);
    free(Lq); free(Lk); free(Lv); free(Lo); free(S1); free(S2); free(lo1); free(lo2);
    if (fails) { printf("test_block: %d FAIL\n", fails); return 1; }
    printf("test_block: PASS\n");
    return 0;
}
