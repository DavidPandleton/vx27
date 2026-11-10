#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "../src/vx27_gpu_persist.h"
#include "../src/gguf.h"
#include "../src/quant.h"
#include "../src/quant_avx2.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    const char *gguf = "C:/Users/ASUS A15/ql/models/Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf";
    char err[256] = {0};
    gguf_file g;
    CHECK(gguf_open(gguf, &g, err, sizeof(err)) == 0, "open gguf");
    const gguf_tensor *t = gguf_find_tensor(&g, "blk.0.ffn_gate.weight");
    CHECK(t && t->type == 12, "q4k tensor");
    int rows = (int)t->ne[1], cols = (int)t->ne[0], B = 8;
    const void *W = (const uint8_t*)g.map + g.data_offset + t->offset;
    float *x = malloc((size_t)cols * B * 4);
    unsigned seed = 7;
    for (int k = 0; k < B * cols; k++) { seed = seed*1664525u+1013904223u; x[k] = ((float)(seed>>8)/16777216.0f)*2.0f-1.0f; }

    vx27_gpu_ctx *ctx = NULL;
    CHECK(vx27_gpu_ctx_init(&ctx, W, rows, cols, err, sizeof(err)) == 0, "persist init (upload once)");
    float *y1 = malloc((size_t)rows*B*4), *y2 = malloc((size_t)rows*B*4);
    double m1 = 0, m2 = 0;
    CHECK(vx27_gpu_ctx_dispatch(ctx, x, y1, &m1, err, sizeof(err)) == 0, "dispatch1");
    CHECK(vx27_gpu_ctx_dispatch(ctx, x, y2, &m2, err, sizeof(err)) == 0, "dispatch2 reuse");
    printf("  dispatch1 %.3f ms dispatch2 %.3f ms y0=%.6f\n", m1, m2, y1[0]);
    double md = 0; for (size_t i = 0; i < (size_t)rows*B; i++) { double d = fabs((double)y1[i]-y2[i]); if (d > md) md = d; }
    CHECK(md == 0.0, "persistent deterministic");
    // vs CPU single dot y0
    size_t nblocks = (size_t)cols/256;
    block_q8_k *q8 = malloc(nblocks*B*sizeof(block_q8_k));
    for (int tt = 0; tt < B; tt++) quantize_q8_k(x + (size_t)tt*cols, q8 + (size_t)tt*nblocks, (int)nblocks);
    float want = dot_q4_k_q8_k_avx2((const block_q4_k*)W, q8, nblocks);
    printf("  y0 gpu=%.6f cpu=%.6f\n", y1[0], want);
    CHECK(fabsf(y1[0]-want) <= 0.02f*fabsf(want)+0.02f, "persist MATCH cpu");
    // bench 20x reuse
#ifdef _WIN32
    LARGE_INTEGER f, c0, c1; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c0);
    for (int it = 0; it < 20; it++) vx27_gpu_ctx_dispatch(ctx, x, y1, NULL, err, sizeof(err));
    QueryPerformanceCounter(&c1);
    double ms = (double)(c1.QuadPart-c0.QuadPart)*1000.0/(double)f.QuadPart;
#else
    double ms = 1.0;
#endif
    printf("  bench 20x reuse: %.2f ms (%.3f ms/dispatch)\n", ms, ms/20);
    free(x); free(y1); free(y2); free(q8);
    vx27_gpu_ctx_free(ctx); gguf_close(&g);
    if (fails) { printf("test_persist: %d FAIL\n", fails); return 1; }
    printf("test_persist: PASS\n");
    return 0;
}
