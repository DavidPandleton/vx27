#include <stdio.h>
#include <stdlib.h>
#include "../src/vx27_gpu.h"
#include "../src/gguf.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    int n = vx27_gpu_device_count();
    printf("  vulkan devices=%d\n", n);
    CHECK(n >= 1, "vulkan device present (RTX 3050)");
    if (n < 1) { printf("test_gpu: SKIP (no vulkan)\n"); return 0; }

    const char *gguf = "C:/Users/ASUS A15/ql/models/Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf";
    char err[256] = {0};
    gguf_file g;
    CHECK(gguf_open(gguf, &g, err, sizeof(err)) == 0, "open ql 4B gguf");
    const gguf_tensor *t = gguf_find_tensor(&g, "blk.0.ffn_gate.weight");
    CHECK(t && t->type == 12, "tensor blk.0.ffn_gate Q4_K");
    int rows = (int)t->ne[1], cols = (int)t->ne[0], B = 8;
    printf("  tensor rows=%d cols=%d B=%d\n", rows, cols, B);
    const void *W = (const uint8_t*)g.map + g.data_offset + t->offset;
    float *x = malloc((size_t)cols * B * 4);
    unsigned seed = 7;
    for (int k = 0; k < B * cols; k++) { seed = seed*1664525u+1013904223u; x[k] = ((float)(seed>>8)/16777216.0f)*2.0f-1.0f; }

    double gpu_ms = 0, cpu_ms = 0, gvals = 0; float yg = 0, yc = 0;
    int rc = vx27_gpu_q4_bench(W, x, rows, cols, B, &gpu_ms, &cpu_ms, &gvals, &yg, &yc, err, sizeof(err));
    printf("  y_gpu0=%.6f y_cpu0=%.6f\n", yg, yc);
    printf("  cpu %.2f ms | gpu dispatch %.2f ms | %.1f Gval/s\n", cpu_ms, gpu_ms, gvals);
    CHECK(rc == 0, "gpu MATCH cpu (2% tol)");
    if (rc != 0) printf("  bench err: %s\n", err);
    // expectation from ql PLAN: one-thread-per-row is slower than CPU 8-core; just report, don't fail on speed
    printf("  note: ql baseline gpu 45 Gval/s one-thread-per-row < cpu; tiling (PLAN Tahap 1) is future work\n");
    free(x);
    gguf_close(&g);
    if (fails) { printf("test_gpu: %d FAIL\n", fails); return 1; }
    printf("test_gpu: PASS\n");
    return 0;
}
