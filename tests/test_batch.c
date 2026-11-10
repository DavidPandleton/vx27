#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/vx27_batch.h"
#include "../src/threadpool.h"
#include "../src/gguf.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    const char *gguf = "C:/Users/ASUS A15/ql/models/Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf";
    char err[512] = {0};
    gguf_file g;
    CHECK(gguf_open(gguf, &g, err, sizeof(err)) == 0, "open gguf");
    const gguf_tensor *t = gguf_find_tensor(&g, "blk.0.ffn_gate.weight");
    CHECK(t && t->type == 12, "q4k tensor");
    int rows = (int)t->ne[1], cols = (int)t->ne[0], B = 8;
    const void *W = (const uint8_t*)g.map + g.data_offset + t->offset;
    float *x = malloc((size_t)cols * B * 4), *y = malloc((size_t)rows * B * 4);
    unsigned seed = 9;
    for (int k = 0; k < B * cols; k++) { seed = seed*1664525u+1013904223u; x[k] = ((float)(seed>>8)/16777216.0f)*2.0f-1.0f; }
    threadpool *tp = tp_create(8, err, sizeof(err));
    CHECK(tp != NULL, "tp 8");
    double ms = 0;
    int path = vx27_matvec_batch(1, tp, W, x, y, rows, cols, B, &ms);
    printf("  batch path=%s ms=%.2f y0=%.6f yN=%.6f best=%d (0=v0 1=persist 2=v2)\n", path ? "GPU" : "CPU", ms, y[0], y[(size_t)(B-1)*rows], vx27_batch_best());
    CHECK(path == 1, "gpu path selected (rows>=512,B==8)");
    // verify full y vs CPU reference (t-major)
    {
        float *yr = malloc((size_t)rows * B * 4);
        double m2 = 0;
        int p2 = vx27_matvec_batch(0, tp, W, x, yr, rows, cols, B, &m2);
        CHECK(p2 == 0, "cpu ref path");
        double md = 0; for (size_t i = 0; i < (size_t)rows*B; i++) { double d = fabs((double)y[i]-yr[i]); if (d > md) md = d; }
        printf("  full-y maxdiff gpu-vs-cpu %.3e\n", md);
        CHECK(md < 2e-2, "device-resident full y MATCH");
        free(yr);
    }
    // small shape falls back to CPU
    double ms2 = 0;
    int path2 = vx27_matvec_batch(1, tp, W, x, y, 64, cols, B, &ms2);
    CHECK(path2 == 0, "small rows fallback CPU");
    free(x); free(y); tp_free(tp); gguf_close(&g);
    if (fails) { printf("test_batch: %d FAIL\n", fails); return 1; }
    printf("test_batch: PASS\n");
    return 0;
}
