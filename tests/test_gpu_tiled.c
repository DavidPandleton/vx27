#include <stdio.h>
#include <stdlib.h>
#include "../src/vx27_gpu.h"
#include "../src/gguf.h"

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
    double ms0=0,c0=0,gv0=0,ms1=0,c1=0,gv1=0; float y0=0,yc0=0,y1=0,yc1=0;
    int r0 = vx27_gpu_q4_bench(W, x, rows, cols, B, &ms0, &c0, &gv0, &y0, &yc0, err, sizeof(err));
    CHECK(r0 == 0, "v0 MATCH");
    printf("  v0 cpu %.2f ms gpu %.2f ms %.1f Gval/s y=%.6f\n", c0, ms0, gv0, y0);
    int r1 = vx27_gpu_q4_bench_tiled(W, x, rows, cols, B, &ms1, &c1, &gv1, &y1, &yc1, err, sizeof(err));
    CHECK(r1 == 0, "v1 tiled MATCH");
    printf("  v1 cpu %.2f ms gpu %.2f ms %.1f Gval/s y=%.6f\n", c1, ms1, gv1, y1);
    if (!r0 && !r1) {
        float rel = (float)((ms0 - ms1) / (ms0 + 1e-9) * 100.0);
        printf("  tiled vs v0: %.1f%% %s\n", rel >= 0 ? rel : -rel, rel >= 0 ? "faster" : "slower");
    }
    free(x); gguf_close(&g);
    if (fails) { printf("test_gpu_tiled: %d FAIL\n", fails); return 1; }
    printf("test_gpu_tiled: PASS\n");
    return 0;
}
