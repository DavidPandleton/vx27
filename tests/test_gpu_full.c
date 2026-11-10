#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
    float *y0 = malloc((size_t)rows*B*4), *y1 = malloc((size_t)rows*B*4), *y8 = malloc((size_t)rows*B*4);
    double m0=0,m1=0,m8=0;
    CHECK(vx27_gpu_q4_bench_full(W,x,rows,cols,B,y0,&m0,err,sizeof(err))==0, "v0 full MATCH");
    printf("  v0 full %.2f ms y0=%.6f yN=%.6f\n", m0, y0[0], y0[(size_t)rows*B-1]);
    CHECK(vx27_gpu_q4_bench_tiled_full(W,x,rows,cols,B,y1,&m1,err,sizeof(err))==0, "v1 tiled full MATCH");
    printf("  v1 full %.2f ms y0=%.6f yN=%.6f\n", m1, y1[0], y1[(size_t)rows*B-1]);
    CHECK(vx27_gpu_q4_bench_tiled8_full(W,x,rows,cols,B,y8,&m8,err,sizeof(err))==0, "v2 8acc full MATCH");
    printf("  v2 full %.2f ms y0=%.6f yN=%.6f\n", m8, y8[0], y8[(size_t)rows*B-1]);
    // full-vector equality across shaders
    float md01=0,md08=0;
    for (size_t i=0;i<(size_t)rows*B;i++){float a=fabsf(y0[i]-y1[i]);if(a>md01)md01=a;a=fabsf(y0[i]-y8[i]);if(a>md08)md08=a;}
    printf("  maxdiff v0-v1 %.3e v0-v2 %.3e\n",(double)md01,(double)md08);
    CHECK(md01<1e-3f && md08<1e-3f, "all shaders agree full y");
    // autotune pick fastest
    const char *best = (m0<m1&&m0<m8)?"v0":(m1<m8?"v1-tiled":"v2-8acc");
    double bm = m0<m1?(m0<m8?m0:m8):(m1<m8?m1:m8);
    printf("  autotune best=%s %.2f ms\n", best, bm);
    CHECK(bm>0&&bm<50.0, "autotune valid");
    free(x);free(y0);free(y1);free(y8);gguf_close(&g);
    if (fails) { printf("test_gpu_full: %d FAIL\n", fails); return 1; }
    printf("test_gpu_full: PASS\n");
    return 0;
}
