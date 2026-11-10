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
#include "../src/vx27_gpu.h"
#include "../src/vx27_batch.h"
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
#include "../src/vx27_pmat.h"
#include "../src/quant.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // pastikan .vx27 Q8 real-dim ada (forge ulang bila perlu; idempoten, ~1 dtk)
    int rc = system("build\\vx_forge.exe forge --in build\\sample_real5120.safetensors --out-core build\\real_core.vx27 --out-head build\\real_head.vx27 --bits 8 --split-at 1");
    CHECK(rc == 0, "forge Q8 real-dim");
    char err[512] = {0};
    vx27_entry ec[8]; int nc = 0;
    CHECK(vx27_list("build/real_core.vx27", ec, 8, &nc, err, sizeof(err)) == 0, "list Q8");
    vx27_map mc = {0};
    CHECK(vx27_mmap_open("build/real_core.vx27", &mc, err, sizeof(err)) == 0, "mmap Q8");
    // cari tensor gate [17408,5120] Q8
    int bi = -1;
    for (int i = 0; i < nc; i++) if (ec[i].ne0 == 17408 && ec[i].ne1 == 5120 && ec[i].dtype == 2) bi = i;
    CHECK(bi >= 0, "gate Q8 ketemu");
    uint64_t bb = 0; const void *W = vx27_mmap_blob(&mc, bi, &bb);
    CHECK(W && bb > 0, "blob readable");
    const int rows = 17408, cols = 5120, B = 8;
    float *x = malloc((size_t)cols * B * 4);
    unsigned seed = 11;
    for (int k = 0; k < B * cols; k++) { seed = seed*1664525u+1013904223u; x[k] = ((float)(seed>>8)/16777216.0f)*2.0f-1.0f; }

    // CPU ref via pmat (Q8 langsung, tanpa ekspansi): per token.
    // Dua pembanding: single-thread (baseline) dan 8-thread (adil vs GPU).
    threadpool *tpp = tp_create(8, err, sizeof(err));
    CHECK(tpp != NULL, "tp 8");
    float *ycpu = malloc((size_t)rows * B * 4);
    double t0 = now_ms();
    for (int t = 0; t < B; t++) {
        float *one = malloc((size_t)rows * 4);
        vx27_pmat(NULL, W, 2, x + (size_t)t * cols, one, rows, cols);
        for (int r = 0; r < rows; r++) ycpu[(size_t)r * B + t] = one[r];
        free(one);
    }
    double t1 = now_ms();
    printf("  cpu Q8 %d x %d B=%d single-thread: %.1f ms\n", rows, cols, B, t1 - t0);
    float *ycpu8 = malloc((size_t)rows * B * 4);
    double t8 = now_ms();
    for (int t = 0; t < B; t++) {
        float *one = malloc((size_t)rows * 4);
        vx27_pmat(tpp, W, 2, x + (size_t)t * cols, one, rows, cols);
        for (int r = 0; r < rows; r++) ycpu8[(size_t)r * B + t] = one[r];
        free(one);
    }
    double t9 = now_ms();
    printf("  cpu Q8 8-thread: %.1f ms\n", t9 - t8);
    tp_free(tpp);

    // GPU
    float *ygpu = malloc((size_t)rows * B * 4);
    double gms = 0;
    rc = vx27_gpu_q8_bench_tiled_full(W, x, rows, cols, B, ygpu, &gms, err, sizeof(err));
    printf("  gpu %.2f ms y0=%.6f cpu0=%.6f %s\n", gms, ygpu[0], ycpu[0], err);
    CHECK(rc == 0, "gpu Q8 MATCH (toleransi internal)");
    double md = 0;
    for (size_t i = 0; i < (size_t)rows * B; i += 7) { double d = fabs((double)ygpu[i]-ycpu[i]); if (d > md) md = d; }
    printf("  maxdiff gpu-vs-cpu %.4f (rel %.4f)\n", md, md / (fabs(ycpu[0]) + 1.0));
    CHECK(md < 0.1, "cross-check vs pmat CPU");
    double gvals = (double)rows * cols * B / (gms / 1000.0) / 1e9;
    printf("  %.1f Gval/s gpu vs %.1f ms cpu-1T / %.1f ms cpu-8T\n", gvals, t1 - t0, t9 - t8);
    free(x); free(ycpu); free(ycpu8); free(ygpu);
    // integrasi batch autotune: panggil 2x, pastikan deterministik + lapor pemenang
    {
        threadpool *tp = tp_create(8, err, sizeof(err));
        float *yb1 = malloc((size_t)rows * B * 4), *yb2 = malloc((size_t)rows * B * 4);
        double m1 = 0, m2 = 0;
        float *xx = malloc((size_t)cols * B * 4);
        unsigned s2 = 11;
        for (int k = 0; k < B * cols; k++) { s2 = s2*1664525u+1013904223u; xx[k] = ((float)(s2>>8)/16777216.0f)*2.0f-1.0f; }
        int p1 = vx27_matvec_batch_q8(tp, W, xx, yb1, rows, cols, B, &m1, err, sizeof(err));
        int p2 = vx27_matvec_batch_q8(tp, W, xx, yb2, rows, cols, B, &m2, err, sizeof(err));
        double md = 0; for (size_t i = 0; i < (size_t)rows*B; i++) { double d = fabs((double)yb1[i]-yb2[i]); if (d > md) md = d; }
        printf("  batch autotune: path=%d/%d (%s) %.1f/%.1f ms maxdiff=%.3e\n", p1, p2,
            vx27_batch_q8_best() ? "GPU" : "CPU", m1, m2, md);
        CHECK(p1 == p2 && md == 0.0, "batch autotune deterministik");
        free(xx); free(yb1); free(yb2); tp_free(tp);
    }
    vx27_mmap_close(&mc);
    if (fails) { printf("test_gpu_q8: %d FAIL\n", fails); return 1; }
    printf("test_gpu_q8: PASS\n");
    return 0;
}
