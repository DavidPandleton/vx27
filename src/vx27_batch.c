#include "vx27_batch.h"
#include "quant.h"
#include "quant_avx2.h"
#include "vx27_gpu.h"
#include "vx27_gpu_persist.h"
#include "vx27_pmat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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

typedef struct { const block_q4_k *W; const block_q8_k *Q8; float *Y; int rows, B; size_t nb, stride, rb; } bj;
static int g_best = -1;
int vx27_batch_best(void) { return g_best; }
static int g_q8best = -1;
int vx27_batch_q8_best(void) { return g_q8best; }
static void worker(void *a, int tid, int nth) {
    bj *j = (bj*)a;
    int rs = (int)((long long)j->rows * tid / nth), re = (int)((long long)j->rows * (tid+1) / nth);
    for (int r = rs; r < re; r++) {
        float o[8];
        dot_q4_k_q8_k_x8((const block_q4_k*)((const uint8_t*)j->W + (size_t)r*j->rb),
                         j->Q8, j->stride, j->nb, o);
        for (int t = 0; t < j->B; t++) j->Y[(size_t)t * j->rows + r] = o[t];
    }
}

int vx27_matvec_batch(int gpu_ok, threadpool *tp,
    const void *Wq4, const float *xB, float *yBx,
    int rows, int cols, int B, double *ms) {
    // Autotune + persisten: pertama kali benchmark v0/v1-persist/v2, cache tercepat.
    static vx27_gpu_ctx *pctx = NULL;
    static int p_rows = 0, p_cols = 0, best = -1;
    static const void *p_W = NULL;
    g_best = best;
    if (gpu_ok && rows >= 512 && B == 8 && (cols % 256) == 0) {
        int need_tune = (best < 0 || rows != p_rows || cols != p_cols || Wq4 != p_W);
        double m0 = 1e9, m1 = 1e9, m2 = 1e9;
        float *yrow = (float*)malloc((size_t)rows * B * 4);
        char err[256] = {0};
        if (need_tune) {
            // v0 non-persisten
            float *tmp = malloc((size_t)rows * B * 4);
            if (vx27_gpu_q4_bench_full(Wq4, xB, rows, cols, B, tmp, &m0, err, sizeof(err)) != 0) m0 = 1e9;
            // v1 persisten (upload sekali)
            if (!pctx || rows != p_rows || cols != p_cols || Wq4 != p_W) {
                // reset ctx bila bentuk/bobot berubah (bocor handle lama: uji saja)
                if (vx27_gpu_ctx_init(&pctx, Wq4, rows, cols, err, sizeof(err)) == 0) { p_rows = rows; p_cols = cols; p_W = Wq4; }
            }
            if (pctx) {
                if (vx27_gpu_ctx_dispatch(pctx, xB, tmp, &m1, err, sizeof(err)) != 0) m1 = 1e9;
            }
            // v2 non-persisten
            if (vx27_gpu_q4_bench_tiled8_full(Wq4, xB, rows, cols, B, tmp, &m2, err, sizeof(err)) != 0) m2 = 1e9;
            free(tmp);
            best = (m0 <= m1 && m0 <= m2) ? 0 : ((m1 <= m2) ? 1 : 2);
            g_best = best;
            p_rows = rows; p_cols = cols; p_W = Wq4;
        }
        double gms = 0; int rc = -1;
        if (best == 1 && pctx) rc = vx27_gpu_ctx_dispatch(pctx, xB, yrow, &gms, err, sizeof(err));
        else if (best == 2) rc = vx27_gpu_q4_bench_tiled8_full(Wq4, xB, rows, cols, B, yrow, &gms, err, sizeof(err));
        else rc = vx27_gpu_q4_bench_full(Wq4, xB, rows, cols, B, yrow, &gms, err, sizeof(err));
        if (rc == 0) {
            for (int r = 0; r < rows; r++)
                for (int t = 0; t < B; t++)
                    yBx[(size_t)t * rows + r] = yrow[(size_t)r * B + t];
            if (ms) *ms = gms;
            free(yrow);
            return 1;
        }
        free(yrow);
    }
    double t0 = now_ms();
    size_t nb = (size_t)cols / 256;
    block_q8_k *Q8 = (block_q8_k*)malloc(nb * B * sizeof(block_q8_k));
    for (int t = 0; t < B; t++) quantize_q8_k(xB + (size_t)t * cols, Q8 + (size_t)t * nb, (int)nb);
    bj j = {(const block_q4_k*)Wq4, Q8, yBx, rows, B, nb, nb, nb * sizeof(block_q4_k)};
    if (tp && rows >= 512) tp_run(tp, worker, &j); else worker(&j, 0, 1);
    double t1 = now_ms();
    if (ms) *ms = t1 - t0;
    free(Q8);
    return 0;
}

// Batch Q8 (dtype=2): autotune CPU-8T (pmat AVX2) vs GPU-Q8 sekali per bentuk.
int vx27_matvec_batch_q8(threadpool *tp,
    const void *Wq8, const float *xB, float *yBx,
    int rows, int cols, int B, double *ms, char *err, size_t errlen) {
    if (B < 1 || B > 16 || (cols % 256)) return -2;
    if (g_q8best < 0) {
        // autotune: CPU-8T (pmat AVX2) vs GPU-Q8, sekali per proses
        float *yq = malloc((size_t)rows * B * 4);
        double t0 = now_ms();
        for (int t = 0; t < B; t++) {
            float *one = malloc((size_t)rows * 4);
            vx27_pmat(tp, Wq8, 2, xB + (size_t)t * cols, one, rows, cols);
            for (int r = 0; r < rows; r++) yq[(size_t)t * rows + r] = one[r];
            free(one);
        }
        double tcpu = now_ms() - t0;
        float *yg = malloc((size_t)rows * B * 4);
        double tg = 1e18; char e2[256] = {0};
        if (vx27_gpu_q8_bench_tiled_full(Wq8, xB, rows, cols, B, yg, &tg, e2, sizeof(e2)) != 0) tg = 1e18;
        g_q8best = (tg < tcpu) ? 1 : 0;
        if (err && errlen) snprintf(err, errlen, "autotune cpu=%.1fms gpu=%.1fms -> %s", tcpu, tg, g_q8best ? "GPU" : "CPU");
        free(yq); free(yg);
        if (ms) *ms = (g_q8best == 1) ? tg : tcpu;
    }
    if (g_q8best == 1) {
        float *yrow = malloc((size_t)rows * B * 4);
        double gms = 0; char e3[256] = {0};
        int rc = vx27_gpu_q8_bench_tiled_full(Wq8, xB, rows, cols, B, yrow, &gms, e3, sizeof(e3));
        if (rc == 0) {
            for (int r = 0; r < rows; r++)
                for (int t = 0; t < B; t++) yBx[(size_t)t * rows + r] = yrow[(size_t)r * B + t];
            if (ms) *ms = gms;
            free(yrow);
            return 1;
        }
        free(yrow);
        g_q8best = 0; // fallback permanen ke CPU bila GPU gagal
    }
    double t0 = now_ms();
    for (int t = 0; t < B; t++) {
        float *one = malloc((size_t)rows * 4);
        vx27_pmat(tp, Wq8, 2, xB + (size_t)t * cols, one, rows, cols);
        for (int r = 0; r < rows; r++) yBx[(size_t)t * rows + r] = one[r];
        free(one);
    }
    double t1 = now_ms();
    if (ms) *ms = t1 - t0;
    return 0;
}
