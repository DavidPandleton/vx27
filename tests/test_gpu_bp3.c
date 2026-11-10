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
#include "../src/vx27_gpu_bp3.h"
#include "../src/vx27_pmat.h"
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    char err[512] = {0};
    vx27_entry ec[4096]; int nc = 0;
    CHECK(vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof(err)) == 0, "list core");
    vx27_map mc = {0};
    CHECK(vx27_mmap_open("build/vx27_core.vx27", &mc, err, sizeof(err)) == 0, "mmap core");
    int bi = -1;
    for (int i = 0; i < nc; i++)
        if (!strcmp(ec[i].name, "model.language_model.layers.52.mlp.gate_proj.weight")) { bi = i; break; }
    if (bi < 0)
        for (int i = 0; i < nc; i++)
            if (!strcmp(ec[i].name, "model.language_model.layers.0.mlp.gate_proj.weight")) { bi = i; break; }
    CHECK(bi >= 0, "gate BP3 ketemu");
    uint64_t bb = 0; const void *W = vx27_mmap_blob(&mc, bi, &bb);
    CHECK(W && bb > 0, "blob readable");
    int rows = (int)ec[bi].ne0, cols = (int)ec[bi].ne1;
    printf("  tensor %s [%d x %d] %.1f MB\n", ec[bi].name, rows, cols, (double)bb/1e6);

    double t0 = now_ms();
    vx27_bctx *ctx = NULL;
    CHECK(vx27_bctx_init(&ctx, W, rows, cols, err, sizeof(err)) == 0, "upload device-resident sekali");
    double t1 = now_ms();
    printf("  upload %.1f ms\n", t1 - t0);

    threadpool *tp = tp_create(8, err, sizeof(err));
    const int B = 8;
    float *x = malloc((size_t)cols * B * 4);
    for (int i = 0; i < cols * B; i++) x[i] = (float)(i % 29) * 0.03f - 0.4f;

    // referensi CPU pmat 8-thread, per token
    float *yc = malloc((size_t)rows * B * 4);
    t0 = now_ms();
    for (int t = 0; t < B; t++) {
        float *one = malloc((size_t)rows * 4);
        vx27_pmat(tp, W, 4, x + (size_t)t * cols, one, rows, cols);
        for (int r = 0; r < rows; r++) yc[(size_t)t * rows + r] = one[r];
        free(one);
    }
    double tcpu = now_ms() - t0;
    printf("  cpu-8T B=%d: %.1f ms\n", B, tcpu);

    // GPU: dispatch per B (1 dan 8), bandingkan dgn CPU
    for (int B2 = 1; B2 <= B; B2 *= 8) {
        float *yg = malloc((size_t)rows * (size_t)B2 * 4);
        double tg = 0;
        int rc = vx27_bctx_dispatch(ctx, x, B2, yg, &tg, err, sizeof(err));
        CHECK(rc == 0, "dispatch ok");
        // layout GPU row-major [r*B+t] vs CPU t-major [t*rows+r]: bandingkan per elemen
        double md = 0, denom = 0, num = 0;
        for (int r = 0; r < rows; r += 7) for (int t = 0; t < B2; t++) {
            float a = yg[(size_t)r * B2 + t], b = yc[(size_t)t * rows + r];
            double d = fabs((double)a - (double)b);
            if (d > md) md = d;
            denom += (double)b * b; num += d * d;
        }
        double nrmse = sqrt(num / (denom + 1e-12));
        double gv = (double)rows * cols * B2 / (tg / 1000.0) / 1e9;
        printf("  B=%d gpu %.1f ms (%.1f Gval/s) maxdiff=%.4f nrmse=%.4f\n", B2, tg, gv, md, nrmse);
        CHECK(md < 0.5f && nrmse < 0.02, "GPU MATCH cpu (toleransi)");
        free(yg);
    }
    vx27_bctx_free(ctx);
    free(x); free(yc); tp_free(tp);
    vx27_mmap_close(&mc);
    if (fails) { printf("test_gpu_bp3: %d FAIL\n", fails); return 1; }
    printf("test_gpu_bp3: PASS\n");
    return 0;
}
