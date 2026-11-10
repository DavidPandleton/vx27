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
static double now_ms(void) { return (double)clock() * 1000.0 / CLOCKS_PER_SEC; }
#endif
#include "../src/vx27_deltanet.h"
#include "../src/vx27_config.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    CHECK(vx27_deltanet_state_per_layer() == 16 * 128 * 128, "state per layer 262144");
    CHECK(vx27_deltanet_state_total() == 262144 * 48, "state total 48 layers");
    double mb = vx27_deltanet_state_mb(2);
    CHECK(mb > 24.0 && mb < 26.0, "state ~25MB BF16 fits L3-adjacent");

    int hd = 128;
    float *S1 = (float *)calloc((size_t)hd * hd, 4);
    float *S2 = (float *)calloc((size_t)hd * hd, 4);
    float *S3 = (float *)calloc((size_t)hd * hd, 4);
    float *k = (float *)malloc((size_t)hd * 4);
    float *v = (float *)malloc((size_t)hd * 4);
    float *q = (float *)malloc((size_t)hd * 4);
    float *o1 = (float *)malloc((size_t)hd * 4);
    float *o2 = (float *)malloc((size_t)hd * 4);
    float *o3 = (float *)malloc((size_t)hd * 4);
    for (int i = 0; i < hd; i++) { k[i] = (float)(i % 7) * 0.1f - 0.3f; v[i] = (float)(i % 5) * 0.2f; q[i] = (float)(i % 3) * 0.15f - 0.1f; }

    vx27_deltanet_update_ref(S1, k, v, 0.9f, hd);
    vx27_deltanet_update_tiled(S2, k, v, 0.9f, hd);
    vx27_deltanet_update_avx2(S3, k, v, 0.9f, hd);
    float md = 0;
    for (int i = 0; i < hd * hd; i++) { float d = fabsf(S1[i] - S2[i]); if (d > md) md = d; }
    CHECK(md < 1e-5f, "update tiled matches ref");
    md = 0;
    for (int i = 0; i < hd * hd; i++) { float d = fabsf(S1[i] - S3[i]); if (d > md) md = d; }
    printf("  avx2 update maxdiff vs ref: %.3e\n", (double)md);
    CHECK(md < 1e-4f, "update avx2 matches ref");

    vx27_deltanet_retrieve_ref(S1, q, o1, hd);
    vx27_deltanet_retrieve_tiled(S2, q, o2, hd);
    vx27_deltanet_retrieve_avx2(S3, q, o3, hd);
    md = 0;
    for (int i = 0; i < hd; i++) { float d = fabsf(o1[i] - o2[i]); if (d > md) md = d; }
    CHECK(md < 1e-5f, "retrieve tiled matches ref");
    md = 0;
    for (int i = 0; i < hd; i++) { float d = fabsf(o1[i] - o3[i]); if (d > md) md = d; }
    printf("  avx2 retrieve maxdiff vs ref: %.3e\n", (double)md);
    CHECK(md < 1e-4f, "retrieve avx2 matches ref");

    // Bench: 2000 update+retrieve on hd=128, single head.
    // FLOPs approx: update 2*hd*hd + retrieve 2*hd*hd = 65536 FLOP/iter.
    const int iters = 2000;
    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        vx27_deltanet_update_avx2(S3, k, v, 0.99f, hd);
        vx27_deltanet_retrieve_avx2(S3, q, o3, hd);
    }
    double t1 = now_ms();
    double ms = t1 - t0, us_per = ms * 1000.0 / iters;
    double gflops = (65536.0 * iters / 1e9) / (ms / 1000.0);
    printf("  bench avx2 hd=128 x%d: %.2f ms total, %.2f us/iter, %.2f GFLOPS\n", iters, ms, us_per, gflops);
    double t0r = now_ms();
    for (int it = 0; it < iters; it++) {
        vx27_deltanet_update_ref(S1, k, v, 0.99f, hd);
        vx27_deltanet_retrieve_ref(S1, q, o1, hd);
    }
    double t1r = now_ms();
    printf("  bench ref  hd=128 x%d: %.2f ms total (%.2fx vs avx2)\n", iters, t1r - t0r, (t1r - t0r) / (ms > 0 ? ms : 1));
    // checksum to prevent dead-code elimination
    double chk = 0; for (int i = 0; i < hd; i++) chk += o3[i] + o1[i];
    printf("  checksum %.6f\n", chk);
    CHECK(ms < (t1r - t0r) * 2.0 + 50.0, "avx2 bench ran");

    free(S1); free(S2); free(S3); free(k); free(v); free(q); free(o1); free(o2); free(o3);
    if (fails) { printf("test_deltanet: %d FAIL\n", fails); return 1; }
    printf("test_deltanet: PASS\n");
    return 0;
}
