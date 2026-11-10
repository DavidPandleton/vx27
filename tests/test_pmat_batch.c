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
#include "../src/vx27_pmat.h"
#include "../src/vx27_bp3.h"
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
#include "../src/threadpool.h"
#include "../src/quant.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

// isi BP3 sintetik dari pola deterministik (tanpa RNG platform)
static void fill_bp3(vx27_bp3_128 *G, int nblk, unsigned seed) {
    unsigned s = seed;
    for (int k = 0; k < nblk; k++) {
        float sc = 0.02f + (float)(k % 17) * 0.003f;
        uint32_t u; memcpy(&u, &sc, 4);
        int e = ((u >> 23) & 0xFF) - 127 + 15;
        uint16_t hh;
        if (e <= 0) hh = (uint16_t)((u >> 16) & 0x8000);
        else if (e >= 31) hh = (uint16_t)(((u >> 16) & 0x8000) | 0x7C00);
        else hh = (uint16_t)(((u >> 16) & 0x8000) | ((uint32_t)e << 10) | ((u >> 13) & 0x3FF));
        G[k].scale_h = hh;
        for (int b = 0; b < 12; b++) G[k].p[b] = 0;
        for (int i = 0; i < 128; i++) {
            s = s * 1664525u + 1013904223u;
            int q = (int)(s >> 29) - 4; if (q > 3) q = 3; // -4..3
            uint32_t uu = (uint32_t)(q & 7);
            if (uu & 1) G[k].p[0*4+i/32] |= (1u << (i % 32));
            if (uu & 2) G[k].p[1*4+i/32] |= (1u << (i % 32));
            if (uu & 4) G[k].p[2*4+i/32] |= (1u << (i % 32));
        }
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    char err[256] = {0};
    threadpool *tp = tp_create(8, err, sizeof(err));
    CHECK(tp != NULL, "tp 8");

    // 1) sintetik kecil: BP3 + F32 + Q8, B=3 -> bit-identik vs loop single
    {
        const int rows = 64, cols = 512, B = 3;
        vx27_bp3_128 *W4 = malloc((size_t)rows * (cols/128) * sizeof(vx27_bp3_128));
        float *W0 = malloc((size_t)rows * cols * 4);
        block_q8_k *W2 = malloc((size_t)rows * (cols/256) * sizeof(block_q8_k));
        float *xb = malloc((size_t)B * cols * 4);
        float *yb = malloc((size_t)B * rows * 4), *yr = malloc((size_t)B * rows * 4);
        fill_bp3(W4, rows * (cols/128), 12345);
        for (int i = 0; i < rows * cols; i++) W0[i] = ((float)((i * 7919) % 101) / 100.0f - 0.5f);
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < cols/256; k++) {
                float *seg = W0 + (size_t)r * cols + k * 256;
                float am = 0; for (int i = 0; i < 256; i++) { float a = fabsf(seg[i]); if (a > am) am = a; }
                W2[(size_t)r * (cols/256) + k].d = am > 0 ? am / 127.0f : 1.0f;
                for (int i = 0; i < 256; i++)
                    W2[(size_t)r * (cols/256) + k].qs[i] = (int8_t)(am > 0 ? roundf(seg[i] / (am/127.0f)) : 0);
            }
        for (int i = 0; i < B * cols; i++) xb[i] = (float)(i % 13) * 0.05f - 0.3f;
        // BP3
        vx27_pmat_batch(NULL, W4, 4, xb, yb, rows, cols, B);
        for (int b = 0; b < B; b++) vx27_pmat(NULL, W4, 4, xb + (size_t)b*cols, yr + (size_t)b*rows, rows, cols);
        double md = 0; for (int i = 0; i < B*rows; i++) { double d = fabs((double)yb[i]-yr[i]); if (d > md) md = d; }
        printf("  bp3 batch-vs-seq maxdiff=%.3e\n", md);
        CHECK(md == 0.0, "bp3 batch bit-identik");
        // F32
        vx27_pmat_batch(NULL, W0, 0, xb, yb, rows, cols, B);
        for (int b = 0; b < B; b++) vx27_pmat(NULL, W0, 0, xb + (size_t)b*cols, yr + (size_t)b*rows, rows, cols);
        md = 0; for (int i = 0; i < B*rows; i++) { double d = fabs((double)yb[i]-yr[i]); if (d > md) md = d; }
        printf("  f32 batch-vs-seq maxdiff=%.3e\n", md);
        CHECK(md == 0.0, "f32 batch bit-identik");
        // Q8
        vx27_pmat_batch(NULL, W2, 2, xb, yb, rows, cols, B);
        for (int b = 0; b < B; b++) vx27_pmat(NULL, W2, 2, xb + (size_t)b*cols, yr + (size_t)b*rows, rows, cols);
        md = 0; for (int i = 0; i < B*rows; i++) { double d = fabs((double)yb[i]-yr[i]); if (d > md) md = d; }
        printf("  q8 batch-vs-seq maxdiff=%.3e\n", md);
        CHECK(md == 0.0, "q8 batch bit-identik");
        // B=1 degenerasi
        vx27_pmat_batch(NULL, W4, 4, xb, yb, rows, cols, 1);
        vx27_pmat(NULL, W4, 4, xb, yr, rows, cols);
        md = 0; for (int i = 0; i < rows; i++) { double d = fabs((double)yb[i]-yr[i]); if (d > md) md = d; }
        CHECK(md == 0.0, "B=1 == single");
        free(W4); free(W0); free(W2); free(xb); free(yb); free(yr);
    }

    // 2) bobot REAL (gate_proj layer-0, 512 baris pertama) B=3: exact + bench
    {
        vx27_entry ec[4096]; int nc = 0;
        if (vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof(err)) != 0) { printf("SKIP real: %s\n", err); }
        else {
            vx27_map mc = {0};
            if (vx27_mmap_open("build/vx27_core.vx27", &mc, err, sizeof(err)) != 0) { printf("SKIP real: %s\n", err); }
            else {
                int bi = -1;
                for (int i = 0; i < nc; i++)
                    if (!strcmp(ec[i].name, "model.language_model.layers.0.mlp.gate_proj.weight")) { bi = i; break; }
                CHECK(bi >= 0, "gate_proj ketemu");
                uint64_t bb = 0; const void *W = vx27_mmap_blob(&mc, bi, &bb);
                CHECK(W && bb > 0, "mmap bobot");
                const int rows = 512, cols = 5120, B = 3;
                // W row-major BP3: baris r mulai di blob + r*40 grup
                float *xb = malloc((size_t)B * cols * 4);
                float *yb = malloc((size_t)B * rows * 4), *yr = malloc((size_t)B * rows * 4);
                for (int i = 0; i < B * cols; i++) xb[i] = (float)(i % 13) * 0.05f - 0.3f;
                vx27_pmat_batch(tp, W, 4, xb, yb, rows, cols, B);
                for (int b = 0; b < B; b++) vx27_pmat(tp, W, 4, xb + (size_t)b*cols, yr + (size_t)b*rows, rows, cols);
                double md = 0;
                for (int i = 0; i < B*rows; i++) { double d = fabs((double)yb[i]-yr[i]); if (d > md) md = d; }
                printf("  real BP3 512x5120 B=3 batch-vs-seq maxdiff=%.3e\n", md);
                CHECK(md == 0.0, "real batch bit-identik");
                double t0 = now_ms();
                for (int it = 0; it < 20; it++)
                    for (int b = 0; b < B; b++) vx27_pmat(tp, W, 4, xb + (size_t)b*cols, yr + (size_t)b*rows, rows, cols);
                double t1 = now_ms();
                double t2 = now_ms();
                for (int it = 0; it < 20; it++) vx27_pmat_batch(tp, W, 4, xb, yb, rows, cols, B);
                double t3 = now_ms();
                double ts = (t1-t0)/20, tb = (t3-t2)/20;
                printf("  bench 512x5120 B=3: 3x-seq %.2f ms vs batch %.2f ms (%.2fx)\n", ts, tb, ts/(tb+1e-9));
                CHECK(tb <= ts, "batch tidak lebih lambat");
                free(xb); free(yb); free(yr);
                vx27_mmap_close(&mc);
            }
        }
    }

    tp_free(tp);
    if (fails) { printf("test_pmat_batch: %d FAIL\n", fails); return 1; }
    printf("test_pmat_batch: PASS\n");
    return 0;
}
