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
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

// Ukur min dari R repetisi (buang noise warmup).
static double bench_matvec(threadpool *tp, const void *W, uint32_t dtype,
                           const float *x, float *y, int rows, int cols, int R) {
    double best = 1e18;
    for (int r = 0; r < R + 1; r++) {
        double t0 = now_ms();
        vx27_pmat(tp, W, dtype, x, y, rows, cols);
        double t1 = now_ms();
        if (r > 0 && t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

static double wbytes(uint32_t dtype, int rows, int cols) {
    if (dtype == 0) return (double)rows * cols * 4;
    if (dtype == 2) return (double)(rows * (cols / 256)) * 292.0;
    return (double)(rows * (cols / 128)) * 50.0; // BP3_128
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    char err[512] = {0};
    threadpool *tp = tp_create(8, err, sizeof err);
    CHECK(tp != NULL, "tp 8");

    vx27_entry ec[4096]; int nc = 0;
    CHECK(vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof err) == 0, "list core");
    vx27_map mc = {0};
    CHECK(vx27_mmap_open("build/vx27_core.vx27", &mc, err, sizeof err) == 0, "mmap core");
    vx27_entry eh[4096]; int nh = 0;
    CHECK(vx27_list("build/vx27_head.vx27", eh, 4096, &nh, err, sizeof err) == 0, "list head");
    vx27_map mh = {0};
    CHECK(vx27_mmap_open("build/vx27_head.vx27", &mh, err, sizeof err) == 0, "mmap head");

    const void *W = NULL; uint64_t bb = 0;
    const char *want = NULL;
    float *x = malloc(17408*4), *y = malloc(248320*4);
    for (int i = 0; i < 17408; i++) x[i] = (float)(i % 13) * 0.05f - 0.3f;

    struct { const char *nm; int rows, cols; double ms; double gb; } tab[8];
    int nt = 0;
    struct { const char *nm; int ishead; } needs[] = {
        {"model.language_model.layers.0.mlp.gate_proj.weight", 0},
        {"model.language_model.layers.0.mlp.down_proj.weight", 0},
        {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 0},
        {"model.language_model.layers.0.linear_attn.out_proj.weight", 0},
        {"model.language_model.layers.3.self_attn.q_proj.weight", 0},
        {"model.language_model.layers.3.self_attn.o_proj.weight", 0},
        {"lm_head.weight", 1},
    };
    for (size_t k = 0; k < sizeof(needs)/sizeof(needs[0]); k++) {
        want = needs[k].nm;
        W = NULL;
        if (!needs[k].ishead) {
            for (int i = 0; i < nc; i++)
                if (!strcmp(ec[i].name, want)) { W = vx27_mmap_blob(&mc, i, &bb); break; }
        } else {
            for (int i = 0; i < nh; i++)
                if (!strcmp(eh[i].name, want)) { W = vx27_mmap_blob(&mh, i, &bb); break; }
        }
        char msg[192]; snprintf(msg, sizeof msg, "tensor %s ketemu", want);
        CHECK(W != NULL, msg);
        if (!W) continue;
        // rows/cols dari entri
        int rows = 0, cols = 0;
        if (!needs[k].ishead) {
            for (int i = 0; i < nc; i++) if (!strcmp(ec[i].name, want)) { rows = (int)ec[i].ne0; cols = (int)ec[i].ne1; break; }
        } else {
            for (int i = 0; i < nh; i++) if (!strcmp(eh[i].name, want)) { rows = (int)eh[i].ne0; cols = (int)eh[i].ne1; break; }
        }
        double ms = bench_matvec(tp, W, 4, x, y, rows, cols, 3);
        double gb = wbytes(4, rows, cols) / 1e9;
        printf("  %-55s [%6d x %5d] %7.2f ms %5.3f GB %5.2f GB/s\n", want, rows, cols, ms, gb, gb / (ms/1000.0 + 1e-9));
        tab[nt].nm = want; tab[nt].rows = rows; tab[nt].cols = cols; tab[nt].ms = ms; tab[nt].gb = gb; nt++;
        double c = 0; for (int i = 0; i < rows && i < 4096; i += 1) c += y[i];
        char m2[192]; snprintf(m2, sizeof m2, "finite %s", want);
        CHECK(c == c && (c != 0.0), m2);
    }

    // Komposisi 1 forward: 48x linear (qkv+z(6144)+a/b+out) + 16x full (q+o) + 64x MLP(gate+up+down) + 1x lm_head.
    // z/a/b kecil -> aproksimasi dgn qkv*0.6; k/v full kecil -> abaikan (disebut eksplisit).
    double mlp1 = 0, lin1 = 0, full1 = 0, lm1 = 0;
    double mlp_gb = 0, lin_gb = 0, full_gb = 0, lm_gb = 0;
    for (int i = 0; i < nt; i++) {
        if (strstr(tab[i].nm, "mlp.gate_proj")) { mlp1 += tab[i].ms * 2; mlp_gb += tab[i].gb * 2; } // gate+up
        if (strstr(tab[i].nm, "mlp.down_proj")) { mlp1 += tab[i].ms; mlp_gb += tab[i].gb; }
        if (strstr(tab[i].nm, "in_proj_qkv")) { lin1 += tab[i].ms * 1.6; lin_gb += tab[i].gb * 1.6; } // qkv + z/a/b ~0.6x
        if (strstr(tab[i].nm, "linear_attn.out_proj")) { lin1 += tab[i].ms; lin_gb += tab[i].gb; }
        if (strstr(tab[i].nm, "self_attn.q_proj")) { full1 += tab[i].ms; full_gb += tab[i].gb; }
        if (strstr(tab[i].nm, "self_attn.o_proj")) { full1 += tab[i].ms; full_gb += tab[i].gb; }
        if (!strcmp(tab[i].nm, "lm_head.weight")) { lm1 = tab[i].ms; lm_gb = tab[i].gb; }
    }
    double pred = 48*lin1 + 16*full1 + 64*mlp1 + lm1;
    double gb_fwd = 48*lin_gb + 16*full_gb + 64*mlp_gb + lm_gb;
    printf("  prediksi/forward: lin48=%.0fms full16=%.0fms mlp64=%.0fms lm=%.0fms TOTAL=%.0fms (~%.1fGB)\n",
           48*lin1, 16*full1, 64*mlp1, lm1, pred, gb_fwd);
    printf("  acuan terukur test_qfwd64: ~1500ms/pass; test_mtp loop: ~930ms/forward\n");
    CHECK(pred > 300, "komposisi substansial (>300ms)");
    double mlp_share = 64*mlp1 / (pred + 1e-9);
    printf("  pangsa MLP: %.0f%%\n", mlp_share*100);
    CHECK(mlp_share > 0.40, "MLP dominan (>40%) -> memory-bound bobot");
    CHECK(lin1 > 0 && full1 > 0 && lm1 > 0, "semua komponen terukur");

    free(x); free(y);
    vx27_mmap_close(&mc); vx27_mmap_close(&mh);
    tp_free(tp);
    if (fails) { printf("test_breakdown: %d FAIL\n", fails); return 1; }
    printf("test_breakdown: PASS\n");
    return 0;
}
