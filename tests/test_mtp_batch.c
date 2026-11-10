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
#include "../src/vx27_run.h"
#include "../src/vx27_mtp.h"
#include "../src/vx27_bp3.h"
#include "../src/tokenizer.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

static uint32_t argmax248k(const float *lg) {
    int b = 0; for (int i = 1; i < 248320; i++) if (lg[i] > lg[b]) b = i;
    return (uint32_t)b;
}

// prefill sekuensial n_ids token; h_out[0..n-1] hidden per posisi, np akhir = n
static int prefill_seq(vx27_engine *E, uint32_t *ids, int n, float *h_out, char *err, size_t el) {
    float x[5120], h[5120];
    for (int i = 0; i < n; i++) {
        if (vx27_engine_embed(E, ids[i], x, err, el) != 0) return -1;
        if (vx27_engine_step(E, x, h, i, err, el) != 0) return -1;
        memcpy(h_out + (size_t)i*5120, h, 5120*4);
    }
    return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    char err[1024] = {0};
    tokenizer *tok = NULL;
    CHECK(tok_load_bin(&tok, "build/tok_vocab.bin", "build/tok_types.bin",
                       "build/tok_merges.bin", err, sizeof err) == 0, "load tokenizer");
    if (!tok) { printf("  err: %s\n", err); return 1; }
    const char *prompt = "Halo dunia apa kabar hari ini";
    uint32_t *ids = NULL; size_t n_ids = 0;
    CHECK(tok_encode(tok, (const uint8_t*)prompt, strlen(prompt), &ids, &n_ids, err, sizeof err) == 0, "encode");
    printf("  prompt ids: %zu\n", n_ids);

    vx27_engine *E = NULL;
    int max_t = (int)n_ids + 16;
    CHECK(vx27_engine_open(&E, "build/vx27_core.vx27", "build/vx27_head.vx27", max_t, err, sizeof err) == 0, "engine open");
    if (!E) { printf("  err: %s\n", err); return 1; }
    vx27_qtab *tab = vx27_engine_qtab(E);
    threadpool *tp = vx27_engine_tp(E);

    float *hsave = malloc(sizeof(float[64][5120]));
    CHECK(prefill_seq(E, ids, (int)n_ids, hsave, err, sizeof err) == 0, "prefill seq");
    int npos = (int)n_ids;

    // ---------- Part 1: batch B=3 vs 3x sequential, bit-identik ----------
    {
        uint32_t ext[3] = { 30, 271, 248068 };
        float (*xs)[5120] = malloc(sizeof(float[3][5120]));
        float (*hs_seq)[5120] = malloc(sizeof(float[3][5120]));
        float (*hs_bat)[5120] = malloc(sizeof(float[3][5120]));
        for (int b = 0; b < 3; b++) vx27_engine_embed(E, ext[b], xs[b], err, sizeof err);
        // sekuensial dari state prefill (lanjutkan engine)
        float h[5120];
        for (int b = 0; b < 3; b++) {
            vx27_engine_step(E, xs[b], h, npos + b, err, sizeof err);
            memcpy(hs_seq[b], h, 5120*4);
        }
        // ulangi dari state sama: reset + prefill ulang (deterministik)
        vx27_engine_reset(E);
        CHECK(prefill_seq(E, ids, (int)n_ids, hsave, err, sizeof err) == 0, "prefill ulang");
        CHECK(vx27_engine_batch_step(E, xs[0], hs_bat[0], 3, npos, NULL, NULL, err, sizeof err) == 0, "batch step B=3");
        double md = 0;
        for (int b = 0; b < 3; b++)
            for (int i = 0; i < 5120; i += 1) {
                double d = fabs((double)hs_bat[b][i] - hs_seq[b][i]);
                if (d > md) md = d;
            }
        printf("  batch-vs-seq B=3 maxdiff=%.3e\n", md);
        CHECK(md == 0.0, "batch forward bit-identik");
        // logits batch vs single
        float *lgB = malloc(sizeof(float[3][248320])), *lgS = malloc(248320*4);
        uint32_t ab[3] = {0};
        vx27_engine_logits_batch(E, hs_bat[0], lgB, ab, 3, err, sizeof err);
        double mdl = 0; int samea = 1;
        for (int b = 0; b < 3; b++) {
            uint32_t a = vx27_engine_logits(E, hs_seq[b], lgS, err, sizeof err);
            if (a != ab[b]) samea = 0;
            for (int i = 0; i < 248320; i += 7) {
                double d = fabs((double)lgB[(size_t)b*248320+i] - lgS[i]);
                if (d > mdl) mdl = d;
            }
        }
        printf("  logits batch-vs-seq maxdiff=%.3e same-arg=%d\n", mdl, samea);
        CHECK(mdl == 0.0 && samea, "logits batch bit-identik");
        free(lgB); free(lgS);
    }

    // ---------- Part 2: ckpt commit parsial m=1, lanjut sekuensial ----------
    {
        vx27_engine_reset(E);
        CHECK(prefill_seq(E, ids, (int)n_ids, hsave, err, sizeof err) == 0, "prefill ulang-2");
        uint32_t ext[3] = { 30, 271, 248068 };
        float (*xs)[5120] = malloc(sizeof(float[3][5120]));
        float (*hb)[5120] = malloc(sizeof(float[3][5120]));
        for (int b = 0; b < 3; b++) vx27_engine_embed(E, ext[b], xs[b], err, sizeof err);
        size_t SONE = (size_t)48*VX27_SSTATE_PER_LAYER, CONE = (size_t)48*VX27_CSTATE_PER_LAYER;
        float *Sckpt = malloc(SONE*3*4), *Cckpt = malloc(CONE*3*4);
        CHECK(vx27_engine_batch_step(E, xs[0], hb[0], 3, npos, Sckpt, Cckpt, err, sizeof err) == 0, "batch+ckpt");
        // commit parsial m=1: kembalikan state setelah token-1 via API?
        // (engine tidak ekspos Sall; uji via perilaku: reset+prefill+seq-2token
        //  harus sama dgn hb[1], lalu commit diuji di Part 3 lewat loop.)
        // Di sini verifikasi ckpt konsisten: state in-place (setelah token-2)
        // dilanjutkan 1 step sekuensial == batch ulang 4-token? Sederhanakan:
        // bandingkan hb[1] vs jalur sekuensial murni.
        vx27_engine *E2 = NULL;
        CHECK(vx27_engine_open(&E2, "build/vx27_core.vx27", "build/vx27_head.vx27", max_t, err, sizeof err) == 0, "engine-2 open");
        float *hs2 = malloc(sizeof(float[64][5120]));
        prefill_seq(E2, ids, (int)n_ids, hs2, err, sizeof err);
        float x[5120], h[5120];
        vx27_engine_embed(E2, ext[0], x, err, sizeof err); vx27_engine_step(E2, x, h, npos, err, sizeof err);
        vx27_engine_embed(E2, ext[1], x, err, sizeof err); vx27_engine_step(E2, x, h, npos+1, err, sizeof err);
        double md = 0;
        for (int i = 0; i < 5120; i++) { double d = fabs((double)hb[1][i]-h[i]); if (d > md) md = d; }
        printf("  ckpt-token1 vs seq maxdiff=%.3e\n", md);
        CHECK(md == 0.0, "batch token-1 == seq");
        // lanjutkan KEDUA engine 2 token sekuensial dari state masing-masing:
        // E (in-place setelah batch-3) vs E2 (setelah seq-2): beri token sama
        // pada posisi npos+3/npos+2? TIDAK sebanding (posisi beda) -> bandingkan
        // hanya sampai pola commit di Part 3. Di sini cukup: E2 seq token-2:
        vx27_engine_embed(E2, ext[2], x, err, sizeof err); vx27_engine_step(E2, x, h, npos+2, err, sizeof err);
        md = 0;
        for (int i = 0; i < 5120; i++) { double d = fabs((double)hb[2][i]-h[i]); if (d > md) md = d; }
        printf("  ckpt-token2 vs seq maxdiff=%.3e\n", md);
        CHECK(md == 0.0, "batch token-2 == seq");
        free(hs2); free(Sckpt); free(Cckpt); free(xs); free(hb);
        vx27_engine_close(E2);
    }

    // ---------- Part 3: prefill batch (7 token sekaligus) ----------
    {
        vx27_engine_reset(E);
        float (*xemb)[5120] = malloc(sizeof(float[64][5120]));
        for (size_t i = 0; i < n_ids; i++) vx27_engine_embed(E, ids[i], xemb[i], err, sizeof err);
        double t0 = now_ms();
        float (*hb)[5120] = malloc(sizeof(float[64][5120]));
        CHECK(vx27_engine_batch_step(E, xemb[0], hb[0], (int)n_ids, 0, NULL, NULL, err, sizeof err) == 0, "prefill batch");
        double t1 = now_ms();
        double md = 0;
        for (size_t p = 0; p < n_ids; p++)
            for (int i = 0; i < 5120; i += 3) {
                double d = fabs((double)hb[p][i] - hsave[(size_t)p*5120+i]);
                if (d > md) md = d;
            }
        printf("  prefill batch %.0f ms, vs-seq maxdiff=%.3e\n", t1 - t0, md);
        CHECK(md == 0.0, "prefill batch bit-identik");
        free(xemb); free(hb);
    }

    // ---------- Part 4: loop spekulatif N=4 K=2: seq vs batch+commit ----------
    {
        enum { N = 4, K = 2, B = 3 };
        uint32_t seq[4] = {0}, bat[4] = {0};
        // --- sekuensial (salin logika test_mtp, order [e,h] gate) ---
        double t0 = now_ms();
        {
            vx27_engine_reset(E);
            float *x = malloc(5120*4), *h = malloc(5120*4), *lg = malloc(248320*4);
            float *emb = malloc(5120*4), *dg = malloc(248320*4);
            int np = 0;
            for (size_t i = 0; i < n_ids; i++) {
                vx27_engine_embed(E, ids[i], x, err, sizeof err);
                vx27_engine_step(E, x, h, np, err, sizeof err);
                np++;
            }
            int out = 0;
            uint32_t pending = 0; int have_pending = 0;
            while (out < N) {
                uint32_t a;
                if (have_pending) { a = pending; have_pending = 0; }
                else a = vx27_engine_logits(E, h, lg, err, sizeof err);
                uint32_t d[2] = {0, 0};
                for (int k = 0; k < K; k++) {
                    uint32_t ein = (k == 0) ? a : d[k-1];
                    vx27_engine_embed(E, ein, emb, err, sizeof err);
                    vx27_mtp_draft(tab, tp, h, emb, dg, 0, 1, 0, err, sizeof err);
                    d[k] = argmax248k(dg);
                }
                seq[out++] = a;
                vx27_engine_embed(E, a, x, err, sizeof err);
                vx27_engine_step(E, x, h, np, err, sizeof err);
                np++;
                for (int k = 0; k < K && out < N; k++) {
                    uint32_t v = vx27_engine_logits(E, h, lg, err, sizeof err);
                    if (v == d[k]) {
                        seq[out++] = v;
                        vx27_engine_embed(E, v, x, err, sizeof err);
                        vx27_engine_step(E, x, h, np, err, sizeof err);
                        np++;
                    } else { pending = v; have_pending = 1; break; }
                }
            }
            free(x); free(h); free(lg); free(emb); free(dg);
        }
        double t1 = now_ms();
        int acc_b = 0, rounds = 0;
        double t2 = now_ms();
        {
            vx27_engine_reset(E);
            float *x = malloc(5120*4), *h = malloc(5120*4), *lg = malloc(248320*4);
            float *emb = malloc(5120*4), *dg = malloc(248320*4);
            float (*xs)[5120] = malloc(sizeof(float[3][5120]));
            float (*hb)[5120] = malloc(sizeof(float[3][5120]));
            float (*lgB)[248320] = malloc(sizeof(float[3][248320]));
            uint32_t lb[3] = {0,0,0};
            size_t SONE = (size_t)48*VX27_SSTATE_PER_LAYER, CONE = (size_t)48*VX27_CSTATE_PER_LAYER;
            float *Sckpt = malloc(SONE*B*4), *Cckpt = malloc(CONE*B*4);
            int np = 0;
            for (size_t i = 0; i < n_ids; i++) {
                vx27_engine_embed(E, ids[i], x, err, sizeof err);
                vx27_engine_step(E, x, h, np, err, sizeof err);
                np++;
            }
            int out = 0;
            uint32_t pending = 0; int have_pending = 0;
            // h = hidden terakhir prefill; sinkron dgn loop seq
            while (out < N) {
                uint32_t a;
                if (have_pending) { a = pending; have_pending = 0; }
                else a = vx27_engine_logits(E, h, lg, err, sizeof err);
                uint32_t d[2] = {0, 0};
                for (int k = 0; k < K; k++) {
                    uint32_t ein = (k == 0) ? a : d[k-1];
                    vx27_engine_embed(E, ein, emb, err, sizeof err);
                    vx27_mtp_draft(tab, tp, h, emb, dg, 0, 1, 0, err, sizeof err);
                    d[k] = argmax248k(dg);
                }
                // teacher-forced batch [a, d0, d1]
                vx27_engine_embed(E, a, xs[0], err, sizeof err);
                vx27_engine_embed(E, d[0], xs[1], err, sizeof err);
                vx27_engine_embed(E, d[1], xs[2], err, sizeof err);
                vx27_engine_batch_step(E, xs[0], hb[0], B, np, Sckpt, Cckpt, err, sizeof err);
                vx27_engine_logits_batch(E, hb[0], lgB[0], lb, B, err, sizeof err);
                rounds++;
                int m = 0;
                if (lb[0] == d[0]) { m = 1; if (lb[1] == d[1]) m = 2; }
                acc_b += m;
                vx27_engine_commit_ckpt(E, Sckpt, Cckpt, m);
                // konsumsi: a + m draft (+ bonus bila m==K)
                bat[out++] = a;
                if (m >= 1 && out < N) bat[out++] = d[0];
                if (m >= 2 && out < N) bat[out++] = d[1];
                if (m == 2 && out < N) { bat[out++] = lb[2]; } // bonus gratis
                np += 1 + m;
                if (m < K) { pending = lb[m]; have_pending = 1; memcpy(h, hb[m], 5120*4); }
                else memcpy(h, hb[B-1], 5120*4);
            }
            free(Sckpt); free(Cckpt);
            free(x); free(h); free(lg); free(emb); free(dg);
            free(xs); free(hb); free(lgB);
        }
        double t3 = now_ms();
        int same = 1;
        for (int t = 0; t < N; t++) if (seq[t] != bat[t]) same = 0;
        printf("  seq : [%u %u %u %u] %.0f ms\n", seq[0], seq[1], seq[2], seq[3], t1 - t0);
        printf("  bat : [%u %u %u %u] %.0f ms (rounds=%d)\n", bat[0], bat[1], bat[2], bat[3], t3 - t2, rounds);
        CHECK(same, "batch-verify == seq (exact)");
        double ts = (t1-t0)/N, tb = (t3-t2)/N;
        printf("  efektif per-token: seq %.0f ms vs bat %.0f ms (%.2fx), accept=%d\n", ts, tb, ts/(tb+1e-9), acc_b);
        CHECK(rounds > 0 && acc_b >= 0, "statistik verify valid");
    }

    free(hsave);
    free(ids);
    vx27_engine_close(E);
    tok_free(tok);
    if (fails) { printf("test_mtp_batch: %d FAIL\n", fails); return 1; }
    printf("test_mtp_batch: PASS\n");
    return 0;
}
