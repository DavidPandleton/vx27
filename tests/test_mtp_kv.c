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
static int full_rank(const float *lg, uint32_t id) {
    float v = lg[id]; int r = 0;
    for (int i = 0; i < 248320; i++) if (lg[i] > v) r++;
    return r;
}
static int finite248k(const float *lg) {
    for (int i = 0; i < 248320; i += 97) if (!(lg[i] > -1e30 && lg[i] < 1e30)) return 0;
    return 1;
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

    CHECK(vx27_mtp_probe(tab, err, sizeof err) == 15, "15 tensor mtp");
    const int T = 512;
    float *kvc = calloc((size_t)2*T*1024, 4);
    float *kvcG = calloc((size_t)2*T*1024, 4);
    float *x = malloc(5120*4), *h = malloc(5120*4);
    float *lg = malloc(248320*4), *dg = malloc(248320*4), *dg2 = malloc(248320*4);
    float *emb = malloc(5120*4), *feat = malloc(5120*4), *feat2 = malloc(5120*4);
    float (*hsave)[5120] = malloc(sizeof(float[64][5120]));
    CHECK(kvc && kvcG && x && h && lg && dg && dg2 && emb && feat && feat2 && hsave, "alloc");

    // prefill utama sekuensial + argmax utama
    int npos = 0;
    for (size_t i = 0; i < n_ids; i++) {
        vx27_engine_embed(E, ids[i], x, err, sizeof err);
        vx27_engine_step(E, x, h, npos, err, sizeof err);
        memcpy(hsave[npos], h, 5120*4);
        npos++;
    }
    uint32_t *main_arg = malloc(sizeof(uint32_t[64]));
    for (int p = 0; p < npos; p++) {
        main_arg[p] = vx27_engine_logits(E, hsave[p], lg, err, sizeof err);
        CHECK(finite248k(lg), "logits utama finite");
    }
    printf("  main argmax:");
    for (int p = 0; p < npos; p++) printf(" %u", main_arg[p]);
    printf("\n");

    // ---------- kalibrasi 1: prefill MTP-KV + agreement (h_p, e_p) vs main p ----------
    int agree_kv = 0, agree_old = 0;
    for (int p = 0; p < npos; p++) {
        vx27_engine_embed(E, ids[p], emb, err, sizeof err);
        CHECK(vx27_mtp_step_kv(tab, tp, hsave[p], emb, kvc, p, T, NULL, dg, 0, 1, err, sizeof err) == 0, "step_kv");
        CHECK(finite248k(dg), "draft_kv finite");
        uint32_t a = argmax248k(dg);
        if (a == main_arg[p]) agree_kv++;
        if (p == 0) {
            int r1 = full_rank(dg, main_arg[p]);
            printf("  rank silang KV: main-top1 di draft=%d (lama=38, pra-fidelity=213rb)\n", r1);
        }
        // pembanding: draft lama single-token pd posisi sama
        CHECK(vx27_mtp_draft(tab, tp, hsave[p], emb, dg2, 0, 1, 0, err, sizeof err) == 0, "draft lama");
        if (argmax248k(dg2) == main_arg[p]) agree_old++;
        // determinisme step_kv (timpa pos sama -> identik)
        CHECK(vx27_mtp_step_kv(tab, tp, hsave[p], emb, kvc, p, T, NULL, dg2, 0, 1, err, sizeof err) == 0, "step_kv ulang");
        double md = 0;
        for (int i = 0; i < 248320; i += 7) { double d = fabs((double)dg[i]-dg2[i]); if (d > md) md = d; }
        if (p == 0) { printf("  step_kv rerun maxdiff=%.3e\n", md); CHECK(md == 0.0, "step_kv deterministik"); }
    }
    printf("  agree: KV=%d/%d lama=%d/%d\n", agree_kv, npos, agree_old, npos);
    // ---------- kalibrasi 1b: sapuan order x gate, pilih terbaik utk loop ----------
    int best_order = 0, best_gate = 1, best_agree = agree_kv;
    {
        float *kvcT = calloc((size_t)2*T*1024, 4);
        int orders[3] = {1, 0, 1}, gates[3] = {1, 0, 0};
        for (int c = 0; c < 3; c++) {
            memset(kvcT, 0, (size_t)2*T*1024*4);
            int ag = 0;
            for (int p = 0; p < npos; p++) {
                vx27_engine_embed(E, ids[p], emb, err, sizeof err);
                if (vx27_mtp_step_kv(tab, tp, hsave[p], emb, kvcT, p, T, NULL, dg, orders[c], gates[c], err, sizeof err) != 0) break;
                if (argmax248k(dg) == main_arg[p]) ag++;
            }
            printf("  combo order=%d gate=%d: agree=%d/%d\n", orders[c], gates[c], ag, npos);
            if (ag > best_agree) { best_agree = ag; best_order = orders[c]; best_gate = gates[c]; }
        }
        free(kvcT);
        printf("  terbaik: order=%d gate=%d agree=%d/%d\n", best_order, best_gate, best_agree, npos);
    }

    // ---------- kalibrasi 2: varian G (h_p, e(main_p)) vs main p+1, dgn KV sendiri ----------
    int agree_G = 0;
    for (int p = 0; p + 1 < npos; p++) {
        // bangun ulang prefix 0..p-1 di kvcG (kasual murni)
        memset(kvcG, 0, (size_t)2*T*1024*4);
        for (int i = 0; i < p; i++) {
            vx27_engine_embed(E, ids[i], emb, err, sizeof err);
            vx27_mtp_step_kv(tab, tp, hsave[i], emb, kvcG, i, T, NULL, NULL, 0, 1, err, sizeof err);
        }
        vx27_engine_embed(E, main_arg[p], emb, err, sizeof err);
        vx27_mtp_step_kv(tab, tp, hsave[p], emb, kvcG, p, T, NULL, dg, 0, 1, err, sizeof err);
        if (argmax248k(dg) == main_arg[p+1]) agree_G++;
    }
    printf("  agree G-KV=%d/%d\n", agree_G, npos - 1);

    // ---------- kalibrasi 3: rollout berantai K=2 gaya EAGLE di ujung prompt ----------
    // d0 = step(h, e(a)); d1 = step(feat0, e(d0)) — bandingkan dgn rantai lama (h tetap).
    uint32_t a_last;
    {
        float *hl = malloc(5120*4);
        memcpy(hl, hsave[npos-1], 5120*4);
        // merujuk pola test_mtp_batch: h terakhir prefill; a = argmax
        a_last = vx27_engine_logits(E, hl, lg, err, sizeof err);
        vx27_engine_embed(E, a_last, emb, err, sizeof err);
        // rantai KV: cache kvc sudah berisi prefill 0..npos-1 -> lanjutkan di pos npos
        CHECK(vx27_mtp_step_kv(tab, tp, hl, emb, kvc, npos, T, feat, dg, 0, 1, err, sizeof err) == 0, "d0 kv");
        uint32_t d0 = argmax248k(dg);
        vx27_engine_embed(E, d0, emb, err, sizeof err);
        CHECK(vx27_mtp_step_kv(tab, tp, feat, emb, kvc, npos+1, T, feat2, dg, 0, 1, err, sizeof err) == 0, "d1 kv berantai");
        uint32_t d1 = argmax248k(dg);
        // rantai lama utk banding: d1' = draft(h, e(d0)) single-token
        CHECK(vx27_mtp_draft(tab, tp, hl, emb, dg2, 0, 1, 0, err, sizeof err) == 0, "d1 lama");
        uint32_t d1old = argmax248k(dg2);
        printf("  rollout K=2: a=%u d0=%u d1_kv=%u d1_lama=%u\n", a_last, d0, d1, d1old);
        free(hl);
    }

    // ---------- loop spekulatif N=4: baseline greedy vs KV-berantai + batch-verify ----------
    enum { N = 4, K = 2, B = 3 };
    uint32_t seq[4] = {0}, kvb[4] = {0};
    double greedy_ms = 0;
    double t0 = now_ms();
    { // baseline greedy murni (prefill + N step)
        vx27_engine_reset(E);
        int np = 0;
        for (size_t i = 0; i < n_ids; i++) {
            vx27_engine_embed(E, ids[i], x, err, sizeof err);
            vx27_engine_step(E, x, h, np, err, sizeof err);
            np++;
        }
        for (int t = 0; t < N; t++) {
            uint32_t a = vx27_engine_logits(E, h, lg, err, sizeof err);
            seq[t] = a;
            vx27_engine_embed(E, a, x, err, sizeof err);
            vx27_engine_step(E, x, h, np, err, sizeof err);
            np++;
        }
    }
    double t1 = now_ms();
    int acc_kv = 0, rounds = 0;
    double t2 = now_ms();
    { // KV-berantai + batch-verify + commit
        vx27_engine_reset(E);
        memset(kvc, 0, (size_t)2*T*1024*4);
        float *xs = malloc(sizeof(float[3][5120])), *hb = malloc(sizeof(float[3][5120]));
        float *lgB = malloc(sizeof(float[3][248320]));
        uint32_t lb[3] = {0,0,0};
        size_t SONE = (size_t)48*VX27_SSTATE_PER_LAYER, CONE = (size_t)48*VX27_CSTATE_PER_LAYER;
        float *Sckpt = malloc(SONE*B*4), *Cckpt = malloc(CONE*B*4);
        int np = 0, nmp = 0;
        for (size_t i = 0; i < n_ids; i++) {
            vx27_engine_embed(E, ids[i], x, err, sizeof err);
            vx27_engine_step(E, x, h, np, err, sizeof err);
            np++;
            vx27_engine_embed(E, ids[i], emb, err, sizeof err);
            // prefill MTP-KV: (h_p, e_p) kausal (tanpa logits)
            vx27_mtp_step_kv(tab, tp, hsave[i], emb, kvc, nmp, T, NULL, NULL, 0, 1, err, sizeof err);
            nmp++;
        }
        int out = 0;
        uint32_t pending = 0; int have_pending = 0;
        while (out < N) {
            uint32_t a;
            if (have_pending) { a = pending; have_pending = 0; }
            else a = vx27_engine_logits(E, h, lg, err, sizeof err);
            // draft berantai: d0 dari (h, e(a)); d1 dari (feat0, e(d0))
            uint32_t d[2] = {0, 0};
            vx27_engine_embed(E, a, emb, err, sizeof err);
            vx27_mtp_step_kv(tab, tp, h, emb, kvc, nmp, T, feat, dg, 0, 1, err, sizeof err);
            d[0] = argmax248k(dg);
            vx27_engine_embed(E, d[0], emb, err, sizeof err);
            vx27_mtp_step_kv(tab, tp, feat, emb, kvc, nmp+1, T, feat2, dg, 0, 1, err, sizeof err);
            d[1] = argmax248k(dg);
            // batch-verify utama [a, d0, d1]
            vx27_engine_embed(E, a, xs, err, sizeof err);
            vx27_engine_embed(E, d[0], xs + 5120, err, sizeof err);
            vx27_engine_embed(E, d[1], xs + 10240, err, sizeof err);
            vx27_engine_batch_step(E, xs, hb, B, np, Sckpt, Cckpt, err, sizeof err);
            vx27_engine_logits_batch(E, hb, lgB, lb, B, err, sizeof err);
            rounds++;
            int m = 0;
            if (lb[0] == d[0]) { m = 1; if (lb[1] == d[1]) m = 2; }
            acc_kv += m;
            vx27_engine_commit_ckpt(E, Sckpt, Cckpt, m);
            kvb[out++] = a;
            if (m >= 1 && out < N) kvb[out++] = d[0];
            if (m >= 2 && out < N) kvb[out++] = d[1];
            if (m == 2 && out < N) kvb[out++] = lb[2];
            np += 1 + m; nmp += 1 + m;
            if (m < K) { pending = lb[m]; have_pending = 1; memcpy(h, hb + (size_t)m*5120, 5120*4); }
            else memcpy(h, hb + (size_t)(B-1)*5120, 5120*4);
        }
        free(xs); free(hb); free(lgB); free(Sckpt); free(Cckpt);
    }
    double t3 = now_ms();
    int same = 1;
    for (int t = 0; t < N; t++) if (seq[t] != kvb[t]) same = 0;
    printf("  greedy: [%u %u %u %u] %.0f ms\n", seq[0], seq[1], seq[2], seq[3], t1 - t0);
    printf("  kv   : [%u %u %u %u] %.0f ms (rounds=%d accept=%d)\n", kvb[0], kvb[1], kvb[2], kvb[3], t3 - t2, rounds, acc_kv);
    CHECK(same, "kv-batch == greedy (exact)");
    greedy_ms = t1 - t0;
    double tg = greedy_ms/N, tk = (t3-t2)/N;
    printf("  efektif per-token: greedy %.0f ms vs kv-batch %.0f ms (%.2fx)\n", tg, tk, tg/(tk+1e-9));
    printf("  accept rate KV-berantai: %d/%d\n", acc_kv, rounds*K);

    // ---------- Part 5: loop K=1 (1 draft, batch B=2) dgn combo terbaik ----------
    // Ambang menang K=1: p>0.2 (vs K=2: p>0.6). E[token]=1+2p per ~1.35F.
    uint32_t kv1[4] = {0,0,0,0};
    int acc1 = 0, rounds1 = 0;
    double t4 = now_ms();
    {
        vx27_engine_reset(E);
        memset(kvc, 0, (size_t)2*T*1024*4);
        float *xs = malloc(sizeof(float[2][5120])), *hb = malloc(sizeof(float[2][5120]));
        float *lgB = malloc(sizeof(float[2][248320]));
        uint32_t lb[2] = {0,0};
        size_t SONE = (size_t)48*VX27_SSTATE_PER_LAYER, CONE = (size_t)48*VX27_CSTATE_PER_LAYER;
        float *Sckpt = malloc(SONE*2*4), *Cckpt = malloc(CONE*2*4);
        int np = 0, nmp = 0;
        for (size_t i = 0; i < n_ids; i++) {
            vx27_engine_embed(E, ids[i], x, err, sizeof err);
            vx27_engine_step(E, x, h, np, err, sizeof err);
            np++;
            vx27_engine_embed(E, ids[i], emb, err, sizeof err);
            vx27_mtp_step_kv(tab, tp, hsave[i], emb, kvc, nmp, T, NULL, NULL, best_order, best_gate, err, sizeof err);
            nmp++;
        }
        int out = 0;
        uint32_t pending = 0; int have_pending = 0;
        while (out < N) {
            uint32_t a;
            if (have_pending) { a = pending; have_pending = 0; }
            else a = vx27_engine_logits(E, h, lg, err, sizeof err);
            vx27_engine_embed(E, a, emb, err, sizeof err);
            vx27_mtp_step_kv(tab, tp, h, emb, kvc, nmp, T, feat, dg, best_order, best_gate, err, sizeof err);
            uint32_t d0 = argmax248k(dg);
            vx27_engine_embed(E, a, xs, err, sizeof err);
            vx27_engine_embed(E, d0, xs + 5120, err, sizeof err);
            vx27_engine_batch_step(E, xs, hb, 2, np, Sckpt, Cckpt, err, sizeof err);
            vx27_engine_logits_batch(E, hb, lgB, lb, 2, err, sizeof err);
            rounds1++;
            int m = (lb[0] == d0) ? 1 : 0;
            acc1 += m;
            vx27_engine_commit_ckpt(E, Sckpt, Cckpt, m);
            kv1[out++] = a;
            if (m && out < N) kv1[out++] = d0;
            if (m && out < N) { kv1[out++] = lb[1]; pending = lb[1]; have_pending = 1; }
            else { pending = lb[m]; have_pending = 1; }
            np += 1 + m; nmp += 1 + m;
            memcpy(h, hb + (size_t)m*5120, 5120*4);
        }
        free(xs); free(hb); free(lgB); free(Sckpt); free(Cckpt);
    }
    double t5 = now_ms();
    int same1 = 1;
    for (int t = 0; t < N; t++) if (seq[t] != kv1[t]) same1 = 0;
    printf("  k1   : [%u %u %u %u] %.0f ms (rounds=%d accept=%d)\n", kv1[0], kv1[1], kv1[2], kv1[3], t5 - t4, rounds1, acc1);
    CHECK(same1, "k1-batch == greedy (exact)");
    double tk1 = (t5-t4)/N, tg1 = greedy_ms/N;
    printf("  efektif per-token: greedy %.0f ms vs k1-batch %.0f ms (%.2fx)\n", tg1, tk1, tg1/(tk1+1e-9));
    printf("  accept rate K=1: %d/%d\n", acc1, rounds1);

    free(kvc); free(kvcG); free(x); free(h); free(lg); free(dg); free(dg2);
    free(emb); free(feat); free(feat2); free(hsave); free(main_arg); free(ids);
    vx27_engine_close(E);
    tok_free(tok);
    if (fails) { printf("test_mtp_kv: %d FAIL\n", fails); return 1; }
    printf("test_mtp_kv: PASS\n");
    return 0;
}
