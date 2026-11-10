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
#include "../src/sampler.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

static uint32_t argmax248k(const float *lg) {
    int b = 0; for (int i = 1; i < 248320; i++) if (lg[i] > lg[b]) b = i;
    return (uint32_t)b;
}
static int finite248k(const float *lg) {
    for (int i = 0; i < 248320; i += 97) if (!(lg[i] > -1e30 && lg[i] < 1e30)) return 0;
    return 1;
}
// 1 bila id masuk top-k logits (seleksi parsial, tanpa sort penuh)
static int topk_hit(const float *lg, uint32_t id, int k) {
    float v = lg[id];
    int greater = 0;
    for (int i = 0; i < 248320 && greater < k; i++)
        if (lg[i] > v) greater++;
    return greater < k;
}
// peringkat penuh id dalam distribusi lg (0 = teratas). O(n).
static int full_rank(const float *lg, uint32_t id) {
    float v = lg[id];
    int r = 0;
    for (int i = 0; i < 248320; i++) if (lg[i] > v) r++;
    return r;
}
// cetak top-5 (id:logit) utk diagnostik
static void print_top5(const char *tag, const float *lg) {
    uint32_t bi[5] = {0,0,0,0,0};
    float bv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
    for (int i = 0; i < 248320; i++) {
        float v = lg[i];
        for (int k = 0; k < 5; k++) if (v > bv[k]) {
            for (int j = 4; j > k; j--) { bv[j] = bv[j-1]; bi[j] = bi[j-1]; }
            bv[k] = v; bi[k] = (uint32_t)i;
            break;
        }
    }
    printf("  top5 %s:", tag);
    for (int k = 0; k < 5; k++) printf(" %u(%.2f)", bi[k], bv[k]);
    printf("\n");
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

    // 1) probe 15 tensor MTP
    int nmtp = vx27_mtp_probe(vx27_engine_qtab(E), err, sizeof err);
    printf("  mtp tensors: %d\n", nmtp);
    CHECK(nmtp == 15, "15 tensor mtp.* bentuk sesuai");

    float *x = malloc(5120*4), *h = malloc(5120*4), *lg = malloc(248320*4), *dg = malloc(248320*4);
    float *emb = malloc(5120*4);
    float (*hsave)[5120] = malloc(sizeof(float[64][5120]));
    int npos = 0;
    // 2) prefill, simpan h per posisi + argmax utama
    for (size_t i = 0; i < n_ids; i++) {
        CHECK(vx27_engine_embed(E, ids[i], x, err, sizeof err) == 0, "embed");
        CHECK(vx27_engine_step(E, x, h, npos, err, sizeof err) == 0, "step");
        memcpy(hsave[npos], h, 5120*4);
        npos++;
    }
    uint32_t main_arg[64];
    for (int p = 0; p < npos; p++) {
        // h dari step perlu dihitung ulang? hsave menyimpan hidden per posisi; logits dari hsave
        uint32_t a = vx27_engine_logits(E, hsave[p], lg, err, sizeof err);
        main_arg[p] = a;
        CHECK(finite248k(lg), "logits utama finite");
    }
    printf("  main argmax:");
    for (int p = 0; p < npos; p++) printf(" %u", main_arg[p]);
    printf("\n");

    // 3) kalibrasi draft MTP vs utama: dua urutan concat + varian input
    //     + varian E tanpa gate + varian G off-by-one (draft(h_p, e_{p+1}) vs main(h_{p+1}))
    //     + top-10. Referensi vLLM: concat [e,h], MTP memprediksi SATU LANGKAH
    //     ke depan dari (hidden_t, emb_{t+1}).
    int agree_he = 0, agree_eh = 0, agree_C = 0, agree_E = 0, agree_G = 0, agree_F = 0;
    int t10_he = 0, t10_E = 0, t10_G = 0;
    uint32_t d_he[64], d_eh[64], d_C[64], d_E[64], d_G[64];
    float *lg2 = malloc(248320*4);
    for (int p = 0; p < npos; p++) {
        // emb token pada posisi p (token input posisi itu)
        CHECK(vx27_engine_embed(E, ids[p], emb, err, sizeof err) == 0, "embed utk draft");
        double t0 = now_ms();
        CHECK(vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[p], emb, dg, 1, 1, 0, err, sizeof err) == 0, "draft [h,e]");
        double t1 = now_ms();
        CHECK(finite248k(dg), "draft finite [h,e]");
        d_he[p] = argmax248k(dg);
        if (p == 0) printf("  draft [h,e] %.0f ms\n", t1 - t0);
        // determinisme: ulangi [h,e] SEBELUM dg dipakai varian lain
        {
            float *dg2 = malloc(248320*4);
            CHECK(vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[p], emb, dg2, 1, 1, 0, err, sizeof err) == 0, "draft ulang");
            double md = 0; for (int i = 0; i < 248320; i += 7) { double d = fabs((double)dg[i]-dg2[i]); if (d > md) md = d; }
            free(dg2);
            if (p == 0) { printf("  draft maxdiff rerun=%.3e\n", md); CHECK(md == 0.0, "draft deterministik bit-identik"); }
        }
        CHECK(vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[p], emb, dg, 0, 1, 0, err, sizeof err) == 0, "draft [e,h]");
        CHECK(finite248k(dg), "draft finite [e,h]");
        d_eh[p] = argmax248k(dg);
        if (d_he[p] == main_arg[p]) agree_he++;
        if (d_eh[p] == main_arg[p]) agree_eh++;
        // varian C: draft(h_{p-1}, e_p) vs main(h_p) — target t+1 dari state t-1 + emb t
        if (p >= 1) {
            CHECK(vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[p-1], emb, dg, 1, 1, 0, err, sizeof err) == 0, "draft varian C");
            d_C[p] = argmax248k(dg);
            if (d_C[p] == main_arg[p]) agree_C++;
        } else d_C[p] = 0;
        // varian E: tanpa gate (hidden-space konsisten dgn jalur utama); top-10 dua arah
        {
            CHECK(vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[p], emb, dg, 1, 0, 0, err, sizeof err) == 0, "draft varian E");
            CHECK(finite248k(dg), "draft finite E");
            d_E[p] = argmax248k(dg);
            if (d_E[p] == main_arg[p]) agree_E++;
            vx27_engine_logits(E, hsave[p], lg2, err, sizeof err);
            if (topk_hit(lg2, d_he[p], 10)) t10_he++;
            if (topk_hit(dg, main_arg[p], 10)) { /* draft top10 memuat main top1 */ }
            if (topk_hit(lg2, d_E[p], 10)) t10_E++;
        }
        // varian G (koreksi off-by-one): draft(h_p, e_{p+1}) vs main(h_{p+1}).
        // hsave[p] = hidden setelah token ids[p]; main_arg[p] = token posisi p+1.
        if (p + 1 < npos) {
            CHECK(vx27_engine_embed(E, main_arg[p], emb, err, sizeof err) == 0, "embed utk varian G");
            CHECK(vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[p], emb, dg, 0, 1, 0, err, sizeof err) == 0, "draft varian G");
            CHECK(finite248k(dg), "draft finite G");
            d_G[p] = argmax248k(dg);
            if (d_G[p] == main_arg[p+1]) agree_G++;
            if (topk_hit(dg, main_arg[p+1], 10)) t10_G++;
            // peringkat silang penuh: sinyal vs noise (acak ~= 124rb)
            if (p == 0) {
                int r1 = full_rank(dg, main_arg[p+1]);
                vx27_engine_logits(E, hsave[p+1], lg2, err, sizeof err);
                int r2 = full_rank(lg2, d_G[p]);
                printf("  rank silang: main-top1 di draft=%d, draft-top1 di main=%d (acak~=124000)\n", r1, r2);
            }
        } else d_G[p] = 0;
    }
    printf("  agree [h,e]=%d/%d [e,h]=%d/%d varC=%d/%d varE=%d/%d varG=%d/%d varF=%d/1 top10(main,draftHE)=%d/%d top10(main,draftE)=%d/%d top10G=%d/%d\n",
           agree_he, npos, agree_eh, npos, agree_C, npos > 1 ? npos - 1 : 0, agree_E, npos, agree_G, npos > 1 ? npos - 1 : 0,
           agree_F, t10_he, npos, t10_E, npos, t10_G, npos > 1 ? npos - 1 : 0);
    // H) matriks offset sistematis: pola sharp-but-disjoint menandakan salah
    //    pasangan (input, target). Uji input A=(h_p,e_p) dan B=(h_p,e(main_p))
    //    melawan target main_arg[p+s], s=-1..+2. Referensi [e,h] + gate.
    {
        int agreeA[4] = {0,0,0,0}, agreeB[4] = {0,0,0,0};
        int denA[4] = {0,0,0,0}, denB[4] = {0,0,0,0};
        const int shifts[4] = {-1, 0, 1, 2};
        for (int p = 0; p < npos; p++) {
            // input A
            vx27_engine_embed(E, ids[p], emb, err, sizeof err);
            vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[p], emb, dg, 0, 1, 0, err, sizeof err);
            uint32_t dA = argmax248k(dg);
            // input B
            vx27_engine_embed(E, main_arg[p], emb, err, sizeof err);
            vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[p], emb, dg, 0, 1, 0, err, sizeof err);
            uint32_t dB = argmax248k(dg);
            for (int si = 0; si < 4; si++) {
                int q = p + shifts[si];
                if (q < 0 || q >= npos) continue;
                denA[si]++; denB[si]++;
                if (dA == main_arg[q]) agreeA[si]++;
                if (dB == main_arg[q]) agreeB[si]++;
            }
        }
        printf("  offset s:      -1  +0  +1  +2\n");
        printf("  A=(h,e)       %d/%d %d/%d %d/%d %d/%d\n",
               agreeA[0], denA[0], agreeA[1], denA[1], agreeA[2], denA[2], agreeA[3], denA[3]);
        printf("  B=(h,e(main)) %d/%d %d/%d %d/%d %d/%d\n",
               agreeB[0], denB[0], agreeB[1], denB[1], agreeB[2], denB[2], agreeB[3], denB[3]);
    }
    // diagnostik top-5 posisi 0: apakah draft "sampah" atau "masuk akal tapi beda"
    {
        vx27_engine_embed(E, ids[0], emb, err, sizeof err);
        vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[0], emb, dg, 1, 1, 0, err, sizeof err);
        print_top5("draftHE", dg);
        vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[0], emb, dg, 1, 0, 0, err, sizeof err);
        print_top5("draftE-nogate", dg);
        vx27_engine_logits(E, hsave[0], lg2, err, sizeof err);
        print_top5("main", lg2);
        // varian F: hidden paska-norm-final (model.norm) sebagai input MTP —
        // hidden target di vLLM/HF adalah keluaran norm terakhir, bukan pra-norm.
        // (dequant BP3 lokal; norm 5120 = 40 grup)
        {
            const vx27_qt *mn = vx27_qfind(vx27_engine_qtab(E), "model.language_model.norm.weight");
            float *wn = malloc(5120*4), *hn = malloc(5120*4);
            if (mn) {
                if (mn->dtype == 4) {
                    const vx27_bp3_128 *G = (const vx27_bp3_128*)mn->blob;
                    for (int k = 0; k < 40; k++) {
                        uint16_t hh = G[k].scale_h; uint32_t u = hh;
                        uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
                        if (exp==0) bits=mant? (sign|((exp+1u)<<23)|(mant<<13)):sign;
                        else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
                        else bits=sign|((exp+112u)<<23)|(mant<<13);
                        float sc; memcpy(&sc,&bits,4);
                        for (int i = 0; i < 128; i++) {
                            uint32_t uu=((G[k].p[0*4+i/32]>>(i%32))&1)|(((G[k].p[1*4+i/32]>>(i%32))&1)<<1)|(((G[k].p[2*4+i/32]>>(i%32))&1)<<2);
                            wn[k*128+i] = (float)((uu>=4)?(int)uu-8:(int)uu) * sc;
                        }
                    }
                } else if (mn->dtype == 0) memcpy(wn, mn->blob, 5120*4);
                else memset(wn, 0, 5120*4);
                if (mn->dtype == 4 || mn->dtype == 0) {
                    double ss = 0; for (int i = 0; i < 5120; i++) ss += (double)hsave[0][i]*hsave[0][i];
                    float s = 1.0f / sqrtf((float)(ss/5120) + 1e-6f);
                    for (int i = 0; i < 5120; i++) hn[i] = hsave[0][i]*s*wn[i];
                    vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hn, emb, dg, 0, 1, 0, err, sizeof err);
                    print_top5("draftF-postnorm", dg);
                    uint32_t aF = argmax248k(dg);
                    printf("  varF top1=%u vs main %u -> %s\n", aF, main_arg[0], aF == main_arg[0] ? "COCOK" : "beda");
                    if (aF == main_arg[0]) agree_F++;
                } else printf("  varF dilewati (norm dtype=%u)\n", mn->dtype);
            }
            free(wn); free(hn);
        }
        // L) ablasi norm: fc saja (bit0) x {norm penuh, tanpa semua norm,
        //     tanpa final-norm, tanpa pre_fc-norm}. Kalau ada yang cocok dengan
        //     main -> konvensi norm yang salah; kalau semua 0 -> input fc salah.
        {
            vx27_engine_embed(E, ids[0], emb, err, sizeof err);
            const int abl[] = {1, 1|2, 1|4, 1|8};
            const char *anm[] = {"fc", "fc-nonorm", "fc-nofinal", "fc-nopre"};
            for (int ai = 0; ai < 4; ai++) {
                vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), hsave[0], emb, dg, 0, 1, abl[ai], err, sizeof err);
                uint32_t aL = argmax248k(dg);
                int rL = full_rank(dg, main_arg[0]);
                printf("  varL-%s top1=%u vs main %u -> %s (rank=%d)\n",
                       anm[ai], aL, main_arg[0], aL == main_arg[0] ? "COCOK" : "beda", rL);
            }
        }
    }
    int best_order = (agree_he > agree_eh) ? 1 : 0; // seri -> [e,h] sesuai referensi vLLM
    printf("  order terbaik: %s\n", best_order ? "[h,e]" : "[e,h]");

    // 4) loop spekulatif Eagle-style K=2 vs baseline greedy, N=4 token.
    //     Draft memakai input yang benar per referensi: d_k dari (h, emb(token
    //     yang baru diprediksi main)), memprediksi SATU posisi ke depan.
    //     Verify sekuensial (state maju alami, tanpa rollback) -> output exact.
    enum { N = 4, K = 2 };
    uint32_t seq_b[4] = {0}, seq_m[4] = {0};
    double t0 = now_ms();
    {
        vx27_engine_reset(E);
        int np = 0;
        for (size_t i = 0; i < n_ids; i++) {
            vx27_engine_embed(E, ids[i], x, err, sizeof err);
            vx27_engine_step(E, x, h, np, err, sizeof err);
            np++;
        }
        for (int t = 0; t < N; t++) {
            uint32_t a = vx27_engine_logits(E, h, lg, err, sizeof err);
            seq_b[t] = a;
            vx27_engine_embed(E, a, x, err, sizeof err);
            vx27_engine_step(E, x, h, np, err, sizeof err);
            np++;
        }
    }
    double t1 = now_ms();
    int acc = 0, drafts = 0, mainpass = 0;
    double t2 = now_ms();
    {
        vx27_engine_reset(E);
        int np = 0;
        for (size_t i = 0; i < n_ids; i++) {
            vx27_engine_embed(E, ids[i], x, err, sizeof err);
            vx27_engine_step(E, x, h, np, err, sizeof err);
            np++;
        }
        mainpass += (int)n_ids;
        int out = 0;
        uint32_t pending = 0; int have_pending = 0;
        while (out < N) {
            // token berikutnya: dari pending (sudah diverifikasi main) atau logits main
            uint32_t a;
            if (have_pending) { a = pending; have_pending = 0; }
            else a = vx27_engine_logits(E, h, lg, err, sizeof err);
            // draft K dari (h, emb(a)): d_0 memprediksi posisi np+1, dst.
            uint32_t d[2] = {0, 0};
            for (int k = 0; k < K; k++) {
                uint32_t ein = (k == 0) ? a : d[k-1];
                vx27_engine_embed(E, ein, emb, err, sizeof err);
                vx27_mtp_draft(vx27_engine_qtab(E), vx27_engine_tp(E), h, emb, dg, best_order, 1, 0, err, sizeof err);
                d[k] = argmax248k(dg);
                drafts++;
            }
            printf("  window: a=%u drafts=[%u %u]\n", a, d[0], d[1]);
            // emit a, majukan state
            seq_m[out++] = a;
            vx27_engine_embed(E, a, x, err, sizeof err);
            vx27_engine_step(E, x, h, np, err, sizeof err);
            np++; mainpass++;
            // verify draft lawan posisi-posisi berikut
            for (int k = 0; k < K && out < N; k++) {
                uint32_t v = vx27_engine_logits(E, h, lg, err, sizeof err);
                if (v == d[k]) {
                    acc++;
                    seq_m[out++] = v;
                    vx27_engine_embed(E, v, x, err, sizeof err);
                    vx27_engine_step(E, x, h, np, err, sizeof err);
                    np++; mainpass++;
                } else {
                    // tolak: v milik main, tunda emit ke iterasi berikut (tanpa recompute)
                    pending = v; have_pending = 1;
                    break;
                }
            }
        }
    }
    double t3 = now_ms();
    printf("  baseline: [%u %u %u %u] %.0f ms\n", seq_b[0], seq_b[1], seq_b[2], seq_b[3], t1 - t0);
    printf("  mtp K=2 : [%u %u %u %u] %.0f ms (accept %d/%d, mainpass %d)\n",
           seq_m[0], seq_m[1], seq_m[2], seq_m[3], t3 - t2, acc, drafts, mainpass);
    int same = 1;
    for (int t = 0; t < N; t++) if (seq_b[t] != seq_m[t]) same = 0;
    CHECK(same, "mtp-assisted == baseline greedy (exact)");
    double tps_b = 1000.0 * N / (t1 - t0), tps_m = 1000.0 * N / (t3 - t2);
    printf("  efektif: baseline %.2f tok/s vs mtp %.2f tok/s (accept %.0f%%)\n",
           tps_b, tps_m, drafts ? 100.0 * acc / drafts : 0.0);

    free(x); free(h); free(lg); free(lg2); free(dg); free(emb); free(hsave);
    vx27_engine_close(E);
    free(ids);
    tok_free(tok);
    if (fails) { printf("test_mtp: %d FAIL\n", fails); return 1; }
    printf("test_mtp: PASS\n");
    return 0;
}
