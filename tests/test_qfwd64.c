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
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
#include "../src/vx27_qfwd.h"
#include "../src/vx27_rope.h"
#include "../src/vx27_bp3.h"
#include "../src/quant.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

// dequant 1 baris BP3/F32 Langsung dari blob mmap (tanpa ekspansi penuh)
static void deq_row(const vx27_qt *t, int row, float *out) {
    int cols = t->cols;
    if (t->dtype == 0) { memcpy(out, (const float*)t->blob + (size_t)row*cols, (size_t)cols*4); return; }
    if (t->dtype == 2) {
        const block_q8_k *q = (const block_q8_k*)t->blob + (size_t)row * (cols/256);
        for (int b = 0; b < cols/256; b++) for (int i = 0; i < 256; i++) out[b*256+i] = (float)q[b].qs[i]*q[b].d;
        return;
    }
    const vx27_bp3_128 *g = (const vx27_bp3_128*)t->blob + (size_t)row * (cols/128);
    for (int k = 0; k < cols/128; k++) {
        uint16_t h = g[k].scale_h; uint32_t u = h;
        uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
        if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
        else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
        else bits=sign|((exp+112u)<<23)|(mant<<13);
        float sc; memcpy(&sc,&bits,4);
        for (int i = 0; i < 128; i++) {
            uint32_t uu=((g[k].p[0*4+i/32]>>(i%32))&1)|(((g[k].p[1*4+i/32]>>(i%32))&1)<<1)|(((g[k].p[2*4+i/32]>>(i%32))&1)<<2);
            out[k*128+i] = (float)((uu>=4)?(int)uu-8:(int)uu) * sc;
        }
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    char err[512] = {0};
    vx27_entry ec[4096], eh[4096]; int nc = 0, nh = 0;
    CHECK(vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof(err)) == 0, "list core penuh");
    CHECK(vx27_list("build/vx27_head.vx27", eh, 4096, &nh, err, sizeof(err)) == 0, "list head penuh");
    printf("  core=%d head=%d total=%d\n", nc, nh, nc + nh);
    vx27_map mc = {0}, mh = {0};
    CHECK(vx27_mmap_open("build/vx27_core.vx27", &mc, err, sizeof(err)) == 0, "mmap core");
    CHECK(vx27_mmap_open("build/vx27_head.vx27", &mh, err, sizeof(err)) == 0, "mmap head");

    // tabel gabungan
    static const char *names[2048]; static vx27_qt tabs[2048]; int N = 0;
    for (int i = 0; i < nc; i++) {
        uint64_t bb = 0; const void *p = vx27_mmap_blob(&mc, i, &bb);
        if (!p) { printf("FAIL: blob core %d\n", i); fails++; continue; }
        names[N] = ec[i].name;
        tabs[N].blob = p; tabs[N].dtype = ec[i].dtype;
        tabs[N].rows = (int)ec[i].ne0; tabs[N].cols = (int)ec[i].ne1; N++;
    }
    for (int i = 0; i < nh; i++) {
        uint64_t bb = 0; const void *p = vx27_mmap_blob(&mh, i, &bb);
        if (!p) { printf("FAIL: blob head %d\n", i); fails++; continue; }
        names[N] = eh[i].name;
        tabs[N].blob = p; tabs[N].dtype = eh[i].dtype;
        tabs[N].rows = (int)eh[i].ne0; tabs[N].cols = (int)eh[i].ne1; N++;
    }
    vx27_qtab tab = {N, names, tabs};
    printf("  tabel gabungan: %d tensor\n", N);
    // pra-validasi nama agar gagal dengan pesan, bukan segfault
    const char *need[] = {"model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight", "lm_head.weight", NULL};
    for (int i = 0; need[i]; i++)
        if (!vx27_qfind(&tab, need[i])) { printf("FAIL: hilang %s\n", need[i]); return 1; }
    const int probeL[4] = {0, 3, 52, 63};
    for (int pi = 0; pi < 4; pi++) {
        int L = probeL[pi], is_full = ((L+1)%4==0);
        char nm[160]; const char *suf[][2] = {
            {"linear_attn.in_proj_qkv.weight", "self_attn.q_proj.weight"},
            {"linear_attn.in_proj_z.weight", "self_attn.k_proj.weight"},
            {"mlp.gate_proj.weight", "mlp.gate_proj.weight"},
            {"mlp.down_proj.weight", "mlp.down_proj.weight"},
            {"input_layernorm.weight", "input_layernorm.weight"},
            {"post_attention_layernorm.weight", "post_attention_layernorm.weight"}, {NULL, NULL}};
        for (int k = 0; suf[k][0]; k++) {
            snprintf(nm, sizeof nm, "model.language_model.layers.%d.%s", L,
                     is_full ? suf[k][1] : suf[k][0]);
            // self_attn: ganti prefix linear->self untuk qkv? sudah di tabel
            if (!vx27_qfind(&tab, nm)) { printf("FAIL: hilang %s\n", nm); return 1; }
        }
    }
    printf("  ok: nama tensor lengkap (probe L=0,3,52,63)\n");
    // Kunci layout referensi Qwen3NextAttention: q_proj 12288 = 24 head x (256 query + 256 gate),
    // o_proj memakai 6144 kolom keluaran attention.
    for (int pi = 0; pi < 4; pi++) {
        int L = probeL[pi];
        if (((L+1)%4==0)) {
            char qnm[160], onm[160];
            snprintf(qnm, sizeof qnm, "model.language_model.layers.%d.self_attn.q_proj.weight", L);
            snprintf(onm, sizeof onm, "model.language_model.layers.%d.self_attn.o_proj.weight", L);
            const vx27_qt *tq = vx27_qfind(&tab, qnm);
            const vx27_qt *to = vx27_qfind(&tab, onm);
            CHECK(tq && tq->rows == 12288 && tq->cols == 5120, "q_proj 12288x5120 (query+gate)");
            CHECK(to && to->rows == 5120 && to->cols == 6144, "o_proj 5120x6144");
        }
    }

    // embed baris token 7
    const vx27_qt *emb = vx27_qfind(&tab, "model.language_model.embed_tokens.weight");
    CHECK(emb != NULL, "embed ketemu");
    float *x0 = malloc(5120*4);
    if (emb) deq_row(emb, 7, x0);

    int full_idx[64]; int fs = 0;
    for (int L = 0; L < 64; L++) { int f = ((L+1)%4==0); full_idx[L] = f ? fs++ : -1; }
    CHECK(fs == 16, "16 full layer");

    float cosv[32], sinv[32];
    vx27_rope_cos_sin(0, cosv, sinv, 64, 10000000.0f, 1.0f);

    float *Sall = calloc((size_t)48*VX27_SSTATE_PER_LAYER, 4); // 48 lin x state/V-head
    float *Call = calloc((size_t)48*VX27_CSTATE_PER_LAYER, 4); // 48 lin x conv-dim x 3
    float *kv = calloc((size_t)16*(size_t)VX27_QGEN_T*4*256*2, 4);
    float *h1 = malloc(5120*4), *h2 = malloc(5120*4);
    const size_t Sstride = VX27_SSTATE_PER_LAYER, Cstride = VX27_CSTATE_PER_LAYER;
    // S/conv per layer: butuh offset per linear layer
    float *Slin[64], *Clin[64]; int lli = 0, cli = 0;
    for (int L = 0; L < 64; L++) if ((L+1)%4!=0) {
        Slin[L] = Sall + (size_t)(lli++)*Sstride;
        Clin[L] = Call + (size_t)(cli++)*Cstride;
    } else { Slin[L] = NULL; Clin[L] = NULL; }

    threadpool *tp = tp_create(8, err, sizeof(err));
    CHECK(tp != NULL, "tp 8");

    for (int i = 0; i < 5120; i++) h1[i] = x0[i];
    double t0 = now_ms();
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        vx27_qlayer(&tab, L, is_full, h1, h2, 0, full_idx, Slin[L], kv, cosv, sinv, tp, Clin[L]);
        float *t = h1; h1 = h2; h2 = t;
        if ((L+1) % 16 == 0) {
            double c = 0; for (int i = 0; i < 5120; i += 64) c += h1[i];
            printf("  layer %d checksum(sebagian)=%.6f\n", L, c);
        }
    }
    double t1 = now_ms();
    double chk = 0; int fin = 1;
    for (int i = 0; i < 5120; i++) { chk += h1[i]; if (!(h1[i] > -1e30 && h1[i] < 1e30)) fin = 0; }
    printf("  fwd64 checksum=%.6f finite=%d ms=%.0f (%.2f ms/layer)\n", chk, fin, t1-t0, (t1-t0)/64);
    CHECK(fin, "output finite");
    CHECK(chk == chk && chk != 0.0, "checksum non-nol");

    float *logits = malloc(248320*4);
    double t2 = now_ms();
    int arg = vx27_qlogits(&tab, h1, logits, tp);
    double t3 = now_ms();
    printf("  argmax=%d ms=%.0f | top5:", arg, t3-t2);
    for (int k = 0; k < 5; k++) {
        int bi = 0; for (int i = 0; i < 248320; i++) if (logits[i] > logits[bi]) bi = i;
        printf(" %d(%.2f)", bi, logits[bi]); logits[bi] = -1e30f;
    }
    printf("\n");
    CHECK(arg >= 0 && arg < 248320, "argmax valid");
    free(logits);

    // determinisme: ulangi dengan state świeży, harus identik
    memset(Sall, 0, (size_t)48*Sstride*4); memset(Call, 0, (size_t)48*Cstride*4);
    memset(kv, 0, (size_t)16*(size_t)VX27_QGEN_T*4*256*2*4);
    lli = 0; cli = 0;
    for (int L = 0; L < 64; L++) if ((L+1)%4!=0) {
        Slin[L] = Sall + (size_t)(lli++)*Sstride;
        Clin[L] = Call + (size_t)(cli++)*Cstride;
    } else { Slin[L] = NULL; Clin[L] = NULL; }
    float *g1 = malloc(5120*4), *g2 = malloc(5120*4);
    for (int i = 0; i < 5120; i++) g1[i] = x0[i];
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        vx27_qlayer(&tab, L, is_full, g1, g2, 0, full_idx, Slin[L], kv, cosv, sinv, tp, Clin[L]);
        float *t = g1; g1 = g2; g2 = t;
    }
    double md = 0, chk2 = 0;
    for (int i = 0; i < 5120; i++) { double d = fabs((double)g1[i]-h1[i]); if (d > md) md = d; chk2 += g1[i]; }
    printf("  rerun checksum=%.6f maxdiff=%.3e\n", chk2, md);
    CHECK(md == 0.0, "deterministik penuh");

    free(x0); free(Sall); free(Call); free(kv); free(h1); free(h2); free(g1); free(g2);
    tp_free(tp);
    vx27_mmap_close(&mc); vx27_mmap_close(&mh);
    if (fails) { printf("test_qfwd64: %d FAIL\n", fails); return 1; }
    printf("test_qfwd64: PASS\n");
    return 0;
}
