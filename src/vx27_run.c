#include "vx27_run.h"
#include "vx27_qfwd.h"
#include "vx27_pmat.h"
#include "vx27_rope.h"
#include "vx27_ffn.h"
#include "vx27_vxfile.h"
#include "vx27_mmap.h"
#include "vx27_bp3.h"
#include "vx27_config.h"
#include "quant.h"
#include "tokenizer.h"
#include "sampler.h"
#include "threadpool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
static double run_now_ms(void) {
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double run_now_ms(void) { return (double)clock()*1000.0/CLOCKS_PER_SEC; }
#endif

struct vx27_engine {
    vx27_map mc, mh;
    vx27_entry *ec, *eh;
    int nc, nh;
    const char **names;
    vx27_qt *tabs;
    int ntab;
    vx27_qtab tab;
    threadpool *tp;
    float *Sall;   // 48 lin * [48*128*128] state DeltaNet per V-head
    float *Call;   // 48 lin * [10240*3] state konvolusi (3 input terakhir)
    float *kv;     // 16 slot * VX27_QGEN_T * 1024 * 2 (stride HARUS VX27_QGEN_T, lihat vx27_qlayer)
    int max_t;
    float *h_a, *h_b; // scratch hidden ping-pong
};

static void deq_bp3_row(const vx27_qt *t, int row, float *out) {
    int cols = t->cols, ng = cols / 128;
    const vx27_bp3_128 *G = (const vx27_bp3_128 *)t->blob + (size_t)row * (size_t)ng;
    for (int k = 0; k < ng; k++) {
        uint16_t h = G[k].scale_h;
        uint32_t u = h;
        uint32_t sign = (u & 0x8000u) << 16, exp = (u >> 10) & 0x1Fu, mant = u & 0x3FFu, bits;
        if (exp == 0) bits = mant ? (sign | ((exp + 1u) << 23) | (mant << 13)) : sign;
        else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
        else bits = sign | ((exp + 112u) << 23) | (mant << 13);
        float sc; memcpy(&sc, &bits, 4);
        for (int i = 0; i < 128; i++) {
            uint32_t uu = ((G[k].p[0*4 + i/32] >> (i % 32)) & 1) |
                          (((G[k].p[1*4 + i/32] >> (i%32)) & 1) << 1) |
                          (((G[k].p[2*4 + i/32] >> (i%32)) & 1) << 2);
            out[k*128+i] = (float)((uu >= 4) ? (int)uu - 8 : (int)uu) * sc;
        }
    }
}

static void deq_q8_row(const vx27_qt *t, int row, float *out) {
    int cols = t->cols, nb = cols / 256;
    const block_q8_k *Q = (const block_q8_k *)t->blob + (size_t)row * (size_t)nb;
    for (int b = 0; b < nb; b++)
        for (int i = 0; i < 256; i++)
            out[b*256+i] = (float)Q[b].qs[i] * Q[b].d;
}

int vx27_engine_open(vx27_engine **e, const char *core, const char *head, int max_t, char *err, size_t errlen) {
    *e = NULL;
    if (max_t < 1 || max_t > VX27_QGEN_T) { snprintf(err, errlen, "max_t range 1..%d (VX27_QGEN_T)", VX27_QGEN_T); return -1; }
    vx27_engine *E = calloc(1, sizeof *E);
    if (!E) { snprintf(err, errlen, "out of memory"); return -1; }
    E->max_t = max_t;
    E->ec = malloc(sizeof(vx27_entry) * 4096);
    E->eh = malloc(sizeof(vx27_entry) * 4096);
    if (!E->ec || !E->eh) { free(E->ec); free(E->eh); free(E); snprintf(err, errlen, "out of memory"); return -1; }
    if (vx27_list(core, E->ec, 4096, &E->nc, err, errlen) != 0) { free(E->ec); free(E->eh); free(E); return -1; }
    if (vx27_list(head, E->eh, 4096, &E->nh, err, errlen) != 0) { free(E->ec); free(E->eh); free(E); return -1; }
    if (vx27_mmap_open(core, &E->mc, err, errlen) != 0) { free(E->ec); free(E->eh); free(E); return -1; }
    if (vx27_mmap_open(head, &E->mh, err, errlen) != 0) { vx27_mmap_close(&E->mc); free(E->ec); free(E->eh); free(E); return -1; }
    E->names = malloc(sizeof(char*) * 2048); E->tabs = malloc(sizeof(vx27_qt) * 2048);
    if (!E->names || !E->tabs) goto oom;
    E->ntab = 0;
    for (int i = 0; i < E->nc; i++) {
        uint64_t bb = 0; const void *p = vx27_mmap_blob(&E->mc, i, &bb);
        if (!p) { snprintf(err, errlen, "blob core %d", i); goto fail; }
        E->names[E->ntab] = E->ec[i].name;
        E->tabs[E->ntab].blob = p; E->tabs[E->ntab].dtype = E->ec[i].dtype;
        E->tabs[E->ntab].rows = (int)E->ec[i].ne0; E->tabs[E->ntab].cols = (int)E->ec[i].ne1;
        E->ntab++;
    }
    for (int i = 0; i < E->nh; i++) {
        uint64_t bb = 0; const void *p = vx27_mmap_blob(&E->mh, i, &bb);
        if (!p) { snprintf(err, errlen, "blob head %d", i); goto fail; }
        E->names[E->ntab] = E->eh[i].name;
        E->tabs[E->ntab].blob = p; E->tabs[E->ntab].dtype = E->eh[i].dtype;
        E->tabs[E->ntab].rows = (int)E->eh[i].ne0; E->tabs[E->ntab].cols = (int)E->eh[i].ne1;
        E->ntab++;
    }
    E->tab.n = E->ntab; E->tab.name = E->names; E->tab.t = E->tabs;
    E->tp = tp_create(8, err, errlen);
    if (!E->tp) goto fail;
    E->Sall = calloc((size_t)48*VX27_SSTATE_PER_LAYER, 4);
    E->Call = calloc((size_t)48*VX27_CSTATE_PER_LAYER, 4);
    E->kv = calloc((size_t)16 * (size_t)VX27_QGEN_T * 1024 * 2, 4);
    E->h_a = malloc(5120*4); E->h_b = malloc(5120*4);
    if (!E->Sall || !E->Call || !E->kv || !E->h_a || !E->h_b) goto oom;
    *e = E;
    return 0;
oom:
    snprintf(err, errlen, "out of memory");
fail:
    if (E->tp) tp_free(E->tp);
    vx27_mmap_close(&E->mc); vx27_mmap_close(&E->mh);
    free(E->names); free(E->tabs); free(E->ec); free(E->eh);
    free(E->Sall); free(E->Call); free(E->kv); free(E->h_a); free(E->h_b); free(E);
    return -1;
}

void vx27_engine_reset(vx27_engine *e) {
    if (!e) return;
    memset(e->Sall, 0, (size_t)48*VX27_SSTATE_PER_LAYER*4);
    memset(e->Call, 0, (size_t)48*VX27_CSTATE_PER_LAYER*4);
    memset(e->kv, 0, (size_t)16 * (size_t)VX27_QGEN_T * 1024 * 2 * 4);
}

void vx27_engine_close(vx27_engine *e) {
    if (!e) return;
    tp_free(e->tp);
    vx27_mmap_close(&e->mc); vx27_mmap_close(&e->mh);
    free(e->names); free(e->tabs); free(e->ec); free(e->eh);
    free(e->Sall); free(e->Call); free(e->kv); free(e->h_a); free(e->h_b); free(e);
}

int vx27_engine_embed(vx27_engine *e, uint32_t id, float *out, char *err, size_t errlen) {
    const vx27_qt *emb = vx27_qfind(&e->tab, "model.language_model.embed_tokens.weight");
    if (!emb) { snprintf(err, errlen, "embed tidak ketemu"); return -1; }
    if ((uint64_t)id >= (uint64_t)emb->rows) { snprintf(err, errlen, "id %u di luar vocab", id); return -1; }
    if (emb->dtype == 4) deq_bp3_row(emb, (int)id, out);
    else if (emb->dtype == 2) deq_q8_row(emb, (int)id, out);
    else if (emb->dtype == 0) memcpy(out, (const float*)emb->blob + (size_t)id * 5120, 5120*4);
    else { snprintf(err, errlen, "dtype embed %u tak didukung", emb->dtype); return -1; }
    return 0;
}

int vx27_engine_step(vx27_engine *e, const float *x, float *out, int n_pos, char *err, size_t errlen) {
    (void)err; (void)errlen;
    if (!e || !x || !out) return -1;
    if (n_pos < 0 || n_pos >= VX27_QGEN_T || n_pos >= e->max_t) return -1;
    float cosv[32], sinv[32];
    vx27_rope_cos_sin(n_pos, cosv, sinv, 64, 10000000.0f, 1.0f);
    int full_idx[64]; int fs = 0, li = 0, ci = 0;
    for (int L = 0; L < 64; L++) { int f = ((L+1)%4==0); full_idx[L] = f ? fs++ : -1; }
    float *Slin[64], *Clin[64];
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        if (is_full) { Slin[L] = NULL; Clin[L] = NULL; }
        else { Slin[L] = e->Sall + (size_t)(li++)*VX27_SSTATE_PER_LAYER; Clin[L] = e->Call + (size_t)(ci++)*VX27_CSTATE_PER_LAYER; }
    }
    memcpy(e->h_a, x, 5120*4);
    float *hi = e->h_a, *ho = e->h_b;
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        vx27_qlayer(&e->tab, L, is_full, hi, ho, n_pos, full_idx, Slin[L], e->kv, cosv, sinv, e->tp, Clin[L]);
        float *t = hi; hi = ho; ho = t;
    }
    memcpy(out, hi, 5120*4);
    return 0;
}

uint32_t vx27_engine_logits(vx27_engine *e, const float *h, float *logits, char *err, size_t errlen) {
    (void)err; (void)errlen;
    return (uint32_t)vx27_qlogits(&e->tab, h, logits, e->tp);
}

// Batch teacher-forced: B token (posisi n_pos0+b) SATU lintasan 64-layer,
// bobot dibaca 1x (pmat_batch). State S/Call/KV milik engine dimajukan
// in-place; Sckpt/Cckpt [B][48 lapis][state] opsional utk commit parsial.
// KV: prefix (trunk) dibagi, tail draft ditambah berurutan (rantai).
int vx27_engine_batch_step(vx27_engine *e, const float *xs, float *hs, int B, int n_pos0,
    float *Sckpt, float *Cckpt, char *err, size_t errlen) {
    if (!e || !xs || !hs) { if (err) snprintf(err, errlen, "arg null"); return -1; }
    if (B < 1 || B > 8) { if (err) snprintf(err, errlen, "B range 1..8"); return -1; }
    if (n_pos0 < 0 || n_pos0 + B > VX27_QGEN_T || n_pos0 + B > e->max_t) {
        if (err) snprintf(err, errlen, "posisi batch di luar max_t");
        return -1;
    }
    int full_idx[64]; int fs = 0, li = 0;
    for (int L = 0; L < 64; L++) { int f = ((L+1)%4==0); full_idx[L] = f ? fs++ : -1; }
    float *Slin[64], *Clin[64];
    size_t Soff[64], Coff[64]; // offset ckpt per layer (li), (size_t)-1 bila full
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        if (is_full) { Slin[L] = NULL; Clin[L] = NULL; Soff[L] = (size_t)-1; Coff[L] = (size_t)-1; }
        else {
            Slin[L] = e->Sall + (size_t)li*VX27_SSTATE_PER_LAYER;
            Clin[L] = e->Call + (size_t)li*VX27_CSTATE_PER_LAYER;
            Soff[L] = (size_t)li*VX27_SSTATE_PER_LAYER;
            Coff[L] = (size_t)li*VX27_CSTATE_PER_LAYER;
            li++;
        }
    }
    float *hiB = malloc((size_t)B*5120*4), *hoB = malloc((size_t)B*5120*4);
    if (!hiB || !hoB) { free(hiB); free(hoB); if (err) snprintf(err, errlen, "oom"); return -1; }
    memcpy(hiB, xs, (size_t)B*5120*4);
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        // row-buffer per layer: qlayer_batch tulis ckpt row-layer,
        // lalu scatter ke ckpt global [b*48+li][state].
        float *rowS = NULL, *rowC = NULL;
        if (!is_full && (Sckpt || Cckpt)) {
            if (Sckpt) rowS = malloc((size_t)B*VX27_SSTATE_PER_LAYER*4);
            if (Cckpt) rowC = malloc((size_t)B*VX27_CSTATE_PER_LAYER*4);
        }
        vx27_qlayer_batch(&e->tab, L, is_full, hiB, hoB, B, n_pos0, full_idx,
            Slin[L], e->kv, e->tp, Clin[L], rowS, rowC);
        if (!is_full && (Sckpt || Cckpt)) {
            for (int b = 0; b < B; b++) {
                if (rowS) memcpy(Sckpt + ((size_t)b*48 + (Soff[L]/VX27_SSTATE_PER_LAYER))*VX27_SSTATE_PER_LAYER,
                    rowS + (size_t)b*VX27_SSTATE_PER_LAYER, VX27_SSTATE_PER_LAYER*4);
                if (rowC) memcpy(Cckpt + ((size_t)b*48 + (Coff[L]/VX27_CSTATE_PER_LAYER))*VX27_CSTATE_PER_LAYER,
                    rowC + (size_t)b*VX27_CSTATE_PER_LAYER, VX27_CSTATE_PER_LAYER*4);
            }
            free(rowS); free(rowC);
        }
        float *t = hiB; hiB = hoB; hoB = t;
    }
    memcpy(hs, hiB, (size_t)B*5120*4);
    free(hiB); free(hoB);
    return 0;
}

void vx27_engine_logits_batch(vx27_engine *e, const float *hs, float *lgs,
    uint32_t *args, int B, char *err, size_t errlen) {
    (void)err; (void)errlen;
    if (!e || !hs || !lgs || !args || B < 1) return;
    vx27_qlogits_batch(&e->tab, hs, lgs, args, B, e->tp);
}

// Commit parsial tanpa rollback: state setelah token batch-ke-m menjadi
// state engine (дрaft tertolak di luar m diabaikan; KV tail di luar panjang
// terabaikan — overwrite oleh langkah berikut).
void vx27_engine_commit_ckpt(vx27_engine *e, const float *Sckpt, const float *Cckpt, int m) {
    if (!e || m < 0) return;
    size_t SONE = (size_t)48*VX27_SSTATE_PER_LAYER, CONE = (size_t)48*VX27_CSTATE_PER_LAYER;
    if (Sckpt) memcpy(e->Sall, Sckpt + (size_t)m*SONE, SONE*4);
    if (Cckpt) memcpy(e->Call, Cckpt + (size_t)m*CONE, CONE*4);
}

vx27_qtab *vx27_engine_qtab(vx27_engine *e) { return e ? &e->tab : 0; }
threadpool *vx27_engine_tp(vx27_engine *e) { return e ? e->tp : 0; }

// ---------- run/chat ----------

static void print_usage_run(void) {
    printf("usage: vx27.exe run \"prompt\" [--max-tokens N] [--temp F] [--top-k K] [--top-p P]\n"
           "                        [--seed S] [--core P] [--head P] [--tok-prefix P] [--no-template]\n");
}

static const char *opt_val(int argc, char **argv, const char *key, const char *dflt) {
    for (int i = 0; i + 1 < argc; i++)
        if (!strcmp(argv[i], key)) return argv[i+1];
    return dflt;
}
static int opt_has(int argc, char **argv, const char *key) {
    for (int i = 0; i < argc; i++) if (!strcmp(argv[i], key)) return 1;
    return 0;
}
static double opt_f64(int argc, char **argv, const char *key, double dflt) {
    const char *v = opt_val(argc, argv, key, NULL);
    if (!v) return dflt;
    return atof(v);
}
static long opt_long(int argc, char **argv, const char *key, long dflt) {
    const char *v = opt_val(argc, argv, key, NULL);
    if (!v) return dflt;
    return atol(v);
}

typedef struct {
    const char *core, *head, *tok_prefix;
    int max_new;
    double temp, top_p;
    int top_k;
    uint64_t seed;
    int use_template;
} run_cfg;

static void run_cfg_defaults(run_cfg *c) {
    c->core = "build/vx27_core.vx27";
    c->head = "build/vx27_head.vx27";
    c->tok_prefix = "build/tok";
    c->max_new = 32;
    c->temp = 0.0;
    c->top_p = 0.9;
    c->top_k = 40;
    c->seed = 12345;
    c->use_template = 1;
}

static int run_generate(vx27_engine *E, tokenizer *tok, sampler *smp,
                        const uint32_t *prompt, int n_prompt, int max_new,
                        int *n_pos_io, int stop_im_end, int stop_eot) {
    char err[512] = {0};
    float *x = malloc(5120*4), *h = malloc(5120*4), *logits = malloc(248320*4);
    if (!x || !h || !logits) { free(x); free(h); free(logits); return -1; }
    int n_pos = *n_pos_io;
    // prefill batch chunked (teacher-forcing, ≤8 token/chunk): bit-identik
    // dgn loop sekuensial (urutan glue sama), ~3-5x lebih cepat karena bobot
    // 11GB dibaca 1x per chunk, bukan per token.
    {
        float *xb = malloc(8*5120*4), *hb = malloc(8*5120*4);
        if (!xb || !hb) { free(xb); free(hb); free(x); free(h); free(logits); return -1; }
        int done = 0;
        while (done < n_prompt) {
            int B = n_prompt - done; if (B > 8) B = 8;
            for (int b = 0; b < B; b++) {
                if (vx27_engine_embed(E, prompt[done+b], xb + (size_t)b*5120, err, sizeof err) != 0) {
                    fprintf(stderr, "embed gagal pos %d: %s\n", n_pos + b, err);
                    free(xb); free(hb); free(x); free(h); free(logits);
                    return -1;
                }
            }
            if (vx27_engine_batch_step(E, xb, hb, B, n_pos, NULL, NULL, err, sizeof err) != 0) {
                fprintf(stderr, "prefill batch gagal pos %d\n", n_pos);
                free(xb); free(hb); free(x); free(h); free(logits);
                return -1;
            }
            memcpy(h, hb + (size_t)(B-1)*5120, 5120*4);
            n_pos += B; done += B;
        }
        free(xb); free(hb);
    }
    // decode
    int generated = 0;
    for (int t = 0; t < max_new; t++) {
        uint32_t id = vx27_engine_logits(E, h, logits, err, sizeof err);
        // sampler memakai logits (diubah in-place utk penalty); salin bila perlu deterministik?
        // sampler_sample mengubah logits utk repeat penalty: pakai langsung.
        id = sampler_sample(smp, logits, 248320);
        sampler_accept(smp, id);
        if (stop_im_end && id == 248046) break;
        if (stop_eot && id == 248044) break;
        uint8_t *bytes = NULL; size_t nb = 0;
        if (tok_decode(tok, &id, 1, &bytes, &nb, err, sizeof err) == 0 && bytes && nb) {
            fwrite(bytes, 1, nb, stdout);
            fflush(stdout);
            free(bytes);
        }
        generated++;
        if (t + 1 >= max_new) break;
        if (vx27_engine_embed(E, id, x, err, sizeof err) != 0) break;
        if (vx27_engine_step(E, x, h, n_pos, err, sizeof err) != 0) break;
        n_pos++;
    }
    *n_pos_io = n_pos;
    free(x); free(h); free(logits);
    return generated;
}

int vx27_run_main(int argc, char **argv) {
    if (argc < 3 || opt_has(argc, argv, "-h") || opt_has(argc, argv, "--help")) { print_usage_run(); return argc < 3 ? 1 : 0; }
    run_cfg c; run_cfg_defaults(&c);
    c.core = opt_val(argc, argv, "--core", c.core);
    c.head = opt_val(argc, argv, "--head", c.head);
    c.tok_prefix = opt_val(argc, argv, "--tok-prefix", c.tok_prefix);
    c.max_new = (int)opt_long(argc, argv, "--max-tokens", c.max_new);
    c.max_new = (int)opt_long(argc, argv, "--max-new", c.max_new);
    c.temp = opt_f64(argc, argv, "--temp", c.temp);
    c.top_k = (int)opt_long(argc, argv, "--top-k", c.top_k);
    c.top_p = opt_f64(argc, argv, "--top-p", c.top_p);
    c.seed = (uint64_t)opt_long(argc, argv, "--seed", (long)c.seed);
    if (opt_has(argc, argv, "--no-template")) c.use_template = 0;
    if (c.max_new < 1) c.max_new = 1;
    if (c.max_new > 256) c.max_new = 256;

    const char *prompt = argv[2];
    char err[1024] = {0};

    // tokenizer
    char pv[512], pt[512], pm[512];
    snprintf(pv, sizeof pv, "%s_vocab.bin", c.tok_prefix);
    snprintf(pt, sizeof pt, "%s_types.bin", c.tok_prefix);
    snprintf(pm, sizeof pm, "%s_merges.bin", c.tok_prefix);
    tokenizer *tok = NULL;
    if (tok_load_bin(&tok, pv, pt, pm, err, sizeof err) != 0) {
        fprintf(stderr, "tokenizer gagal: %s\n", err);
        return 1;
    }

    // chat template Qwen3: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    char *wrapped = NULL;
    const char *src_text = prompt;
    if (c.use_template) {
        size_t need = strlen(prompt) + 64;
        wrapped = malloc(need);
        if (!wrapped) { fprintf(stderr, "oom\n"); tok_free(tok); return 1; }
        snprintf(wrapped, need, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
        src_text = wrapped;
    }
    uint32_t *ids = NULL; size_t n_ids = 0;
    if (tok_encode(tok, (const uint8_t*)src_text, strlen(src_text), &ids, &n_ids, err, sizeof err) != 0) {
        fprintf(stderr, "encode gagal: %s\n", err);
        free(wrapped); tok_free(tok);
        return 1;
    }
    free(wrapped);
    if ((int)n_ids + c.max_new > VX27_QGEN_T) {
        fprintf(stderr, "prompt %zu + max_new %d melebihi KV %d; pendekkan prompt/--max-tokens\n",
                n_ids, c.max_new, VX27_QGEN_T);
        free(ids); tok_free(tok);
        return 1;
    }

    vx27_engine *E = NULL;
    if (vx27_engine_open(&E, c.core, c.head, (int)n_ids + c.max_new, err, sizeof err) != 0) {
        fprintf(stderr, "engine gagal: %s\n", err);
        free(ids); tok_free(tok);
        return 1;
    }

    sampler smp; sample_params sp;
    sample_params_default(&sp);
    sp.temp = (float)c.temp; sp.top_k = c.top_k; sp.top_p = (float)c.top_p;
    if (sampler_init(&smp, &sp, c.seed, err, sizeof err) != 0) {
        fprintf(stderr, "sampler gagal: %s\n", err);
        vx27_engine_close(E); free(ids); tok_free(tok);
        return 1;
    }
    for (size_t i = 0; i < n_ids; i++) sampler_accept(&smp, ids[i]);

    double t0 = run_now_ms();
    int n_pos = 0;
    // prefill+decode diukur terpisah: prefill = prompt, decode = generate
    // run_generate menggabungkan; ukur total lalu hitung tok/s efektif
    int gen = run_generate(E, tok, &smp, ids, (int)n_ids, c.max_new, &n_pos, 1, 1);
    double t1 = run_now_ms();
    printf("\n");
    double secs = (t1 - t0) / 1000.0;
    double tps = secs > 0 ? (double)(gen > 0 ? gen : 1) / secs : 0.0;
    fprintf(stderr, "[vx27] prompt=%zu gen=%d total_pos=%d waktu=%.1f ms (%.2f tok/s, temp=%.2f top_k=%d top_p=%.2f seed=%llu)\n",
            n_ids, gen, n_pos, t1 - t0, tps, c.temp, c.top_k, c.top_p, (unsigned long long)c.seed);
    sampler_free(&smp);
    vx27_engine_close(E);
    free(ids);
    tok_free(tok);
    return 0;
}

int vx27_chat_main(int argc, char **argv) {
    run_cfg c; run_cfg_defaults(&c);
    c.core = opt_val(argc, argv, "--core", c.core);
    c.head = opt_val(argc, argv, "--head", c.head);
    c.tok_prefix = opt_val(argc, argv, "--tok-prefix", c.tok_prefix);
    c.max_new = (int)opt_long(argc, argv, "--max-tokens", 64);
    c.temp = opt_f64(argc, argv, "--temp", 0.3);
    c.top_k = (int)opt_long(argc, argv, "--top-k", c.top_k);
    c.top_p = opt_f64(argc, argv, "--top-p", c.top_p);
    c.seed = (uint64_t)opt_long(argc, argv, "--seed", 777);
    char err[1024] = {0};

    char pv[512], pt[512], pm[512];
    snprintf(pv, sizeof pv, "%s_vocab.bin", c.tok_prefix);
    snprintf(pt, sizeof pt, "%s_types.bin", c.tok_prefix);
    snprintf(pm, sizeof pm, "%s_merges.bin", c.tok_prefix);
    tokenizer *tok = NULL;
    if (tok_load_bin(&tok, pv, pt, pm, err, sizeof err) != 0) {
        fprintf(stderr, "tokenizer gagal: %s\n", err);
        return 1;
    }
    vx27_engine *E = NULL;
    if (vx27_engine_open(&E, c.core, c.head, VX27_QGEN_T, err, sizeof err) != 0) {
        fprintf(stderr, "engine gagal: %s\n", err);
        tok_free(tok);
        return 1;
    }
    sampler smp; sample_params sp;
    sample_params_default(&sp);
    sp.temp = (float)c.temp; sp.top_k = c.top_k; sp.top_p = (float)c.top_p;
    if (sampler_init(&smp, &sp, c.seed, err, sizeof err) != 0) {
        fprintf(stderr, "sampler gagal: %s\n", err);
        vx27_engine_close(E); tok_free(tok);
        return 1;
    }

    printf("vx27 chat (ketik /exit keluar, /reset reset konteks, /stats statistik)\n");
    int n_pos = 0;
    char *line = NULL; size_t cap = 0;
    // pakai fgets dinamis sederhana
    char buf[8192];
    while (printf("> "), fflush(stdout), fgets(buf, sizeof buf, stdin)) {
        size_t L = strlen(buf);
        while (L && (buf[L-1] == '\n' || buf[L-1] == '\r')) buf[--L] = 0;
        if (!strcmp(buf, "/exit") || !strcmp(buf, "/quit")) break;
        if (!strcmp(buf, "/reset")) { vx27_engine_reset(E); sampler_reset(&smp); n_pos = 0; printf("(konteks direset)\n"); continue; }
        if (!strcmp(buf, "/stats")) { printf("(pos=%d/%d)\n", n_pos, VX27_QGEN_T); continue; }
        if (!L) continue;
        size_t need = L + 64;
        char *wrapped = malloc(need);
        if (!wrapped) break;
        snprintf(wrapped, need, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", buf);
        uint32_t *ids = NULL; size_t n_ids = 0;
        if (tok_encode(tok, (const uint8_t*)wrapped, strlen(wrapped), &ids, &n_ids, err, sizeof err) != 0) {
            fprintf(stderr, "encode gagal: %s\n", err);
            free(wrapped);
            continue;
        }
        free(wrapped);
        if (n_pos + (int)n_ids + c.max_new > VX27_QGEN_T) {
            fprintf(stderr, "konteks penuh (%d+%zu); /reset dulu\n", n_pos, n_ids);
            free(ids);
            continue;
        }
        for (size_t i = 0; i < n_ids; i++) sampler_accept(&smp, ids[i]);
        double t0 = run_now_ms();
        int gen = run_generate(E, tok, &smp, ids, (int)n_ids, c.max_new, &n_pos, 1, 1);
        double t1 = run_now_ms();
        printf("\n");
        double secs = (t1 - t0) / 1000.0;
        fprintf(stderr, "[%.1f ms, %d token, %.2f tok/s, pos %d]\n", t1 - t0, gen, secs > 0 ? gen / secs : 0, n_pos);
        free(ids);
        (void)line; (void)cap;
    }
    sampler_free(&smp);
    vx27_engine_close(E);
    tok_free(tok);
    printf("bye.\n");
    return 0;
}
