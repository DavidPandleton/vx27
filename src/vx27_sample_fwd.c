#include "vx27_sample_fwd.h"
#include "vx27_vxfile.h"
#include "vx27_mmap.h"
#include "vx27_block.h"
#include "vx27_bp3.h"
#include "quant.h"
#include "quant_avx2.h"
#include "threadpool.h"
#include <math.h>
#include <stdio.h>
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
static double now_ms(void) { return (double)clock()*1000.0/CLOCKS_PER_SEC; }
#endif

static float *dequant_blob(const vx27_entry *e, const void *blob) {
    float *f = malloc((size_t)e->ne0 * e->ne1 * 4 + 64);
    if (e->dtype == 0) { memcpy(f, blob, (size_t)e->ne0*e->ne1*4); }
    else if (e->dtype == 2) {
        const block_q8_k *q = blob;
        size_t nb = (size_t)e->ne0 * e->ne1 / 256;
        for (size_t b = 0; b < nb; b++)
            for (int i = 0; i < 256; i++) f[b*256+i] = (float)q[b].qs[i] * q[b].d;
    } else if (e->dtype == 4) {
        const vx27_bp3_128 *g = blob;
        size_t ng = (size_t)e->ne0 * e->ne1 / 128;
        for (size_t k = 0; k < ng; k++) {
            float sc = 0; { uint16_t h = g[k].scale_h; uint32_t u = h; /*fp16->f32*/
                uint32_t sign = (u & 0x8000u) << 16; uint32_t exp = (u >> 10) & 0x1Fu; uint32_t mant = u & 0x3FFu; uint32_t bits;
                if (exp == 0) bits = mant ? (sign | ((exp+1+112u)<<23) | (mant<<13)) : sign;
                else if (exp == 31) bits = sign | 0x7F800000u | (mant<<13);
                else bits = sign | ((exp+112u)<<23) | (mant<<13);
                memcpy(&sc, &bits, 4); }
            for (int i = 0; i < 128; i++) {
                uint32_t uu = ((g[k].p[0*4 + i/32] >> (i%32)) & 1) | (((g[k].p[1*4 + i/32] >> (i%32)) & 1) << 1) | (((g[k].p[2*4 + i/32] >> (i%32)) & 1) << 2);
                int q = (uu >= 4) ? (int)uu - 8 : (int)uu;
                f[k*128+i] = q * sc;
            }
        }
    } else { free(f); return NULL; }
    return f;
}

float vx27_sample_fwd(const char *core_path, const char *head_path,
                      int use_tp, double *ms, char *err, size_t errlen) {
    vx27_map mc = {0}, mh = {0};
    if (vx27_mmap_open(core_path, &mc, err, errlen) != 0) return 0;
    if (vx27_mmap_open(head_path, &mh, err, errlen) != 0) { vx27_mmap_close(&mc); return 0; }
    vx27_entry ec[64], eh[64]; int nc = 0, nh = 0;
    if (vx27_list(core_path, ec, 64, &nc, err, errlen) != 0) return 0;
    if (vx27_list(head_path, eh, 64, &nh, err, errlen) != 0) return 0;
    const void *cb[64], *hb[64]; uint64_t cbb[64], hbb[64];
    for (int i = 0; i < nc; i++) cb[i] = vx27_mmap_blob(&mc, i, &cbb[i]);
    for (int i = 0; i < nh; i++) hb[i] = vx27_mmap_blob(&mh, i, &hbb[i]);

    // dequant semua ke float (referensi); nama sesuai make_sample.py
    vx27_entry *all[128]; const void **allb[128]; int nn = 0;
    for (int i = 0; i < nc; i++) { all[nn] = &ec[i]; allb[nn] = &cb[i]; nn++; }
    for (int i = 0; i < nh; i++) { all[nn] = &eh[i]; allb[nn] = &hb[i]; nn++; }
    // helper: cari float per nama (dequant on demand + cache sederhana)
    const char *need[] = {
        "model.embed_tokens.weight", "model.norm.weight",
        "model.layers.0.linear_attn.q_proj.weight", "model.layers.0.linear_attn.k_proj.weight",
        "model.layers.0.linear_attn.v_proj.weight", "model.layers.0.linear_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight", "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.v_proj.weight", "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.self_attn.q_norm.weight", "model.layers.1.self_attn.k_norm.weight",
        "model.layers.1.mlp.gate_proj.weight", "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight", "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "lm_head.weight", "mtp.head.weight", NULL };
    float *F[32] = {0};
    for (int k = 0; need[k]; k++) {
        vx27_entry *e = NULL; const void *b = NULL;
        for (int i = 0; i < nn; i++) if (!strcmp(all[i]->name, need[k])) { e = all[i]; b = *allb[i]; break; }
        if (!e) { if (err) snprintf(err, errlen, "missing %s", need[k]); return 0; }
        F[k] = dequant_blob(e, b);
        if (!F[k]) { if (err) snprintf(err, errlen, "dtype %s", need[k]); return 0; }
    }
    // block cfg sampel
    vx27_block_cfg bc = {64, 128, 4, 2, 8, 8, 2, 2, 8, 10000.0f, 1.0f};
    vx27_block_weights wl = {F[9], F[10], F[2], F[3], F[4], F[5], F[6], F[7], F[8], NULL};
    vx27_block_weights wf = {F[20], F[21], F[11], F[12], F[13], F[14], F[17], F[18], F[19], F[15]};
    threadpool *tp = NULL;
    if (use_tp) { char e2[256] = {0}; tp = tp_create(8, e2, sizeof(e2)); }
    // embed: x = row[y] of embed (pakai token 7)
    const float *emb = F[0];
    float h0[64]; for (int i = 0; i < 64; i++) h0[i] = emb[7*64+i];
    float *S = calloc((size_t)2*8*8, 4);
    float h1[64], h2[64];
    float *kc = calloc((size_t)4*2*8, 4), *vc = calloc((size_t)4*2*8, 4);
    double t0 = now_ms();
    vx27_linear_block_forward(&bc, tp, h0, S, h1, &wl);
    vx27_full_block_forward(&bc, tp, h1, kc, vc, 0, h2, &wf);
    // norm + lm_head (argmax sebagai checksum fungsional)
    float *normed = malloc(64*4); float *logits = malloc(256*4);
    { double ss = 0; for (int i = 0; i < 64; i++) ss += (double)h2[i]*h2[i];
      float s = 1.0f / sqrtf((float)(ss/64) + 1e-6f);
      for (int i = 0; i < 64; i++) normed[i] = h2[i]*s*F[1][i]; }
    for (int r = 0; r < 256; r++) { double s = 0; for (int j = 0; j < 64; j++) s += (double)F[22][r*64+j]*normed[j]; logits[r] = (float)s; }
    double t1 = now_ms();
    if (ms) *ms = t1 - t0;
    double chk = 0; for (int i = 0; i < 64; i++) chk += h2[i];
    int arg = 0; for (int i = 1; i < 256; i++) if (logits[i] > logits[arg]) arg = i;
    // stash argmax in fractional part? kembalikan checksum; argmax dicetak via err? gunakan stdout di test.
    (void)arg;
    if (tp) tp_free(tp);
    free(S); free(kc); free(vc); free(normed); free(logits);
    for (int k = 0; need[k]; k++) free(F[k]);
    vx27_mmap_close(&mc); vx27_mmap_close(&mh);
    return (float)chk;
}
