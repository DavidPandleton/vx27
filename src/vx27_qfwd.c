#include "vx27_qfwd.h"
#include "vx27_pmat.h"
#include "vx27_deltanet.h"
#include "vx27_rope.h"
#include "vx27_ffn.h"
#include "vx27_bp3.h"
#include "quant.h"
#include "vx27_config.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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

const vx27_qt *vx27_qfind(const vx27_qtab *tab, const char *name) {
    for (int i = 0; i < tab->n; i++) if (!strcmp(tab->name[i], name)) return &tab->t[i];
    return NULL;
}
#define QF(tab, nm) vx27_qfind(tab, nm)

static void rmsn(const float *x, const float *w, float *y, int n) {
    // Qwen3_5RMSNorm zero-centered: y = x*s*(1+w). Wajib untuk SEMUA norm teks
    // (input_ln, post_ln, q/k_norm full-attn, model.norm, pre_fc, mtp.norm).
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float s = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    if (w) for (int i = 0; i < n; i++) y[i] = x[i]*s*(w[i]+1.0f);
    else for (int i = 0; i < n; i++) y[i] = x[i]*s;
}
// RMSNorm standar (xw, tanpa +1): HANYA untuk GatedRMSNorm DeltaNet yang
// bobotnya init ones (referensi Qwen3NextRMSNormGated).
static void rmsn_std(const float *x, const float *w, float *y, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float s = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    if (w) for (int i = 0; i < n; i++) y[i] = x[i]*s*w[i];
    else for (int i = 0; i < n; i++) y[i] = x[i]*s;
}
static float silu1(float v) { return v / (1.0f + expf(-v)); }

// dequant 1 tensor ke float (hanya utk norm 1D kecil di jalur-Q; matriks besar tetap packed)
static void deq_small(const vx27_qt *t, float *out, int n) {
    if (t->dtype == 0) { memcpy(out, t->blob, (size_t)n*4); return; }
    if (t->dtype == 2) {
        const block_q8_k *q = (const block_q8_k*)t->blob;
        for (int b = 0; b < n/256; b++) for (int i = 0; i < 256; i++) out[b*256+i] = (float)q[b].qs[i]*q[b].d;
        return;
    }
    const vx27_bp3_128 *g = (const vx27_bp3_128*)t->blob;
    extern void vx27_bp3_128_unpack(const vx27_bp3_128 *g, float *w); // didefinisikan di bawah
    for (int k = 0; k < n/128; k++) vx27_bp3_128_unpack(&g[k], out + k*128);
}
void vx27_bp3_128_unpack(const vx27_bp3_128 *g, float *w) {
    uint16_t h = g->scale_h; uint32_t u = h;
    uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
    if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
    else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
    else bits=sign|((exp+112u)<<23)|(mant<<13);
    float sc; memcpy(&sc,&bits,4);
    for (int i = 0; i < 128; i++) {
        uint32_t uu=((g->p[0*4+i/32]>>(i%32))&1)|(((g->p[1*4+i/32]>>(i%32))&1)<<1)|(((g->p[2*4+i/32]>>(i%32))&1)<<2);
        w[i] = (float)((uu>=4)?(int)uu-8:(int)uu) * sc;
    }
}

void vx27_qlayer(const vx27_qtab *tab, int L, int is_full,
                 const float *x, float *out, int n_pos,
                 const int *full_idx, float *S, float *kvc,
                 const float *rope_cos, const float *rope_sin,
                 threadpool *tp, float *conv) {
    const int H = 5120, F = 17408;
    char nm[160];
    float *xn = malloc(H*4), *x2 = malloc(H*4), *fo = malloc(H*4);
    float *g = malloc((size_t)F*4), *u = malloc((size_t)F*4), *h = malloc((size_t)F*4);
    float *fn = malloc(H*4), *ao = malloc(H*4);
    const vx27_qt *t;
    char tmp[160];

    snprintf(nm, sizeof nm, "model.language_model.layers.%d.input_layernorm.weight", L);
    t = QF(tab, nm); float wn[5120]; deq_small(t, wn, H);
    rmsn(x, wn, xn, H);

    if (!is_full) {
        // Gated DeltaNet, pola referensi torch_recurrent_gated_delta_rule +
        // causal_conv1d_update (lihat vx27_deltanet.h): conv carryover per token,
        // q/k l2norm + q/=sqrt(hd), beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias),
        // state per V-head [48 x 128 x 128], gated RMSNorm output (bobot init ones).
        const int nqk = 16, nv = 48, hd = 128, qd = nqk*hd, vd = nv*hd, qkvd = qd*2+vd;
        float *qkv = malloc((size_t)qkvd*4), *qkv2 = malloc((size_t)qkvd*4), *z = malloc((size_t)vd*4);
        float *avec = malloc(nv*4), *bvec = malloc(nv*4);
        float *ret = malloc((size_t)vd*4), *o = malloc((size_t)vd*4);
        float *q = malloc((size_t)qd*4), *k = malloc((size_t)qd*4), *v = malloc((size_t)vd*4);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, xn, qkv, qkvd, H);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.in_proj_z.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, xn, z, vd, H);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.in_proj_a.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, xn, avec, nv, H);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.in_proj_b.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, xn, bvec, nv, H);
        // conv1d depthwise k=4 + SiLU, state = 3 input terakhir (nol di awal).
        // Layout bobot [10240,1,4] row-major: tap j = cw[i*4+j], j=0 terlama.
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.conv1d.weight", L);
        const vx27_qt *tc = QF(tab, tmp);
        float *cw = malloc(40960*4); deq_small(tc, cw, 40960);
        vx27_conv1d_step(conv, qkv, cw, qkv2, qkvd, 1);
        free(cw);
        memcpy(q, qkv2, (size_t)qd*4); memcpy(k, qkv2+qd, (size_t)qd*4); memcpy(v, qkv2+qd*2, (size_t)vd*4);
        // QK-norm (1+w) + l2norm kernel + q/=sqrt(hd): qn/kn per head 16x128.
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.norm.weight", L);
        t = QF(tab, tmp); float qn[128]; deq_small(t, qn, 128);
        for (int hh = 0; hh < nqk; hh++) {
            float tq[128], tk[128];
            rmsn(q+hh*hd, qn, tq, hd); rmsn(k+hh*hd, qn, tk, hd);
            vx27_deltanet_l2norm(tq, q+hh*hd, hd);
            vx27_deltanet_l2norm(tk, k+hh*hd, hd);
        }
        {
            float qs = 1.0f / sqrtf((float)hd);
            for (int i = 0; i < qd; i++) q[i] *= qs;
        }
        // A_log / dt_bias + aturan delta berpintu per V-head (state milik V-head).
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.A_log", L);
        t = QF(tab, tmp); float alog[48]; deq_small(t, alog, 48);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.dt_bias", L);
        t = QF(tab, tmp); float dtb[48]; deq_small(t, dtb, 48);
        for (int vh = 0; vh < nv; vh++) {
            int qk = vh * nqk / nv;
            double beta, g;
            vx27_deltanet_gate_params((double)avec[vh], (double)alog[vh], (double)bvec[vh], (double)dtb[vh], &beta, &g);
            float *Svh = S + (size_t)vh*hd*hd;
            float *rv = ret + (size_t)vh*hd;
            vx27_deltanet_step_exact(Svh, q+(size_t)qk*hd, k+(size_t)qk*hd, v+(size_t)vh*hd, beta, g, rv, hd);
        }
        // Gated RMSNorm output: norm(ret)*w*silu(z), bobot = linear_attn.norm (init ones).
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.norm.weight", L);
        t = QF(tab, tmp); float nw[128]; deq_small(t, nw, 128);
        for (int vh = 0; vh < nv; vh++) {
            float tn[128];
            rmsn_std(ret+(size_t)vh*hd, nw, tn, hd);
            for (int i = 0; i < hd; i++) o[(size_t)vh*hd+i] = tn[i] * silu1(z[(size_t)vh*hd+i]);
        }
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.out_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, o, ao, H, vd);
        free(qkv); free(qkv2); free(z); free(avec); free(bvec); free(ret); free(o);
        free(q); free(k); free(v);
    } else {
        const int nq = 24, nkv = 4, hd = 256, rd = 64;
        float *qg = malloc((size_t)nq*hd*2*4), *q = malloc((size_t)nq*hd*4), *gate = malloc((size_t)nq*hd*4);
        float *k1 = malloc((size_t)nkv*hd*4), *v1 = malloc((size_t)nkv*hd*4);
        // q_proj keluaran 12288 = 24 head x (256 query + 256 gate), sesuai Qwen3NextAttention.
        // Baris q_proj: [h*512, h*512+256) query head-h; [h*512+256, h*512+512) gate head-h.
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.q_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, xn, qg, nq*hd*2, H);
        for (int hh = 0; hh < nq; hh++) {
            memcpy(q + (size_t)hh*hd, qg + (size_t)hh*hd*2, (size_t)hd*4);
            memcpy(gate + (size_t)hh*hd, qg + (size_t)hh*hd*2 + hd, (size_t)hd*4);
        }
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.k_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, xn, k1, nkv*hd, H);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.v_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, xn, v1, nkv*hd, H);
        // QK-norm
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.q_norm.weight", L);
        t = QF(tab, tmp); float qnw[256]; deq_small(t, qnw, 256);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.k_norm.weight", L);
        t = QF(tab, tmp); float knw[256]; deq_small(t, knw, 256);
        for (int hh = 0; hh < nq; hh++) { float tt[256]; rmsn(q+hh*hd, qnw, tt, hd); memcpy(q+hh*hd, tt, hd*4); }
        for (int hh = 0; hh < nkv; hh++) { float tt[256]; rmsn(k1+hh*hd, knw, tt, hd); memcpy(k1+hh*hd, tt, hd*4); }
        vx27_rope_apply(q, nq, hd, rd, rope_cos, rope_sin);
        vx27_rope_apply(k1, nkv, hd, rd, rope_cos, rope_sin);
        // KV cache append per posisi + attention softmax penuh (n_pos kecil di test)
        int slot = full_idx[L];
        float *kc = kvc + ((size_t)slot*VX27_QGEN_T + (size_t)n_pos)*nkv*hd;
        float *vc = kvc + ((size_t)16*VX27_QGEN_T*nkv*hd) + ((size_t)slot*VX27_QGEN_T + (size_t)n_pos)*nkv*hd;
        memcpy(kc, k1, (size_t)nkv*hd*4); memcpy(vc, v1, (size_t)nkv*hd*4);
        int nq1 = n_pos + 1;
        float *att = malloc((size_t)nq*hd*4);
        float *sc = malloc((size_t)(nq1 > 0 ? nq1 : 1)*4);
        float scl = 1.0f / sqrtf((float)hd);
        for (int hh = 0; hh < nq; hh++) {
            int kvh = hh * nkv / nq;
            const float *qh = q + (size_t)hh*hd;
            float smax = -1e30f, ssum = 0.0f;
            for (int pp = 0; pp < nq1; pp++) {
                const float *kh = kvc + ((size_t)slot*VX27_QGEN_T + (size_t)pp)*nkv*hd + (size_t)kvh*hd;
                double s = 0; for (int i = 0; i < hd; i++) s += (double)qh[i]*kh[i];
                sc[pp] = (float)(s * scl);
                if (sc[pp] > smax) smax = sc[pp];
            }
            for (int pp = 0; pp < nq1; pp++) { sc[pp] = expf(sc[pp]-smax); ssum += sc[pp]; }
            float *oh = att + (size_t)hh*hd;
            for (int i = 0; i < hd; i++) oh[i] = 0.0f;
            for (int pp = 0; pp < nq1; pp++) {
                float a = sc[pp]/ssum;
                const float *vh = kvc + ((size_t)16*VX27_QGEN_T*nkv*hd) + ((size_t)slot*VX27_QGEN_T + (size_t)pp)*nkv*hd + (size_t)kvh*hd;
                for (int i = 0; i < hd; i++) oh[i] += a*vh[i];
            }
        }
        free(sc);
        // Output gate referensi: attn_out *= sigmoid(gate) sebelum o_proj.
        for (int i = 0; i < nq*hd; i++) att[i] *= 1.0f / (1.0f + expf(-gate[i]));
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.o_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat(tp, t->blob, t->dtype, att, ao, H, nq*hd);
        free(qg); free(q); free(gate); free(k1); free(v1); free(att);
    }
    for (int i = 0; i < H; i++) x2[i] = x[i] + ao[i];
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.post_attention_layernorm.weight", L);
    t = QF(tab, nm); float pn[5120]; deq_small(t, pn, H);
    rmsn(x2, pn, fn, H);
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.mlp.gate_proj.weight", L);
    t = QF(tab, nm); vx27_pmat(tp, t->blob, t->dtype, fn, g, F, H);
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.mlp.up_proj.weight", L);
    t = QF(tab, nm); vx27_pmat(tp, t->blob, t->dtype, fn, u, F, H);
    vx27_swiglu_forward_avx2(g, u, h, F);
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.mlp.down_proj.weight", L);
    t = QF(tab, nm); vx27_pmat(tp, t->blob, t->dtype, h, fo, H, F);
    for (int i = 0; i < H; i++) out[i] = x2[i] + fo[i];
    free(xn); free(x2); free(fn); free(g); free(u); free(h); free(fo); free(ao);
}

// Teacher-forced batch satu layer (lihat qfwd.h). 1<=B<=8.
void vx27_qlayer_batch(const vx27_qtab *tab, int L, int is_full,
                 const float *xs, float *outs, int B, int n_pos0,
                 const int *full_idx, float *S, float *kvc,
                 threadpool *tp, float *conv, float *Sckpt, float *Cckpt) {
    const int H = 5120, F = 17408;
    if (B < 1) return;
    if (B > 8) B = 8;
    char nm[160], tmp[160];
    const vx27_qt *t;
    float *xnB = malloc((size_t)B*H*4), *x2B = malloc((size_t)B*H*4);
    float *fnB = malloc((size_t)B*H*4), *aoB = malloc((size_t)B*H*4);
    float *gB = malloc((size_t)B*F*4), *uB = malloc((size_t)B*F*4), *hB = malloc((size_t)B*F*4);
    float *foB = malloc((size_t)B*H*4);
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.input_layernorm.weight", L);
    t = QF(tab, nm); float wn[5120]; deq_small(t, wn, H);
    for (int b = 0; b < B; b++) rmsn(xs + (size_t)b*H, wn, xnB + (size_t)b*H, H);

    if (!is_full) {
        const int nqk = 16, nv = 48, hd = 128, qd = nqk*hd, vd = nv*hd, qkvd = qd*2+vd;
        float *qkvB = malloc((size_t)B*qkvd*4), *qkv2B = malloc((size_t)B*qkvd*4);
        float *zB = malloc((size_t)B*vd*4);
        float *aB = malloc((size_t)B*nv*4), *bB = malloc((size_t)B*nv*4);
        float *retB = malloc((size_t)B*vd*4), *oB = malloc((size_t)B*vd*4);
        float *qB = malloc((size_t)B*qd*4), *kB = malloc((size_t)B*qd*4), *vB = malloc((size_t)B*vd*4);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, xnB, qkvB, qkvd, H, B);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.in_proj_z.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, xnB, zB, vd, H, B);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.in_proj_a.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, xnB, aB, nv, H, B);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.in_proj_b.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, xnB, bB, nv, H, B);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.conv1d.weight", L);
        t = QF(tab, tmp);
        float *cw = malloc(40960*4); deq_small(t, cw, 40960);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.norm.weight", L);
        t = QF(tab, tmp); float qn[128]; deq_small(t, qn, 128);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.A_log", L);
        t = QF(tab, tmp); float alog[48]; deq_small(t, alog, 48);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.dt_bias", L);
        t = QF(tab, tmp); float dtb[48]; deq_small(t, dtb, 48);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.norm.weight", L);
        t = QF(tab, tmp); float nwgt[128]; deq_small(t, nwgt, 128);
        const float qs = 1.0f / sqrtf((float)hd);
        for (int b = 0; b < B; b++) {
            // conv carryover sekuensial (trajektori tunggal, urutan == single)
            vx27_conv1d_step(conv, qkvB + (size_t)b*qkvd, cw, qkv2B + (size_t)b*qkvd, qkvd, 1);
            float *q = qB + (size_t)b*qd, *k = kB + (size_t)b*qd, *v = vB + (size_t)b*vd;
            memcpy(q, qkv2B + (size_t)b*qkvd, (size_t)qd*4);
            memcpy(k, qkv2B + (size_t)b*qkvd + qd, (size_t)qd*4);
            memcpy(v, qkv2B + (size_t)b*qkvd + qd*2, (size_t)vd*4);
            for (int hh = 0; hh < nqk; hh++) {
                float tq[128], tk[128];
                rmsn(q+hh*hd, qn, tq, hd); rmsn(k+hh*hd, qn, tk, hd);
                vx27_deltanet_l2norm(tq, q+hh*hd, hd);
                vx27_deltanet_l2norm(tk, k+hh*hd, hd);
            }
            for (int i = 0; i < qd; i++) q[i] *= qs;
            for (int vh = 0; vh < nv; vh++) {
                int qk = vh * nqk / nv;
                double beta, gg;
                vx27_deltanet_gate_params((double)aB[(size_t)b*nv+vh], (double)alog[vh],
                    (double)bB[(size_t)b*nv+vh], (double)dtb[vh], &beta, &gg);
                vx27_deltanet_step_exact(S + (size_t)vh*hd*hd, q+(size_t)qk*hd,
                    k+(size_t)qk*hd, v+(size_t)vh*hd, beta, gg, retB + (size_t)b*vd + (size_t)vh*hd, hd);
            }
            float *o = oB + (size_t)b*vd;
            for (int vh = 0; vh < nv; vh++) {
                float tn[128];
                rmsn_std(retB+(size_t)b*vd+(size_t)vh*hd, nwgt, tn, hd);
                float zb = 0; (void)zb;
                for (int i = 0; i < hd; i++)
                    o[(size_t)vh*hd+i] = tn[i] * silu1(zB[(size_t)b*vd+(size_t)vh*hd+i]);
            }
            if (Sckpt) memcpy(Sckpt + (size_t)b*nv*hd*hd, S, (size_t)nv*hd*hd*4);
            if (Cckpt) memcpy(Cckpt + (size_t)b*qkvd*3, conv, (size_t)qkvd*3*4);
        }
        free(cw);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.linear_attn.out_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, oB, aoB, H, vd, B);
        free(qkvB); free(qkv2B); free(zB); free(aB); free(bB);
        free(retB); free(oB); free(qB); free(kB); free(vB);
    } else {
        const int nq = 24, nkv = 4, hd = 256, rd = 64;
        float *qgB = malloc((size_t)B*nq*hd*2*4), *qB = malloc((size_t)B*nq*hd*4), *gateB = malloc((size_t)B*nq*hd*4);
        float *k1B = malloc((size_t)B*nkv*hd*4), *v1B = malloc((size_t)B*nkv*hd*4);
        float *attB = malloc((size_t)B*nq*hd*4);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.q_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, xnB, qgB, nq*hd*2, H, B);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.k_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, xnB, k1B, nkv*hd, H, B);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.v_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, xnB, v1B, nkv*hd, H, B);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.q_norm.weight", L);
        t = QF(tab, tmp); float qnw[256]; deq_small(t, qnw, 256);
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.k_norm.weight", L);
        t = QF(tab, tmp); float knw[256]; deq_small(t, knw, 256);
        int slot = full_idx[L];
        float *kc0 = kvc + ((size_t)slot*VX27_QGEN_T)*nkv*hd;
        float *vc0 = kvc + ((size_t)16*VX27_QGEN_T*nkv*hd) + ((size_t)slot*VX27_QGEN_T)*nkv*hd;
        for (int b = 0; b < B; b++) {
            float *q = qB + (size_t)b*nq*hd, *gate = gateB + (size_t)b*nq*hd;
            float *k1 = k1B + (size_t)b*nkv*hd, *v1 = v1B + (size_t)b*nkv*hd;
            for (int hh = 0; hh < nq; hh++) {
                memcpy(q + (size_t)hh*hd, qgB + (size_t)b*nq*hd*2 + (size_t)hh*hd*2, (size_t)hd*4);
                memcpy(gate + (size_t)hh*hd, qgB + (size_t)b*nq*hd*2 + (size_t)hh*hd*2 + hd, (size_t)hd*4);
            }
            for (int hh = 0; hh < nq; hh++) { float tt[256]; rmsn(q+hh*hd, qnw, tt, hd); memcpy(q+hh*hd, tt, hd*4); }
            for (int hh = 0; hh < nkv; hh++) { float tt[256]; rmsn(k1+hh*hd, knw, tt, hd); memcpy(k1+hh*hd, tt, hd*4); }
            float cosv[32], sinv[32];
            vx27_rope_cos_sin(n_pos0 + b, cosv, sinv, 64, 10000000.0f, 1.0f);
            vx27_rope_apply(q, nq, hd, rd, cosv, sinv);
            vx27_rope_apply(k1, nkv, hd, rd, cosv, sinv);
            memcpy(kc0 + (size_t)(n_pos0+b)*nkv*hd, k1, (size_t)nkv*hd*4);
            memcpy(vc0 + (size_t)(n_pos0+b)*nkv*hd, v1, (size_t)nkv*hd*4);
            float *att = attB + (size_t)b*nq*hd;
            float scl = 1.0f / sqrtf((float)hd);
            int nq1 = n_pos0 + b + 1;
            for (int hh = 0; hh < nq; hh++) {
                int kvh = hh * nkv / nq;
                const float *qh = q + (size_t)hh*hd;
                float smax = -1e30f, ssum = 0.0f;
                // skor per posisi (buffer stack, T<=512)
                float sc[512];
                for (int pp = 0; pp < nq1; pp++) {
                    const float *kh = kc0 + (size_t)pp*nkv*hd + (size_t)kvh*hd;
                    double s = 0; for (int i = 0; i < hd; i++) s += (double)qh[i]*kh[i];
                    sc[pp] = (float)(s * scl);
                    if (sc[pp] > smax) smax = sc[pp];
                }
                for (int pp = 0; pp < nq1; pp++) { sc[pp] = expf(sc[pp]-smax); ssum += sc[pp]; }
                float *oh = att + (size_t)hh*hd;
                for (int i = 0; i < hd; i++) oh[i] = 0.0f;
                for (int pp = 0; pp < nq1; pp++) {
                    float a = sc[pp]/ssum;
                    const float *vh = vc0 + (size_t)pp*nkv*hd + (size_t)kvh*hd;
                    for (int i = 0; i < hd; i++) oh[i] += a*vh[i];
                }
            }
            for (int i = 0; i < nq*hd; i++) att[i] *= 1.0f / (1.0f + expf(-gate[i]));
        }
        snprintf(tmp, sizeof tmp, "model.language_model.layers.%d.self_attn.o_proj.weight", L);
        t = QF(tab, tmp); vx27_pmat_batch(tp, t->blob, t->dtype, attB, aoB, H, nq*hd, B);
        free(qgB); free(qB); free(gateB); free(k1B); free(v1B); free(attB);
    }
    // ekor bersama: residual + post-norm + MLP (batched proyeksi)
    for (int b = 0; b < B; b++)
        for (int i = 0; i < H; i++) x2B[(size_t)b*H+i] = xs[(size_t)b*H+i] + aoB[(size_t)b*H+i];
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.post_attention_layernorm.weight", L);
    t = QF(tab, nm); float pn[5120]; deq_small(t, pn, H);
    for (int b = 0; b < B; b++) rmsn(x2B + (size_t)b*H, pn, fnB + (size_t)b*H, H);
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.mlp.gate_proj.weight", L);
    t = QF(tab, nm); vx27_pmat_batch(tp, t->blob, t->dtype, fnB, gB, F, H, B);
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.mlp.up_proj.weight", L);
    t = QF(tab, nm); vx27_pmat_batch(tp, t->blob, t->dtype, fnB, uB, F, H, B);
    for (int b = 0; b < B; b++) vx27_swiglu_forward_avx2(gB + (size_t)b*F, uB + (size_t)b*F, hB + (size_t)b*F, F);
    snprintf(nm, sizeof nm, "model.language_model.layers.%d.mlp.down_proj.weight", L);
    t = QF(tab, nm); vx27_pmat_batch(tp, t->blob, t->dtype, hB, foB, H, F, B);
    for (int b = 0; b < B; b++)
        for (int i = 0; i < H; i++) outs[(size_t)b*H+i] = x2B[(size_t)b*H+i] + foB[(size_t)b*H+i];
    free(xnB); free(x2B); free(fnB); free(gB); free(uB); free(hB); free(foB); free(aoB);
}

int vx27_qlogits(const vx27_qtab *tab, const float *h, float *logits, threadpool *tp) {
    const int H = 5120, V = 248320;
    float *hn = malloc(H*4);
    const vx27_qt *t = QF(tab, "model.language_model.norm.weight");
    float wn[5120]; deq_small(t, wn, H);
    rmsn(h, wn, hn, H);
    t = QF(tab, "lm_head.weight");
    vx27_pmat(tp, t->blob, t->dtype, hn, logits, V, H);
    free(hn);
    int arg = 0;
    for (int i = 1; i < V; i++) if (logits[i] > logits[arg]) arg = i;
    return arg;
}

// Batch logits: lm_head 1 pass utk B hidden.
void vx27_qlogits_batch(const vx27_qtab *tab, const float *hs, float *lgs,
                 uint32_t *args, int B, threadpool *tp) {
    const int H = 5120, V = 248320;
    if (B < 1) return;
    if (B > 8) B = 8;
    float *hnB = malloc((size_t)B*H*4);
    const vx27_qt *t = QF(tab, "model.language_model.norm.weight");
    float wn[5120]; deq_small(t, wn, H);
    for (int b = 0; b < B; b++) rmsn(hs + (size_t)b*H, wn, hnB + (size_t)b*H, H);
    t = QF(tab, "lm_head.weight");
    vx27_pmat_batch(tp, t->blob, t->dtype, hnB, lgs, V, H, B);
    free(hnB);
    for (int b = 0; b < B; b++) {
        const float *lg = lgs + (size_t)b*V;
        uint32_t a = 0;
        for (int i = 1; i < V; i++) if (lg[i] > lg[a]) a = (uint32_t)i;
        args[b] = a;
    }
}

// dequant 1 baris embed BP3 (row-major: baris r = grup [r*40, r*40+40))
static void embed_row_bp3(const vx27_qt *emb, int row, float *out, int H) {
    const vx27_bp3_128 *G = (const vx27_bp3_128*)emb->blob + (size_t)row * (H/128);
    for (int k = 0; k < H/128; k++) {
        uint16_t h = G[k].scale_h; uint32_t u = h;
        uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
        if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
        else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
        else bits=sign|((exp+112u)<<23)|(mant<<13);
        float sc; memcpy(&sc,&bits,4);
        for (int i = 0; i < 128; i++) {
            uint32_t uu=((G[k].p[0*4+i/32]>>(i%32))&1)|(((G[k].p[1*4+i/32]>>(i%32))&1)<<1)|(((G[k].p[2*4+i/32]>>(i%32))&1)<<2);
            out[k*128+i] = (float)((uu>=4)?(int)uu-8:(int)uu) * sc;
        }
    }
}

void vx27_qgen(const vx27_qtab *tab, threadpool *tp, const float *x0,
               int T, int *out_ids, double *ms_total) {
    const int H = 5120;
    float *Sall = calloc((size_t)48*VX27_SSTATE_PER_LAYER, 4);
    float *Call = calloc((size_t)48*VX27_CSTATE_PER_LAYER, 4);
    float *kv = calloc((size_t)16*VX27_QGEN_T*4*256*2, 4);
    float *h1 = malloc(H*4), *h2 = malloc(H*4);
    float *logits = malloc(248320*4);
    for (int i = 0; i < H; i++) h1[i] = x0[i];
    const vx27_qt *emb = QF(tab, "model.language_model.embed_tokens.weight");
    int full_idx[64]; int fs = 0;
    for (int L = 0; L < 64; L++) { int f = ((L+1)%4==0); full_idx[L] = f ? fs++ : -1; }
    double t0 = now_ms();
    for (int t = 0; t < T; t++) {
        float cosv[32], sinv[32];
        vx27_rope_cos_sin(t, cosv, sinv, 64, 10000000.0f, 1.0f);
        float *Slin[64], *Clin[64]; int li = 0, ci = 0;
        for (int L = 0; L < 64; L++) if (!(((L+1)%4==0))) {
            Slin[L] = Sall + (size_t)(li++)*VX27_SSTATE_PER_LAYER;
            Clin[L] = Call + (size_t)(ci++)*VX27_CSTATE_PER_LAYER;
        } else { Slin[L] = NULL; Clin[L] = NULL; }
        for (int L = 0; L < 64; L++) {
            int is_full = ((L+1)%4==0);
            vx27_qlayer(tab, L, is_full, h1, h2, t, full_idx, Slin[L], kv, cosv, sinv, tp, Clin[L]);
            float *tt = h1; h1 = h2; h2 = tt;
        }
        int arg = vx27_qlogits(tab, h1, logits, tp);
        out_ids[t] = arg;
        if (t + 1 < T && emb) embed_row_bp3(emb, arg % 248320, h1, H);
    }
    double t1 = now_ms();
    if (ms_total) *ms_total = t1 - t0;
    free(Sall); free(Call); free(kv); free(h1); free(h2); free(logits);
}
