#include "vx27_mtp.h"
#include "vx27_pmat.h"
#include "vx27_ffn.h"
#include "vx27_bp3.h"
#include "vx27_rope.h"
#include "quant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern void vx27_bp3_128_unpack(const vx27_bp3_128 *g, float *w); // di vx27_qfwd.c

static void mtp_rmsn(const float *x, const float *w, float *y, int n) {
    // Qwen3_5RMSNorm zero-centered, sama seperti jalur utama (1+w).
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float s = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    if (w) for (int i = 0; i < n; i++) y[i] = x[i]*s*(w[i]+1.0f);
    else for (int i = 0; i < n; i++) y[i] = x[i]*s;
}

static void mtp_deq_small(const vx27_qt *t, float *out, int n) {
    if (t->dtype == 0) { memcpy(out, t->blob, (size_t)n*4); return; }
    if (t->dtype == 2) {
        const block_q8_k *q = (const block_q8_k*)t->blob;
        for (int b = 0; b < n/256; b++) for (int i = 0; i < 256; i++) out[b*256+i] = (float)q[b].qs[i]*q[b].d;
        return;
    }
    for (int k = 0; k < n/128; k++) vx27_bp3_128_unpack(&((const vx27_bp3_128*)t->blob)[k], out + k*128);
}

typedef struct { const char *suf; int r, c; } mtp_need;
static const mtp_need NEED[] = {
    {"mtp.fc.weight", 5120, 10240},
    {"mtp.layers.0.input_layernorm.weight", 5120, 1},
    {"mtp.layers.0.mlp.down_proj.weight", 5120, 17408},
    {"mtp.layers.0.mlp.gate_proj.weight", 17408, 5120},
    {"mtp.layers.0.mlp.up_proj.weight", 17408, 5120},
    {"mtp.layers.0.post_attention_layernorm.weight", 5120, 1},
    {"mtp.layers.0.self_attn.k_norm.weight", 256, 1},
    {"mtp.layers.0.self_attn.k_proj.weight", 1024, 5120},
    {"mtp.layers.0.self_attn.o_proj.weight", 5120, 6144},
    {"mtp.layers.0.self_attn.q_norm.weight", 256, 1},
    {"mtp.layers.0.self_attn.q_proj.weight", 12288, 5120},
    {"mtp.layers.0.self_attn.v_proj.weight", 1024, 5120},
    {"mtp.norm.weight", 5120, 1},
    {"mtp.pre_fc_norm_embedding.weight", 5120, 1},
    {"mtp.pre_fc_norm_hidden.weight", 5120, 1},
};
#define MTP_N (int)(sizeof(NEED)/sizeof(NEED[0]))

int vx27_mtp_probe(const vx27_qtab *tab, char *err, size_t errlen) {
    int found = 0;
    for (int i = 0; i < MTP_N; i++) {
        const vx27_qt *t = vx27_qfind(tab, NEED[i].suf);
        if (!t) { if (err) snprintf(err, errlen, "mtp hilang: %s", NEED[i].suf); return -1; }
        if (t->rows != NEED[i].r || t->cols != NEED[i].c) {
            if (err) snprintf(err, errlen, "mtp bentuk salah: %s [%d x %d], mau [%d x %d]",
                NEED[i].suf, t->rows, t->cols, NEED[i].r, NEED[i].c);
            return -1;
        }
        found++;
    }
    return found;
}

int vx27_mtp_draft(const vx27_qtab *tab, threadpool *tp,
                   const float *h_prev, const float *emb,
                   float *draft_logits, int order_he, int use_gate, int ablate,
                   char *err, size_t errlen) {
    const vx27_qt *fc = vx27_qfind(tab, "mtp.fc.weight");
    const vx27_qt *pn_e = vx27_qfind(tab, "mtp.pre_fc_norm_embedding.weight");
    const vx27_qt *pn_h = vx27_qfind(tab, "mtp.pre_fc_norm_hidden.weight");
    const vx27_qt *iln = vx27_qfind(tab, "mtp.layers.0.input_layernorm.weight");
    const vx27_qt *qp = vx27_qfind(tab, "mtp.layers.0.self_attn.q_proj.weight");
    const vx27_qt *vp = vx27_qfind(tab, "mtp.layers.0.self_attn.v_proj.weight");
    const vx27_qt *op = vx27_qfind(tab, "mtp.layers.0.self_attn.o_proj.weight");
    const vx27_qt *pln = vx27_qfind(tab, "mtp.layers.0.post_attention_layernorm.weight");
    const vx27_qt *gp = vx27_qfind(tab, "mtp.layers.0.mlp.gate_proj.weight");
    const vx27_qt *up = vx27_qfind(tab, "mtp.layers.0.mlp.up_proj.weight");
    const vx27_qt *dp = vx27_qfind(tab, "mtp.layers.0.mlp.down_proj.weight");
    const vx27_qt *fn = vx27_qfind(tab, "mtp.norm.weight");
    const vx27_qt *lm = vx27_qfind(tab, "lm_head.weight");
    if (!fc || !pn_e || !pn_h || !iln || !qp || !vp || !op || !pln || !gp || !up || !dp || !fn || !lm) {
        if (err) snprintf(err, errlen, "mtp tensor hilang");
        return -1;
    }
    float *e_n = malloc(5120*4), *h_n = malloc(5120*4), *xb = malloc(10240*4);
    float *x = malloc(5120*4), *xn = malloc(5120*4);
    float *qf = malloc(12288*4), *vv = malloc(1024*4), *vr = malloc(6144*4), *ao = malloc(5120*4);
    float *y = malloc(5120*4), *yn = malloc(5120*4);
    float *g = malloc(17408*4), *u = malloc(17408*4), *hd = malloc(17408*4), *fo = malloc(5120*4);
    float *z = malloc(5120*4), *o = malloc(5120*4);
    float *we = malloc(5120*4), *wh = malloc(5120*4), *wi = malloc(5120*4), *wp = malloc(5120*4), *wf = malloc(5120*4);
    if (!e_n || !h_n || !xb || !x || !xn || !qf || !vv || !vr || !ao || !y || !yn ||
        !g || !u || !hd || !fo || !z || !o || !we || !wh || !wi || !wp || !wf) {
        if (err) snprintf(err, errlen, "mtp oom");
        goto fail;
    }
    mtp_deq_small(pn_e, we, 5120); mtp_deq_small(pn_h, wh, 5120);
    mtp_deq_small(iln, wi, 5120); mtp_deq_small(pln, wp, 5120); mtp_deq_small(fn, wf, 5120);
    if (ablate & 8) { for (int i = 0; i < 5120; i++) { we[i] = 1.0f; wh[i] = 1.0f; } }
    if (ablate & 2) { for (int i = 0; i < 5120; i++) { wi[i] = 1.0f; wp[i] = 1.0f; wf[i] = 1.0f; } }
    mtp_rmsn(emb, we, e_n, 5120);
    mtp_rmsn(h_prev, wh, h_n, 5120);
    if (order_he) { memcpy(xb, h_n, 5120*4); memcpy(xb + 5120, e_n, 5120*4); }
    else { memcpy(xb, e_n, 5120*4); memcpy(xb + 5120, h_n, 5120*4); }
    vx27_pmat(tp, fc->blob, fc->dtype, xb, x, 5120, 10240);

    if (ablate & 1) {
        // Lewati blok attention + MLP: o = norm(x). Diagnostik input pairing.
        // Bit 4 (final-norm saja dilewati) ditangani di bawah via wf==NULL marker:
        // di sini o = norm(x) penuh bila bit 4 tidak diset.
        if (ablate & 4) { memcpy(o, x, 5120*4); }
        else mtp_rmsn(x, wf, o, 5120);
        vx27_pmat(tp, lm->blob, lm->dtype, o, draft_logits, 248320, 5120);
        goto done_ok;
    }

    // decoder layer (single-token: attn_out = repeat(v) * sigmoid(gate))
    mtp_rmsn(x, wi, xn, 5120);
    vx27_pmat(tp, vp->blob, vp->dtype, xn, vv, 1024, 5120);
    vx27_pmat(tp, qp->blob, qp->dtype, xn, qf, 12288, 5120);
    // gate: per head h (24x256): baris q h*512+256 .. h*512+512
    for (int h = 0; h < 24; h++) {
        int kvh = h * 4 / 24;
        for (int i = 0; i < 256; i++) {
            float gv = qf[(size_t)h*512 + 256 + i];
            float gate = use_gate ? 1.0f / (1.0f + expf(-gv)) : 1.0f;
            vr[(size_t)h*256 + i] = vv[(size_t)kvh*256 + i] * gate;
        }
    }
    vx27_pmat(tp, op->blob, op->dtype, vr, ao, 5120, 6144);
    for (int i = 0; i < 5120; i++) y[i] = x[i] + ao[i];
    mtp_rmsn(y, wp, yn, 5120);
    vx27_pmat(tp, gp->blob, gp->dtype, yn, g, 17408, 5120);
    vx27_pmat(tp, up->blob, up->dtype, yn, u, 17408, 5120);
    vx27_swiglu_forward_avx2(g, u, hd, 17408);
    vx27_pmat(tp, dp->blob, dp->dtype, hd, fo, 5120, 17408);
    for (int i = 0; i < 5120; i++) z[i] = y[i] + fo[i];
    if (ablate & 4) memcpy(o, z, 5120*4);
    else mtp_rmsn(z, wf, o, 5120);
    vx27_pmat(tp, lm->blob, lm->dtype, o, draft_logits, 248320, 5120);

done_ok:
    free(e_n); free(h_n); free(xb); free(x); free(xn);
    free(qf); free(vv); free(vr); free(ao); free(y); free(yn);
    free(g); free(u); free(hd); free(fo); free(z); free(o);
    free(we); free(wh); free(wi); free(wp); free(wf);
    return 0;
fail:
    free(e_n); free(h_n); free(xb); free(x); free(xn);
    free(qf); free(vv); free(vr); free(ao); free(y); free(yn);
    free(g); free(u); free(hd); free(fo); free(z); free(o);
    free(we); free(wh); free(wi); free(wp); free(wf);
    return -1;
}

// Decoder MTP kausal dgn KV-cache sendiri (fidelity skema vLLM/EAGLE).
// Menggantikan aproksimasi single-token (softmax=1 -> out=v) dgn attention
// penuh atas riwayat 0..pos + q_norm/k_norm + RoPE + gate + o_proj.
int vx27_mtp_step_kv(const vx27_qtab *tab, threadpool *tp,
                   const float *h_prev, const float *emb, float *kvc, int pos, int T,
                   float *feat_out, float *logits_out, int order_he, int use_gate,
                   char *err, size_t errlen) {
    const int H = 5120, nq = 24, nkv = 4, hd = 256, rd = 64;
    if (!tab || !h_prev || !emb || !kvc || pos < 0 || pos >= T || T < 1) {
        if (err) snprintf(err, errlen, "mtp_kv arg buruk");
        return -1;
    }
    const vx27_qt *fc = vx27_qfind(tab, "mtp.fc.weight");
    const vx27_qt *pn_e = vx27_qfind(tab, "mtp.pre_fc_norm_embedding.weight");
    const vx27_qt *pn_h = vx27_qfind(tab, "mtp.pre_fc_norm_hidden.weight");
    const vx27_qt *iln = vx27_qfind(tab, "mtp.layers.0.input_layernorm.weight");
    const vx27_qt *qp = vx27_qfind(tab, "mtp.layers.0.self_attn.q_proj.weight");
    const vx27_qt *kp = vx27_qfind(tab, "mtp.layers.0.self_attn.k_proj.weight");
    const vx27_qt *vp = vx27_qfind(tab, "mtp.layers.0.self_attn.v_proj.weight");
    const vx27_qt *qnw = vx27_qfind(tab, "mtp.layers.0.self_attn.q_norm.weight");
    const vx27_qt *knw = vx27_qfind(tab, "mtp.layers.0.self_attn.k_norm.weight");
    const vx27_qt *op = vx27_qfind(tab, "mtp.layers.0.self_attn.o_proj.weight");
    const vx27_qt *pln = vx27_qfind(tab, "mtp.layers.0.post_attention_layernorm.weight");
    const vx27_qt *gp = vx27_qfind(tab, "mtp.layers.0.mlp.gate_proj.weight");
    const vx27_qt *up = vx27_qfind(tab, "mtp.layers.0.mlp.up_proj.weight");
    const vx27_qt *dp = vx27_qfind(tab, "mtp.layers.0.mlp.down_proj.weight");
    const vx27_qt *fn = vx27_qfind(tab, "mtp.norm.weight");
    const vx27_qt *lm = vx27_qfind(tab, "lm_head.weight");
    if (!fc || !pn_e || !pn_h || !iln || !qp || !kp || !vp || !qnw || !knw || !op ||
        !pln || !gp || !up || !dp || !fn || !lm) {
        if (err) snprintf(err, errlen, "mtp tensor hilang (kv)");
        return -1;
    }
    float *e_n = malloc(H*4), *h_n = malloc(H*4), *xb = malloc(10240*4);
    float *x = malloc(H*4), *xn = malloc(H*4);
    float *qf = malloc(12288*4), *q = malloc(nq*hd*4), *gate = malloc(nq*hd*4);
    float *k1 = malloc(nkv*hd*4), *v1 = malloc(nkv*hd*4);
    float *att = malloc(nq*hd*4), *ao = malloc(H*4);
    float *y = malloc(H*4), *yn = malloc(H*4);
    float *g = malloc(17408*4), *u = malloc(17408*4), *hdn = malloc(17408*4), *fo = malloc(H*4);
    float *z = malloc(H*4), *o = malloc(H*4);
    float *we = malloc(H*4), *wh = malloc(H*4), *wi = malloc(H*4), *wp = malloc(H*4), *wf = malloc(H*4);
    float *wqn = malloc(256*4), *wkn = malloc(256*4);
    if (!e_n || !h_n || !xb || !x || !xn || !qf || !q || !gate || !k1 || !v1 ||
        !att || !ao || !y || !yn || !g || !u || !hdn || !fo || !z || !o ||
        !we || !wh || !wi || !wp || !wf || !wqn || !wkn) {
        if (err) snprintf(err, errlen, "mtp_kv oom");
        goto fail;
    }
    mtp_deq_small(pn_e, we, H); mtp_deq_small(pn_h, wh, H);
    mtp_deq_small(iln, wi, H); mtp_deq_small(pln, wp, H); mtp_deq_small(fn, wf, H);
    mtp_deq_small(qnw, wqn, 256); mtp_deq_small(knw, wkn, 256);
    mtp_rmsn(emb, we, e_n, H);
    mtp_rmsn(h_prev, wh, h_n, H);
    if (order_he) { memcpy(xb, h_n, H*4); memcpy(xb + H, e_n, H*4); }
    else { memcpy(xb, e_n, H*4); memcpy(xb + H, h_n, H*4); }
    vx27_pmat(tp, fc->blob, fc->dtype, xb, x, H, 10240);
    // decoder: q/k/v + norm + rope + append + causal attn + gate
    mtp_rmsn(x, wi, xn, H);
    vx27_pmat(tp, qp->blob, qp->dtype, xn, qf, 12288, H);
    vx27_pmat(tp, kp->blob, kp->dtype, xn, k1, nkv*hd, H);
    vx27_pmat(tp, vp->blob, vp->dtype, xn, v1, nkv*hd, H);
    for (int hh = 0; hh < nq; hh++) {
        memcpy(q + (size_t)hh*hd, qf + (size_t)hh*hd*2, (size_t)hd*4);
        memcpy(gate + (size_t)hh*hd, qf + (size_t)hh*hd*2 + hd, (size_t)hd*4);
    }
    for (int hh = 0; hh < nq; hh++) { float tt[256]; mtp_rmsn(q+hh*hd, wqn, tt, hd); memcpy(q+hh*hd, tt, hd*4); }
    for (int hh = 0; hh < nkv; hh++) { float tt[256]; mtp_rmsn(k1+hh*hd, wkn, tt, hd); memcpy(k1+hh*hd, tt, hd*4); }
    {
        float cosv[32], sinv[32];
        vx27_rope_cos_sin(pos, cosv, sinv, 64, 10000000.0f, 1.0f);
        vx27_rope_apply(q, nq, hd, rd, cosv, sinv);
        vx27_rope_apply(k1, nkv, hd, rd, cosv, sinv);
    }
    float *kc = kvc + (size_t)pos*nkv*hd;
    float *vc = kvc + (size_t)T*nkv*hd + (size_t)pos*nkv*hd;
    memcpy(kc, k1, (size_t)nkv*hd*4); memcpy(vc, v1, (size_t)nkv*hd*4);
    {
        float scl = 1.0f / sqrtf((float)hd);
        float sc[512];
        for (int hh = 0; hh < nq; hh++) {
            int kvh = hh * nkv / nq;
            const float *qh = q + (size_t)hh*hd;
            float smax = -1e30f, ssum = 0.0f;
            for (int pp = 0; pp <= pos; pp++) {
                const float *kh = kvc + (size_t)pp*nkv*hd + (size_t)kvh*hd;
                double s = 0; for (int i = 0; i < hd; i++) s += (double)qh[i]*kh[i];
                sc[pp] = (float)(s * scl);
                if (sc[pp] > smax) smax = sc[pp];
            }
            for (int pp = 0; pp <= pos; pp++) { sc[pp] = expf(sc[pp]-smax); ssum += sc[pp]; }
            float *oh = att + (size_t)hh*hd;
            for (int i = 0; i < hd; i++) oh[i] = 0.0f;
            for (int pp = 0; pp <= pos; pp++) {
                float a = sc[pp]/ssum;
                const float *vh = kvc + (size_t)T*nkv*hd + (size_t)pp*nkv*hd + (size_t)kvh*hd;
                for (int i = 0; i < hd; i++) oh[i] += a*vh[i];
            }
            float *gh = gate + (size_t)hh*hd;
            if (use_gate) for (int i = 0; i < hd; i++) oh[i] *= 1.0f / (1.0f + expf(-gh[i]));
        }
    }
    vx27_pmat(tp, op->blob, op->dtype, att, ao, H, nq*hd);
    for (int i = 0; i < H; i++) y[i] = x[i] + ao[i];
    mtp_rmsn(y, wp, yn, H);
    vx27_pmat(tp, gp->blob, gp->dtype, yn, g, 17408, H);
    vx27_pmat(tp, up->blob, up->dtype, yn, u, 17408, H);
    vx27_swiglu_forward_avx2(g, u, hdn, 17408);
    vx27_pmat(tp, dp->blob, dp->dtype, hdn, fo, H, 17408);
    for (int i = 0; i < H; i++) z[i] = y[i] + fo[i];
    mtp_rmsn(z, wf, o, H);
    if (feat_out) memcpy(feat_out, o, H*4);
    if (logits_out) vx27_pmat(tp, lm->blob, lm->dtype, o, logits_out, 248320, H);
    free(e_n); free(h_n); free(xb); free(x); free(xn);
    free(qf); free(q); free(gate); free(k1); free(v1);
    free(att); free(ao); free(y); free(yn);
    free(g); free(u); free(hdn); free(fo); free(z); free(o);
    free(we); free(wh); free(wi); free(wp); free(wf);
    free(wqn); free(wkn);
    return 0;
fail:
    free(e_n); free(h_n); free(xb); free(x); free(xn);
    free(qf); free(q); free(gate); free(k1); free(v1);
    free(att); free(ao); free(y); free(yn);
    free(g); free(u); free(hdn); free(fo); free(z); free(o);
    free(we); free(wh); free(wi); free(wp); free(wf);
    free(wqn); free(wkn);
    return -1;
}
