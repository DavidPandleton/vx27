#include "vx27_block.h"
#include "vx27_config.h"
#include "vx27_rope.h"
#include "vx27_gqa.h"
#include "vx27_ffn.h"
#include "vx27_deltanet.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ---- threadpool matvec: y = W[r*cols] @ x, rows split across workers ----
typedef struct {
    const float *W, *x;
    float *y;
    int rows, cols;
} tp_mv_job;

static void tp_mv_worker(void *arg, int tid, int nthreads) {
    tp_mv_job *j = (tp_mv_job *)arg;
    int rs = (int)((long long)j->rows * tid / nthreads);
    int re = (int)((long long)j->rows * (tid + 1) / nthreads);
    for (int r = rs; r < re; r++) {
        const float *row = j->W + (size_t)r * j->cols;
        double s = 0;
        for (int c = 0; c < j->cols; c++) s += (double)row[c] * j->x[c];
        j->y[r] = (float)s;
    }
}

static void matvec_tp(threadpool *tp, const float *W, const float *x, float *y, int rows, int cols) {
    if (!tp || rows < 64) {
        tp_mv_job j = {W, x, y, rows, cols};
        tp_mv_worker(&j, 0, 1);
        return;
    }
    tp_mv_job j = {W, x, y, rows, cols};
    tp_run(tp, tp_mv_worker, &j);
}

void vx27_full_block_forward(const vx27_block_cfg *bc, threadpool *tp,
    const float *x, float *kcache, float *vcache, int n_past,
    float *out, const vx27_block_weights *w) {
    int H = bc->hidden, F = bc->ffn, nq = bc->n_q, nkv = bc->n_kv, hd = bc->hd, rd = bc->rope_dim;
    // scratch on stack (tiny test dims; production uses caller buffers)
    float *xn = (float*)malloc((size_t)H*4);
    float *q = (float*)malloc((size_t)nq*hd*4);
    float *k1 = (float*)malloc((size_t)nkv*hd*4);
    float *v1 = (float*)malloc((size_t)nkv*hd*4);
    float *att = (float*)malloc((size_t)nq*hd*4);
    float *ao = (float*)malloc((size_t)H*4);
    float *x2 = (float*)malloc((size_t)H*4);
    float *fn = (float*)malloc((size_t)H*4);
    float *g = (float*)malloc((size_t)F*4);
    float *u = (float*)malloc((size_t)F*4);
    float *h = (float*)malloc((size_t)F*4);
    float *fo = (float*)malloc((size_t)H*4);
    float *cosv = (float*)malloc((size_t)(rd/2)*4);
    float *sinv = (float*)malloc((size_t)(rd/2)*4);

    vx27_rmsnorm(x, w->attn_norm_w, xn, H, 1e-6f);
    matvec_tp(tp, w->Wq, xn, q, nq*hd, H);
    matvec_tp(tp, w->Wk, xn, k1, nkv*hd, H);
    matvec_tp(tp, w->Wv, xn, v1, nkv*hd, H);
    // QK-norm per head if provided
    if (w->qk_norm_w) {
        for (int hh = 0; hh < nq; hh++) {
            float *qh = q + (size_t)hh*hd;
            float *tmp = (float*)malloc((size_t)hd*4);
            vx27_rmsnorm(qh, w->qk_norm_w, tmp, hd, 1e-6f);
            memcpy(qh, tmp, (size_t)hd*4); free(tmp);
        }
        for (int hh = 0; hh < nkv; hh++) {
            float *kh = k1 + (size_t)hh*hd;
            float *tmp = (float*)malloc((size_t)hd*4);
            vx27_rmsnorm(kh, w->qk_norm_w, tmp, hd, 1e-6f);
            memcpy(kh, tmp, (size_t)hd*4); free(tmp);
        }
    }
    vx27_rope_cos_sin(n_past, cosv, sinv, rd, bc->rope_theta, bc->yarn_factor);
    vx27_rope_apply(q, nq, hd, rd, cosv, sinv);
    vx27_rope_apply(k1, nkv, hd, rd, cosv, sinv);
    // append to cache
    memcpy(kcache + ((size_t)n_past*nkv*hd), k1, (size_t)nkv*hd*4);
    memcpy(vcache + ((size_t)n_past*nkv*hd), v1, (size_t)nkv*hd*4);
    // attention over n_past+1
    {
        float scale = 1.0f / sqrtf((float)hd);
        for (int hh = 0; hh < nq; hh++) {
            int kvh = hh * nkv / nq;
            const float *qh = q + (size_t)hh*hd;
            float smax = -1e30f, ssum = 0; static float sc[1024];
            for (int p = 0; p <= n_past; p++) {
                const float *kh = kcache + ((size_t)p*nkv+kvh)*hd;
                double s = 0; for (int i = 0; i < hd; i++) s += (double)qh[i]*kh[i];
                sc[p] = (float)(s * scale);
                if (sc[p] > smax) smax = sc[p];
            }
            for (int p = 0; p <= n_past; p++) { sc[p] = expf(sc[p]-smax); ssum += sc[p]; }
            float *oh = att + (size_t)hh*hd;
            for (int i = 0; i < hd; i++) oh[i] = 0;
            for (int p = 0; p <= n_past; p++) {
                float a = sc[p]/ssum;
                const float *vh = vcache + ((size_t)p*nkv+kvh)*hd;
                for (int i = 0; i < hd; i++) oh[i] += a*vh[i];
            }
        }
    }
    matvec_tp(tp, w->Wo, att, ao, H, nq*hd);
    for (int i = 0; i < H; i++) x2[i] = x[i] + ao[i];
    vx27_rmsnorm(x2, w->ffn_norm_w, fn, H, 1e-6f);
    matvec_tp(tp, w->Wg, fn, g, F, H);
    matvec_tp(tp, w->Wu, fn, u, F, H);
    vx27_swiglu_forward_avx2(g, u, h, F);
    matvec_tp(tp, w->Wd, h, fo, H, F);
    for (int i = 0; i < H; i++) out[i] = x2[i] + fo[i];

    free(xn); free(q); free(k1); free(v1); free(att); free(ao); free(x2);
    free(fn); free(g); free(u); free(h); free(fo); free(cosv); free(sinv);
}

void vx27_linear_block_forward(const vx27_block_cfg *bc, threadpool *tp,
    const float *x, float *S, float *out, const vx27_block_weights *w) {
    int H = bc->hidden, F = bc->ffn;
    int nqk = bc->lin_qk, nv = bc->lin_v, lhd = bc->lin_hd;
    float *xn = (float*)malloc((size_t)H*4);
    float *q = (float*)malloc((size_t)nqk*lhd*4);
    float *k = (float*)malloc((size_t)nqk*lhd*4);
    float *v = (float*)malloc((size_t)nv*lhd*4);
    float *ret = (float*)malloc((size_t)nv*lhd*4);
    float *ao = (float*)malloc((size_t)H*4);
    float *x2 = (float*)malloc((size_t)H*4);
    float *fn = (float*)malloc((size_t)H*4);
    float *g = (float*)malloc((size_t)F*4);
    float *u = (float*)malloc((size_t)F*4);
    float *h = (float*)malloc((size_t)F*4);
    float *fo = (float*)malloc((size_t)H*4);

    vx27_rmsnorm(x, w->attn_norm_w, xn, H, 1e-6f);
    matvec_tp(tp, w->Wq, xn, q, nqk*lhd, H);
    matvec_tp(tp, w->Wk, xn, k, nqk*lhd, H);
    matvec_tp(tp, w->Wv, xn, v, nv*lhd, H);
    // grouped DeltaNet: nv assumed == nqk or multiple; map v-group = q-group*(nv/nqk)
    int ratio = nv / (nqk > 0 ? nqk : 1);
    if (ratio < 1) ratio = 1;
    for (int grp = 0; grp < nqk; grp++) {
        float *Sg = S + (size_t)grp*lhd*lhd;
        const float *kg = k + (size_t)grp*lhd;
        const float *qg = q + (size_t)grp*lhd;
        // update with first v in group (simplified proxy), retrieve per v in group
        int v0 = grp * ratio;
        if (v0 >= nv) v0 = nv - 1;
        const float *vg = v + (size_t)v0*lhd;
        vx27_deltanet_update_avx2(Sg, kg, vg, 0.99f, lhd);
        for (int vi = 0; vi < ratio && grp*ratio+vi < nv; vi++) {
            float *rv = ret + (size_t)(grp*ratio+vi)*lhd;
            vx27_deltanet_retrieve_avx2(Sg, qg, rv, lhd);
        }
    }
    matvec_tp(tp, w->Wo, ret, ao, H, nv*lhd);
    for (int i = 0; i < H; i++) x2[i] = x[i] + ao[i];
    vx27_rmsnorm(x2, w->ffn_norm_w, fn, H, 1e-6f);
    matvec_tp(tp, w->Wg, fn, g, F, H);
    matvec_tp(tp, w->Wu, fn, u, F, H);
    vx27_swiglu_forward_avx2(g, u, h, F);
    matvec_tp(tp, w->Wd, h, fo, H, F);
    for (int i = 0; i < H; i++) out[i] = x2[i] + fo[i];

    free(xn); free(q); free(k); free(v); free(ret); free(ao); free(x2);
    free(fn); free(g); free(u); free(h); free(fo);
}

void vx27_block_estimate(int is_full, double *flops, double *weight_bytes_q4) {
    // Real dims: hidden 5120, ffn 17408
    double H = 5120, F = 17408;
    double attn_params, ffn_params = 3*H*F;
    if (is_full) {
        double nqhd = 24*256, nkvhd = 4*256;
        attn_params = H*nqhd + H*nkvhd*2 + nqhd*H;
    } else {
        double qk = 16*128, vv = 48*128;
        attn_params = H*qk*2 + H*vv + vv*H;
    }
    double total = attn_params + ffn_params;
    if (weight_bytes_q4) *weight_bytes_q4 = total * 0.5;
    // ~2 FLOP per param per token (matvec)
    if (flops) *flops = 2 * total;
}
