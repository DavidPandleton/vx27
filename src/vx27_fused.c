#include "vx27_fused.h"
#include "vx27_ffn.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void rmsn(const float *x, const float *w, float *y, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float s = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = x[i]*s*(w?w[i]:1.0f);
}
static float ssum_silu(float v) { return v / (1.0f + expf(-v)); }

typedef struct { const float *W; const float *x; float *y; int rows, cols; } mvj;
static void mv_worker(void *a, int tid, int nth) {
    mvj *j = (mvj*)a;
    int rs = (int)((long long)j->rows*tid/nth), re = (int)((long long)j->rows*(tid+1)/nth);
    for (int r = rs; r < re; r++) {
        const float *row = j->W + (size_t)r*j->cols;
        double s = 0; for (int c = 0; c < j->cols; c++) s += (double)row[c]*j->x[c];
        j->y[r] = (float)s;
    }
}
static void matvec(threadpool *tp, const float *W, const float *x, float *y, int rows, int cols) {
    if (tp && rows >= 256) { mvj j = {W,x,y,rows,cols}; tp_run(tp, mv_worker, &j); }
    else { mvj j = {W,x,y,rows,cols}; mv_worker(&j, 0, 1); }
}

void vx27_fused_delta_forward(const vx27_fused_w *w, vx27_fused_state *st,
                              const float *x, float *out, threadpool *tp) {
    int H = w->H, nqk = w->nqk, nv = w->nv, hd = w->hd, F = w->F;
    int qd = nqk*hd, vd = nv*hd, qkvd = qd*2 + vd;
    int ratio = nv / (nqk > 0 ? nqk : 1); if (ratio < 1) ratio = 1;

    float *xn = malloc((size_t)H*4);
    float *qkv = malloc((size_t)qkvd*4);
    float *z = malloc((size_t)vd*4);
    float *avec = malloc((size_t)nv*4), *bvec = malloc((size_t)nv*4);
    float *ret = malloc((size_t)vd*4);
    float *ao = malloc((size_t)H*4);
    float *x2 = malloc((size_t)H*4);
    float *fn = malloc((size_t)H*4);
    float *g = malloc((size_t)F*4), *u = malloc((size_t)F*4), *h = malloc((size_t)F*4);
    float *fo = malloc((size_t)H*4);
    float *o = malloc((size_t)vd*4);
    float *on = malloc((size_t)vd*4);

    rmsn(x, w->attn_norm_w, xn, H);
    matvec(tp, w->Wqkv, xn, qkv, qkvd, H);
    matvec(tp, w->Wz, xn, z, vd, H);
    matvec(tp, w->Wa, xn, avec, nv, H);
    matvec(tp, w->Wb, xn, bvec, nv, H);

    // conv1d depthwise k=4 + SiLU pada qkv (state nol utk token tunggal:
    // conv = w0*x + w3*s2 + w2*s1 + w1*s0 dengan s*=0 -> w0*x)
    float *q = malloc((size_t)qd*4), *k = malloc((size_t)qd*4), *v = malloc((size_t)vd*4);
    for (int i = 0; i < qkvd; i++) {
        float c0 = w->conv ? w->conv[i*4+0] : 1.0f;
        qkv[i] = ssum_silu(qkv[i] * c0);
    }
    memcpy(q, qkv, (size_t)qd*4);
    memcpy(k, qkv + qd, (size_t)qd*4);
    memcpy(v, qkv + qd*2, (size_t)vd*4);
    // QK-norm per head bila ada
    if (w->qknorm) {
        for (int hh = 0; hh < nqk; hh++) {
            float *qh = q + (size_t)hh*hd, *kh = k + (size_t)hh*hd, t[256];
            rmsn(qh, w->qknorm, t, hd); memcpy(qh, t, (size_t)hd*4);
            rmsn(kh, w->qknorm, t, hd); memcpy(kh, t, (size_t)hd*4);
        }
    }
    // recurrent per grup (QK 16 grup, tiap grup `ratio` V-head berbagi k/q grup)
    for (int grp = 0; grp < nqk; grp++) {
        float *Sg = st->S + (size_t)grp*hd*hd;
        const float *qg = q + (size_t)grp*hd, *kg = k + (size_t)grp*hd;
        for (int vi = 0; vi < ratio && grp*ratio+vi < nv; vi++) {
            int vh = grp*ratio+vi;
            float bh = 1.0f / (1.0f + expf(-(bvec[vh] + (w->dt_bias ? w->dt_bias[vh] : 0.0f))));
            float ah = expf(-expf((w->Alog ? w->Alog[vh] : 0.0f) + avec[vh]));
            const float *vv = v + (size_t)vh*hd;
            float *rv = ret + (size_t)vh*hd;
            // Sk dari state terkini, lalu delta-rule update, lalu retrieve
            float sk[256];
            for (int j = 0; j < hd; j++) { double s = 0; for (int i2 = 0; i2 < hd; i2++) s += (double)Sg[i2*hd+j]*kg[i2]; sk[j] = (float)s; }
            for (int i2 = 0; i2 < hd; i2++)
                for (int j = 0; j < hd; j++)
                    Sg[i2*hd+j] = ah*Sg[i2*hd+j] + bh*(vv[i2]-ah*sk[j])*kg[j];
            for (int j = 0; j < hd; j++) { double s = 0; for (int i2 = 0; i2 < hd; i2++) s += (double)Sg[i2*hd+j]*qg[j]; rv[j] = (float)s; }
        }
    }
    // norm + z-gate + out_proj + residual
    rmsn(ret, NULL, on, vd);
    for (int i = 0; i < vd; i++) o[i] = on[i] * ssum_silu(z[i]);
    matvec(tp, w->Wo, o, ao, H, vd);
    for (int i = 0; i < H; i++) x2[i] = x[i] + ao[i];
    rmsn(x2, w->post_norm_w, fn, H);
    matvec(tp, w->Wg, fn, g, F, H);
    matvec(tp, w->Wu, fn, u, F, H);
    extern void vx27_swiglu_forward_avx2(const float *, const float *, float *, int);
    vx27_swiglu_forward_avx2(g, u, h, F);
    matvec(tp, w->Wd, h, fo, H, F);
    for (int i = 0; i < H; i++) out[i] = x2[i] + fo[i];

    free(xn); free(qkv); free(z); free(avec); free(bvec); free(ret);
    free(ao); free(x2); free(fn); free(g); free(u); free(h); free(fo);
    free(o); free(on); free(q); free(k); free(v);
}
