#include "vx27_head.h"
#include "vx27_config.h"
#include "vx27_ffn.h"
#include <immintrin.h>
#include <math.h>

uint64_t vx27_lmhead_params(void) {
    return (uint64_t)VX27_VOCAB * (uint64_t)VX27_HIDDEN;
}
double vx27_lmhead_gb(double bits) {
    return (double)vx27_lmhead_params() * bits / 8.0 / 1e9;
}
int vx27_lmhead_rows_gpu(double bits, double budget_gb, uint64_t *rows_gpu) {
    double total = vx27_lmhead_gb(bits);
    if (total <= budget_gb) { if (rows_gpu) *rows_gpu = VX27_VOCAB; return 1; }
    // rows proportional to budget (row-major, uniform bits)
    uint64_t r = (uint64_t)(VX27_VOCAB * budget_gb / total);
    if (rows_gpu) *rows_gpu = r;
    return 0; // needs split
}

void vx27_matvec_ref(const float *W, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        double s = 0;
        const float *row = W + (size_t)r * cols;
        for (int c = 0; c < cols; c++) s += (double)row[c] * x[c];
        y[r] = (float)s;
    }
}

void vx27_matvec_avx2(const float *W, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row = W + (size_t)r * cols;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 w = _mm256_loadu_ps(row + c);
            __m256 v = _mm256_loadu_ps(x + c);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(w, v, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(w, v));
#endif
        }
        float tmp[8]; _mm256_storeu_ps(tmp, acc);
        double s = (double)tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; c < cols; c++) s += (double)row[c] * x[c];
        y[r] = (float)s;
    }
}

static void rmsnorm_vec(const float *x, const float *w, float *y, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float s = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = x[i]*s*w[i];
}

void vx27_mtp_forward_ref(const float *h, const float *Wg, const float *Wu,
                          const float *Wd, const float *Wlm,
                          const float *norm_w, float *draft,
                          int hidden, int ffn_tiny, int vocab) {
    // double-precision reference for gate/up/down + residual + norm + lm
    double *g = (double*)0; (void)g;
    static float gt[8192], up[8192], hd[8192], h2[5120+32], res[5120+32];
    for (int i = 0; i < ffn_tiny; i++) {
        double sg = 0, su = 0;
        for (int j = 0; j < hidden; j++) { sg += (double)Wg[i*hidden+j]*h[j]; su += (double)Wu[i*hidden+j]*h[j]; }
        gt[i] = (float)sg; up[i] = (float)su;
    }
    vx27_swiglu_forward_ref(gt, up, hd, ffn_tiny);
    for (int j = 0; j < hidden; j++) {
        double s = 0; for (int i = 0; i < ffn_tiny; i++) s += (double)Wd[j*ffn_tiny+i]*hd[i];
        res[j] = h[j] + (float)s;
    }
    rmsnorm_vec(res, norm_w, h2, hidden);
    for (int r = 0; r < vocab; r++) {
        double s = 0; for (int j = 0; j < hidden; j++) s += (double)Wlm[r*hidden+j]*h2[j];
        draft[r] = (float)s;
    }
}

void vx27_mtp_forward(const float *h, const float *Wg, const float *Wu,
                      const float *Wd, const float *Wlm,
                      const float *norm_w, float *draft,
                      int hidden, int ffn_tiny, int vocab) {
    static float gt[8192], up[8192], hd[8192], h2[5120+32], res[5120+32];
    for (int i = 0; i < ffn_tiny; i++) {
        float sg = 0, su = 0;
        // tiny dims: scalar is fine; production uses quant matvec
        for (int j = 0; j < hidden; j++) { sg += Wg[i*hidden+j]*h[j]; su += Wu[i*hidden+j]*h[j]; }
        gt[i] = sg; up[i] = su;
    }
    vx27_swiglu_forward_avx2(gt, up, hd, ffn_tiny);
    for (int j = 0; j < hidden; j++) {
        float s = 0; for (int i = 0; i < ffn_tiny; i++) s += Wd[j*ffn_tiny+i]*hd[i];
        res[j] = h[j] + s;
    }
    rmsnorm_vec(res, norm_w, h2, hidden);
    vx27_matvec_avx2(Wlm, h2, draft, vocab, hidden);
}
