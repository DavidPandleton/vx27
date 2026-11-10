#include "vx27_gqa.h"
#include "vx27_config.h"
#include <math.h>
#include <string.h>

int vx27_gqa_kv_for_q(int q_head) {
    return q_head * VX27_ATTN_KV_HEADS / VX27_ATTN_Q_HEADS;
}

void vx27_rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float s = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * s * w[i];
}

void vx27_attn_1q(const float *q, const float *kcache, const float *vcache, int n_pos,
                  float *out) {
    const int nh = VX27_ATTN_Q_HEADS, nkv = VX27_ATTN_KV_HEADS, hd = VX27_ATTN_HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)hd);
    // scores per head, small n_pos in tests
    for (int h = 0; h < nh; h++) {
        int kvh = vx27_gqa_kv_for_q(h);
        const float *qh = q + (size_t)h * hd;
        float smax = -1e30f, ssum = 0.0f;
        // two-pass softmax over n_pos; store scores on stack (n_pos<=512 in tests)
        static float sc[1024];
        for (int p = 0; p < n_pos; p++) {
            const float *kh = kcache + ((size_t)p * nkv + kvh) * hd;
            float s = 0.0f;
            for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
            s *= scale;
            sc[p] = s;
            if (s > smax) smax = s;
        }
        for (int p = 0; p < n_pos; p++) { sc[p] = expf(sc[p] - smax); ssum += sc[p]; }
        float *oh = out + (size_t)h * hd;
        for (int i = 0; i < hd; i++) oh[i] = 0.0f;
        for (int p = 0; p < n_pos; p++) {
            float a = sc[p] / ssum;
            const float *vh = vcache + ((size_t)p * nkv + kvh) * hd;
            for (int i = 0; i < hd; i++) oh[i] += a * vh[i];
        }
    }
}
