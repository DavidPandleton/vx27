#include "quant.h"

void dequant_q4_k(const block_q4_k *b, float *out) {
    const float d = fp16_to_f32(b->d);
    const float min = fp16_to_f32(b->dmin);
    const uint8_t *q = b->qs;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, b->scales, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = min * (float)m;
        get_scale_min_k4(is + 1, b->scales, &sc, &m);
        float d2 = d * (float)sc;
        float m2 = min * (float)m;
        for (int l = 0; l < 32; l++) out[j + l] = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++)
            out[j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

void dequant_q6_k(const block_q6_k *b, float *out) {
    const float d = fp16_to_f32(b->d);
    const uint8_t *ql = b->ql;
    const uint8_t *qh = b->qh;
    const int8_t *sc = b->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[n + l + 0] = d * (float)sc[is + 0] * (float)q1;
            out[n + l + 32] = d * (float)sc[is + 2] * (float)q2;
            out[n + l + 64] = d * (float)sc[is + 4] * (float)q3;
            out[n + l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

float dot_q4_k_f32(const block_q4_k *b, const float *x, size_t nblocks) {
    float sum = 0.0f;
    for (size_t blk = 0; blk < nblocks; blk++) {
        const block_q4_k *q = b + blk;
        const float d = fp16_to_f32(q->d);
        const float min = fp16_to_f32(q->dmin);
        const float *xp = x + blk * QK_K;
        for (int j = 0; j < 8; j++) {
            uint8_t sc, m;
            get_scale_min_k4(j, q->scales, &sc, &m);
            float dd = d * (float)sc;
            float mm = min * (float)m;
            const uint8_t *qs = q->qs + 32 * (j / 2);
            const float *xq = xp + j * 32;
            if (j & 1) {
                for (int i = 0; i < 32; i++)
                    sum += (dd * (float)(qs[i] >> 4) - mm) * xq[i];
            } else {
                for (int i = 0; i < 32; i++)
                    sum += (dd * (float)(qs[i] & 0xF) - mm) * xq[i];
            }
        }
    }
    return sum;
}

float dot_q6_k_f32(const block_q6_k *b, const float *x, size_t nblocks) {
    float sum = 0.0f;
    for (size_t blk = 0; blk < nblocks; blk++) {
        const block_q6_k *q = b + blk;
        const float d = fp16_to_f32(q->d);
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql = q->ql + half * 64;
            const uint8_t *qh = q->qh + half * 32;
            const int8_t *sc = q->scales + half * 8;
            const float *xp = x + blk * QK_K + half * 128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * (float)sc[is + 0] * (float)q1 * xp[l + 0];
                sum += d * (float)sc[is + 2] * (float)q2 * xp[l + 32];
                sum += d * (float)sc[is + 4] * (float)q3 * xp[l + 64];
                sum += d * (float)sc[is + 6] * (float)q4 * xp[l + 96];
            }
        }
    }
    return sum;
}
