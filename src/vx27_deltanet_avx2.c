#include "vx27_deltanet.h"
#include <immintrin.h>

// Update: row i: S_row = decay*S_row + k[i]*v[0..hd)
// AVX2: 8 floats per step, broadcast k[i].
void vx27_deltanet_update_avx2(float *S, const float *k, const float *v,
                               float decay, int hd) {
    __m256 d = _mm256_set1_ps(decay);
    for (int i = 0; i < hd; i++) {
        __m256 ki = _mm256_set1_ps(k[i]);
        float *row = S + (size_t)i * hd;
        for (int j = 0; j < hd; j += 8) {
            __m256 s = _mm256_loadu_ps(row + j);
            __m256 vv = _mm256_loadu_ps(v + j);
            // s = s*decay + ki*vv  (fmadd keeps one rounding)
            __m256 kv = _mm256_mul_ps(ki, vv);
#if defined(__FMA__)
            s = _mm256_fmadd_ps(s, d, kv);
#else
            s = _mm256_add_ps(_mm256_mul_ps(s, d), kv);
#endif
            _mm256_storeu_ps(row + j, s);
        }
    }
}

// Retrieve: o[j] = sum_i S[i*hd+j]*q[i]
// AVX2: 8 columns at a time, broadcast q[i].
void vx27_deltanet_retrieve_avx2(const float *S, const float *q, float *o, int hd) {
    for (int j = 0; j < hd; j += 8) _mm256_storeu_ps(o + j, _mm256_setzero_ps());
    for (int i = 0; i < hd; i++) {
        __m256 qi = _mm256_set1_ps(q[i]);
        const float *row = S + (size_t)i * hd;
        for (int j = 0; j < hd; j += 8) {
            __m256 acc = _mm256_loadu_ps(o + j);
            __m256 s = _mm256_loadu_ps(row + j);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(s, qi, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(s, qi));
#endif
            _mm256_storeu_ps(o + j, acc);
        }
    }
}
