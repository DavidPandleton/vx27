#include "vx27_ffn.h"
#include <immintrin.h>
#include <math.h>

static inline float silu1(float x) {
    return x / (1.0f + expf(-x));
}

void vx27_swiglu_forward_ref(const float *gate_in, const float *up,
                             float *hidden, int ffn) {
    for (int i = 0; i < ffn; i++)
        hidden[i] = silu1(gate_in[i]) * up[i];
}

// AVX2: silu scalar per element (exp has no AVX2), multiply 8-wide.
void vx27_swiglu_forward_avx2(const float *gate_in, const float *up,
                              float *hidden, int ffn) {
    int i = 0;
    for (; i + 8 <= ffn; i += 8) {
        float sg[8];
        for (int k = 0; k < 8; k++) sg[k] = silu1(gate_in[i + k]);
        __m256 g = _mm256_loadu_ps(sg);
        __m256 u = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(hidden + i, _mm256_mul_ps(g, u));
    }
    for (; i < ffn; i++) hidden[i] = silu1(gate_in[i]) * up[i];
}
