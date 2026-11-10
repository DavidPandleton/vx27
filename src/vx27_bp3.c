#include "vx27_bp3.h"
#include "quant.h"
#include <immintrin.h>
#include <math.h>

void vx27_bp3_pack(const float *w32, vx27_bp3_grp *g) {
    float amax = 0; for (int i = 0; i < 32; i++) { float a = fabsf(w32[i]); if (a > amax) amax = a; }
    float scale = amax > 0 ? amax / 3.5f : 1.0f;
    g->scale = scale;
    uint32_t p0 = 0, p1 = 0, p2 = 0;
    for (int i = 0; i < 32; i++) {
        int q = (int)roundf(w32[i] / scale);
        if (q > 3) q = 3;
        if (q < -4) q = -4;
        uint32_t u = (uint32_t)(q & 7);
        if (u & 1) p0 |= (1u << i);
        if (u & 2) p1 |= (1u << i);
        if (u & 4) p2 |= (1u << i);
    }
    g->p0 = p0; g->p1 = p1; g->p2 = p2;
}

void vx27_bp3_unpack_ref(const vx27_bp3_grp *g, float *w32) {
    for (int i = 0; i < 32; i++) {
        uint32_t u = ((g->p0 >> i) & 1) | (((g->p1 >> i) & 1) << 1) | (((g->p2 >> i) & 1) << 2);
        int q = (u >= 4) ? (int)u - 8 : (int)u;
        w32[i] = q * g->scale;
    }
}

float vx27_bp3_dot_ref(const vx27_bp3_grp *G, const float *x, int ngroups) {
    double s = 0; float w[32];
    for (int g = 0; g < ngroups; g++) {
        vx27_bp3_unpack_ref(&G[g], w);
        for (int i = 0; i < 32; i++) s += (double)w[i] * x[g*32+i];
    }
    return (float)s;
}

float vx27_bp3_dot_avx2(const vx27_bp3_grp *G, const float *x, int ngroups) {
    // AVX2: unpack 8 at a time via plane gathers (scalar bit extract, vector mul).
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < ngroups; g++) {
        float sc = G[g].scale;
        __m256 scv = _mm256_set1_ps(sc);
        for (int i = 0; i < 32; i += 8) {
            int qi[8];
            for (int k = 0; k < 8; k++) {
                uint32_t u = ((G[g].p0 >> (i+k)) & 1) | (((G[g].p1 >> (i+k)) & 1) << 1) | (((G[g].p2 >> (i+k)) & 1) << 2);
                qi[k] = (u >= 4) ? (int)u - 8 : (int)u;
            }
            __m256 qv = _mm256_set_ps((float)qi[7],(float)qi[6],(float)qi[5],(float)qi[4],
                                      (float)qi[3],(float)qi[2],(float)qi[1],(float)qi[0]);
            __m256 xv = _mm256_loadu_ps(x + g*32 + i);
            // q*scale*x
            __m256 wv = _mm256_mul_ps(qv, scv);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(wv, xv, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
#endif
        }
    }
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}

void vx27_bp3_128_pack(const float *w128, vx27_bp3_128 *g) {
    float amax = 0; for (int i = 0; i < 128; i++) { float a = fabsf(w128[i]); if (a > amax) amax = a; }
    float sc = amax > 0 ? amax / 3.5f : 1.0f;
    g->scale_h = f32_to_fp16(sc);
    for (int k = 0; k < 12; k++) g->p[k] = 0;
    for (int i = 0; i < 128; i++) {
        int q = (int)roundf(w128[i] / sc);
        if (q > 3) q = 3;
        if (q < -4) q = -4;
        uint32_t u = (uint32_t)(q & 7);
        int plane = i / 32, bit = i % 32;
        // plane layout: p[plane*4 + word]? Use p[0..3]=bit0 chunks, p[4..7]=bit1, p[8..11]=bit2
        int widx = (plane < 0) ? 0 : plane; (void)widx;
        // simpler: bit b of weight i lives in p[b*4 + i/32] at bit i%32
        if (u & 1) g->p[0*4 + i/32] |= (1u << bit);
        if (u & 2) g->p[1*4 + i/32] |= (1u << bit);
        if (u & 4) g->p[2*4 + i/32] |= (1u << bit);
    }
}

static inline int bp128_q(const vx27_bp3_128 *g, int i) {
    uint32_t u = ((g->p[0*4 + i/32] >> (i%32)) & 1) |
                 (((g->p[1*4 + i/32] >> (i%32)) & 1) << 1) |
                 (((g->p[2*4 + i/32] >> (i%32)) & 1) << 2);
    return (u >= 4) ? (int)u - 8 : (int)u;
}

float vx27_bp3_128_dot_ref(const vx27_bp3_128 *G, const float *x, int ng128) {
    double s = 0;
    for (int g = 0; g < ng128; g++) {
        float sc = fp16_to_f32(G[g].scale_h);
        for (int i = 0; i < 128; i++) s += (double)(bp128_q(&G[g], i) * sc) * x[g*128+i];
    }
    return (float)s;
}

float vx27_bp3_128_dot_avx2(const vx27_bp3_128 *G, const float *x, int ng128) {
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < ng128; g++) {
        __m256 scv = _mm256_set1_ps(fp16_to_f32(G[g].scale_h));
        for (int i = 0; i < 128; i += 8) {
            int qi[8]; for (int k = 0; k < 8; k++) qi[k] = bp128_q(&G[g], i+k);
            __m256 qv = _mm256_set_ps((float)qi[7],(float)qi[6],(float)qi[5],(float)qi[4],
                                      (float)qi[3],(float)qi[2],(float)qi[1],(float)qi[0]);
            __m256 xv = _mm256_loadu_ps(x + g*128 + i);
            __m256 wv = _mm256_mul_ps(qv, scv);
#if defined(__FMA__)
            acc = _mm256_fmadd_ps(wv, xv, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
#endif
        }
    }
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}
