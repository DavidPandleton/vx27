#ifndef QL_QUANT_H
#define QL_QUANT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_k;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} block_q6_k;

typedef struct {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
} block_q8_k;

#if defined(__F16C__)
#include <immintrin.h>
static inline float fp16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static inline uint16_t f32_to_fp16(float f) { return (uint16_t)_cvtss_sh(f, 0); }
#else
static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t f32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t m = mant >> shift;
        if ((mant >> (shift - 1)) & 1) m++;
        return (uint16_t)(sign | m);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m++;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
}
#endif

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                    uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

void dequant_q4_k(const block_q4_k *b, float *out);
void dequant_q6_k(const block_q6_k *b, float *out);

/* Fused dot products: row is nblocks consecutive blocks, x has 256*nblocks
 * floats. Scalar reference implementation. */
float dot_q4_k_f32(const block_q4_k *b, const float *x, size_t nblocks);
float dot_q6_k_f32(const block_q6_k *b, const float *x, size_t nblocks);

#endif /* QL_QUANT_H */
