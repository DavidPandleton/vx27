#include "quant_avx2.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Q8_K activation quantization                                        */
/* ------------------------------------------------------------------ */

void quantize_q8_k(const float *x, block_q8_k *y, int nblocks) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (int b = 0; b < nblocks; b++) {
        const float *xb = x + (size_t)b * 256;
        __m256 amaxv = _mm256_setzero_ps();
        for (int i = 0; i < 256; i += 8) {
            __m256 v = _mm256_andnot_ps(sign, _mm256_loadu_ps(xb + i));
            amaxv = _mm256_max_ps(amaxv, v);
        }
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(amaxv),
                               _mm256_extractf128_ps(amaxv, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        float amax = _mm_cvtss_f32(m4);

        block_q8_k *q = &y[b];
        if (amax == 0.0f) {
            q->d = 0.0f;
            memset(q->qs, 0, 256);
            memset(q->bsums, 0, sizeof(q->bsums));
        } else {
            float d = amax / 127.0f;
            float id = 1.0f / d;
            __m256 idv = _mm256_set1_ps(id);
            for (int i = 0; i < 256; i += 32) {
                __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(xb + i + 0), idv);
                __m256 v1 = _mm256_mul_ps(_mm256_loadu_ps(xb + i + 8), idv);
                __m256 v2 = _mm256_mul_ps(_mm256_loadu_ps(xb + i + 16), idv);
                __m256 v3 = _mm256_mul_ps(_mm256_loadu_ps(xb + i + 24), idv);
                __m256i i0 = _mm256_cvtps_epi32(
                    _mm256_round_ps(v0, _MM_FROUND_TO_NEAREST_INT |
                                            _MM_FROUND_NO_EXC));
                __m256i i1 = _mm256_cvtps_epi32(
                    _mm256_round_ps(v1, _MM_FROUND_TO_NEAREST_INT |
                                            _MM_FROUND_NO_EXC));
                __m256i i2 = _mm256_cvtps_epi32(
                    _mm256_round_ps(v2, _MM_FROUND_TO_NEAREST_INT |
                                            _MM_FROUND_NO_EXC));
                __m256i i3 = _mm256_cvtps_epi32(
                    _mm256_round_ps(v3, _MM_FROUND_TO_NEAREST_INT |
                                            _MM_FROUND_NO_EXC));
                __m256i p01 = _mm256_packs_epi32(i0, i1);
                __m256i p23 = _mm256_packs_epi32(i2, i3);
                p01 = _mm256_permute4x64_epi64(p01, 0xD8);
                p23 = _mm256_permute4x64_epi64(p23, 0xD8);
                __m128i lo01 = _mm256_castsi256_si128(p01);
                __m128i hi01 = _mm256_extracti128_si256(p01, 1);
                __m128i lo23 = _mm256_castsi256_si128(p23);
                __m128i hi23 = _mm256_extracti128_si256(p23, 1);
                _mm_storeu_si128((__m128i *)(q->qs + i),
                                 _mm_packs_epi16(lo01, hi01));
                _mm_storeu_si128((__m128i *)(q->qs + i + 16),
                                 _mm_packs_epi16(lo23, hi23));
            }
            q->d = d;
            for (int j = 0; j < 16; j++) {
                __m128i qb =
                    _mm_loadu_si128((const __m128i *)(q->qs + j * 16));
                __m128i p16 = _mm_maddubs_epi16(
                    _mm256_castsi256_si128(one8), qb);
                __m128i s32 = _mm_madd_epi16(
                    _mm256_castsi256_si128(one16), p16);
                s32 = _mm_hadd_epi32(s32, s32);
                s32 = _mm_hadd_epi32(s32, s32);
                q->bsums[j] = (int16_t)_mm_cvtsi128_si32(s32);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static inline int hsum_i32_8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

static inline float hsum_f32_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* ------------------------------------------------------------------ */
/* Q4_K x Q8_K                                                         */
/* ------------------------------------------------------------------ */

float dot_q4_k_q8_k_avx2(const block_q4_k *b, const block_q8_k *y,
                         size_t nblocks) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    float total = 0.0f;

    for (size_t i = 0; i < nblocks; i++) {
        const float d16 = fp16_to_f32(b[i].d);
        const float m16 = fp16_to_f32(b[i].dmin);
        const uint8_t *qs = b[i].qs;
        const int8_t *q8 = y[i].qs;
        int16_t sc16[16];
        uint8_t mn8[8];
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(j, b[i].scales, &sc, &mn);
            sc16[j] = (int16_t)sc;
            mn8[j] = mn;
        }
        __m256i sumi = _mm256_setzero_si256();
        float msum = 0.0f;

        for (int j = 0; j < 4; j++) {
            __m256i q4bits =
                _mm256_loadu_si256((const __m256i *)(qs + 32 * j));
            __m256i q4l = _mm256_and_si256(q4bits, m4);
            __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            __m256i q8l =
                _mm256_loadu_si256((const __m256i *)(q8 + 64 * j));
            __m256i q8h =
                _mm256_loadu_si256((const __m256i *)(q8 + 64 * j + 32));
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16l = _mm256_madd_epi16(_mm256_set1_epi16(sc16[2 * j]), p16l);
            p16h = _mm256_madd_epi16(_mm256_set1_epi16(sc16[2 * j + 1]), p16h);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            msum += (float)mn8[2 * j] *
                        (float)((int)y[i].bsums[4 * j] +
                                (int)y[i].bsums[4 * j + 1]) +
                    (float)mn8[2 * j + 1] *
                        (float)((int)y[i].bsums[4 * j + 2] +
                                (int)y[i].bsums[4 * j + 3]);
        }

        float f = d16 * (float)hsum_i32_8(sumi) - m16 * msum;
        total += y[i].d * f;
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Q6_K x Q8_K                                                         */
/* ------------------------------------------------------------------ */

/* For get_scale_shuffle(i), each pair of scales occupies 16 bytes:
 * scale 2i repeated 8 times, then scale 2i+1 repeated 8 times. */
static const uint8_t scale_shuffle_tab[128] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
    12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
    14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
};

static inline __m128i get_scale_shuffle(int i) {
    return _mm_loadu_si128((const __m128i *)(scale_shuffle_tab + 16 * i));
}

float dot_q6_k_q8_k_avx2(const block_q6_k *b, const block_q8_k *y,
                         size_t nblocks) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8((char)0xC0);
    __m256 acc = _mm256_setzero_ps();

    for (size_t i = 0; i < nblocks; i++) {
        const float d = y[i].d * fp16_to_f32(b[i].d);
        const uint8_t *ql = b[i].ql;
        const uint8_t *qh = b[i].qh;
        const int8_t *q8 = y[i].qs;

        __m256i q8sums = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        __m128i scales8 = _mm_loadu_si128((const __m128i *)b[i].scales);
        __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
        __m256i q8sclsub =
            _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5);

        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        for (int half = 0; half < 2; half++) {
            __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)ql);
            ql += 32;
            __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)ql);
            ql += 32;
            __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
            qh += 32;

            __m256i q4h0 = _mm256_slli_epi16(
                _mm256_and_si256(q4bitsH, m3), 4);
            __m256i q4h1 = _mm256_slli_epi16(
                _mm256_and_si256(q4bitsH, m12), 2);
            __m256i q4h2 = _mm256_and_si256(q4bitsH, m48);
            __m256i q4h3 = _mm256_srli_epi16(
                _mm256_and_si256(q4bitsH, m192), 2);

            __m256i q4_0 = _mm256_or_si256(
                _mm256_and_si256(q4bits1, m15), q4h0);
            __m256i q4_1 = _mm256_or_si256(
                _mm256_and_si256(q4bits2, m15), q4h1);
            __m256i q4_2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h2);
            __m256i q4_3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h3);

            __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;
            __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
            q8 += 32;

            __m256i p0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q4_3, q8_3);

            __m128i s0 = _mm_shuffle_epi8(scales8, get_scale_shuffle(is + 0));
            __m128i s1 = _mm_shuffle_epi8(scales8, get_scale_shuffle(is + 1));
            __m128i s2 = _mm_shuffle_epi8(scales8, get_scale_shuffle(is + 2));
            __m128i s3 = _mm_shuffle_epi8(scales8, get_scale_shuffle(is + 3));
            is += 4;

            p0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s0), p0);
            p1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s1), p1);
            p2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s2), p2);
            p3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s3), p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p2, p3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d),
                              _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_f32_8(acc);
}

/* ------------------------------------------------------------------ */
/* Batched Q4_K / Q6_K x Q8_K (prefill)                                */
/*                                                                    */
/* Each weight row is decoded once and multiplied with up to eight     */
/* Q8_K activations. Per-token accumulation order matches the single   */
/* dot kernels, so results are bit-identical.                          */
/* ------------------------------------------------------------------ */

/* Build mn[j] duplicated per 16-value bsum pair, as int16 lanes. */
static inline __m256i q4_mvec(const uint8_t *mn8) {
    __m128i b = _mm_loadl_epi64((const __m128i *)mn8);
    __m128i d = _mm_unpacklo_epi8(b, b);
    return _mm256_cvtepu8_epi16(d);
}

/* Integer dot of mn[j] with the two 16-value sums of each 32-value block.
 * Values are small enough that int32 accumulation is exact, matching the
 * scalar float expression bit for bit. */
static inline float q4_block_msum(const block_q8_k *y, __m256i mvec) {
    __m256i p = _mm256_madd_epi16(
        mvec, _mm256_loadu_si256((const __m256i *)y->bsums));
    return (float)hsum_i32_8(p);
}

void dot_q4_k_q8_k_x8(const block_q4_k *b, const block_q8_k *y,
                      size_t stride, size_t nblocks, float out[8]) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    float total[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t i = 0; i < nblocks; i++) {
        const float d16 = fp16_to_f32(b[i].d);
        const float m16 = fp16_to_f32(b[i].dmin);
        const uint8_t *qs = b[i].qs;
        int16_t sc16[16];
        uint8_t mn8[8];
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(j, b[i].scales, &sc, &mn);
            sc16[j] = (int16_t)sc;
            mn8[j] = mn;
        }

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        __m256i sumi3 = _mm256_setzero_si256();
        __m256i sumi4 = _mm256_setzero_si256();
        __m256i sumi5 = _mm256_setzero_si256();
        __m256i sumi6 = _mm256_setzero_si256();
        __m256i sumi7 = _mm256_setzero_si256();
        __m256i mvec = q4_mvec(mn8);
        float msum0 = q4_block_msum(y + (size_t)0 * stride + i, mvec);
        float msum1 = q4_block_msum(y + (size_t)1 * stride + i, mvec);
        float msum2 = q4_block_msum(y + (size_t)2 * stride + i, mvec);
        float msum3 = q4_block_msum(y + (size_t)3 * stride + i, mvec);
        float msum4 = q4_block_msum(y + (size_t)4 * stride + i, mvec);
        float msum5 = q4_block_msum(y + (size_t)5 * stride + i, mvec);
        float msum6 = q4_block_msum(y + (size_t)6 * stride + i, mvec);
        float msum7 = q4_block_msum(y + (size_t)7 * stride + i, mvec);

        for (int j = 0; j < 4; j++) {
            __m256i q4bits =
                _mm256_loadu_si256((const __m256i *)(qs + 32 * j));
            __m256i q4l = _mm256_and_si256(q4bits, m4);
            __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            __m256i s0 = _mm256_set1_epi16(sc16[2 * j]);
            __m256i s1 = _mm256_set1_epi16(sc16[2 * j + 1]);

#define Q4_ACC(T)                                                        \
    do {                                                                    \
        const block_q8_k *yt = y + (size_t)(T) * stride + i;                \
        __m256i q8l = _mm256_loadu_si256(                                   \
            (const __m256i *)(yt->qs + 64 * j));                            \
        __m256i q8h = _mm256_loadu_si256(                                   \
            (const __m256i *)(yt->qs + 64 * j + 32));                       \
        __m256i pl = _mm256_maddubs_epi16(q4l, q8l);                        \
        __m256i ph = _mm256_maddubs_epi16(q4h, q8h);                        \
        pl = _mm256_madd_epi16(s0, pl);                                     \
        ph = _mm256_madd_epi16(s1, ph);                                     \
        sumi##T = _mm256_add_epi32(sumi##T, _mm256_add_epi32(pl, ph));                  \
    } while (0)

            Q4_ACC(0);
            Q4_ACC(1);
            Q4_ACC(2);
            Q4_ACC(3);
            Q4_ACC(4);
            Q4_ACC(5);
            Q4_ACC(6);
            Q4_ACC(7);
#undef Q4_ACC
        }

        {
            const block_q8_k *yt;
#define Q4_FINAL(T)                                                         \
    do {                                                                    \
        yt = y + (size_t)(T) * stride + i;                                  \
        float f = d16 * (float)hsum_i32_8(sumi##T) - m16 * msum##T;         \
        total[T] += yt->d * f;                                              \
    } while (0)
            Q4_FINAL(0);
            Q4_FINAL(1);
            Q4_FINAL(2);
            Q4_FINAL(3);
            Q4_FINAL(4);
            Q4_FINAL(5);
            Q4_FINAL(6);
            Q4_FINAL(7);
#undef Q4_FINAL
        }
    }
    for (int t = 0; t < 8; t++) out[t] = total[t];
}

void dot_q6_k_q8_k_x8(const block_q6_k *b, const block_q8_k *y,
                      size_t stride, size_t nblocks, float out[8]) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8((char)0xC0);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    for (size_t i = 0; i < nblocks; i++) {
        const float dbase = fp16_to_f32(b[i].d);
        const uint8_t *ql = b[i].ql;
        const uint8_t *qh = b[i].qh;
        __m128i scales8 = _mm_loadu_si128((const __m128i *)b[i].scales);
        __m256i scales16 = _mm256_cvtepi8_epi16(scales8);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        __m256i sumi3 = _mm256_setzero_si256();
        __m256i sumi4 = _mm256_setzero_si256();
        __m256i sumi5 = _mm256_setzero_si256();
        __m256i sumi6 = _mm256_setzero_si256();
        __m256i sumi7 = _mm256_setzero_si256();
        __m256i q8sclsub0 = _mm256_setzero_si256();
        __m256i q8sclsub1 = _mm256_setzero_si256();
        __m256i q8sclsub2 = _mm256_setzero_si256();
        __m256i q8sclsub3 = _mm256_setzero_si256();
        __m256i q8sclsub4 = _mm256_setzero_si256();
        __m256i q8sclsub5 = _mm256_setzero_si256();
        __m256i q8sclsub6 = _mm256_setzero_si256();
        __m256i q8sclsub7 = _mm256_setzero_si256();
        __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
        __m256 d2 = _mm256_setzero_ps(), d3 = _mm256_setzero_ps();
        __m256 d4 = _mm256_setzero_ps(), d5 = _mm256_setzero_ps();
        __m256 d6 = _mm256_setzero_ps(), d7 = _mm256_setzero_ps();

#define Q6_PREP(T)                                                          \
    do {                                                                    \
        const block_q8_k *yt = y + (size_t)(T) * stride + i;                \
        __m256i q8sums = _mm256_loadu_si256(                                \
            (const __m256i *)yt->bsums);                                    \
        q8sclsub##T = _mm256_slli_epi32(                                    \
            _mm256_madd_epi16(q8sums, scales16), 5);                        \
        d##T = _mm256_set1_ps(yt->d * dbase);                               \
    } while (0)
        Q6_PREP(0); Q6_PREP(1); Q6_PREP(2); Q6_PREP(3);
        Q6_PREP(4); Q6_PREP(5); Q6_PREP(6); Q6_PREP(7);
#undef Q6_PREP

        int is = 0;
        for (int half = 0; half < 2; half++) {
            __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)ql);
            ql += 32;
            __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)ql);
            ql += 32;
            __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
            qh += 32;

            __m256i q4h0 = _mm256_slli_epi16(
                _mm256_and_si256(q4bitsH, m3), 4);
            __m256i q4h1 = _mm256_slli_epi16(
                _mm256_and_si256(q4bitsH, m12), 2);
            __m256i q4h2 = _mm256_and_si256(q4bitsH, m48);
            __m256i q4h3 = _mm256_srli_epi16(
                _mm256_and_si256(q4bitsH, m192), 2);

            __m256i q4v0 = _mm256_or_si256(
                _mm256_and_si256(q4bits1, m15), q4h0);
            __m256i q4v1 = _mm256_or_si256(
                _mm256_and_si256(q4bits2, m15), q4h1);
            __m256i q4v2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h2);
            __m256i q4v3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h3);

            __m128i s0 = _mm_shuffle_epi8(scales8, get_scale_shuffle(is + 0));
            __m128i s1 = _mm_shuffle_epi8(scales8, get_scale_shuffle(is + 1));
            __m128i s2 = _mm_shuffle_epi8(scales8, get_scale_shuffle(is + 2));
            __m128i s3 = _mm_shuffle_epi8(scales8, get_scale_shuffle(is + 3));
            is += 4;
            __m256i sv0 = _mm256_cvtepi8_epi16(s0);
            __m256i sv1 = _mm256_cvtepi8_epi16(s1);
            __m256i sv2 = _mm256_cvtepi8_epi16(s2);
            __m256i sv3 = _mm256_cvtepi8_epi16(s3);

            /* Spill the shared weight fragment so token unrolling keeps
             * all eight accumulators in registers. */
            __m256i q4s[4] = {q4v0, q4v1, q4v2, q4v3};
            __m256i scs[4] = {sv0, sv1, sv2, sv3};
            const int qoff = half * 128;

#define Q6_ACC(T)                                                           \
    do {                                                                    \
        const block_q8_k *yt = y + (size_t)(T) * stride + i;                \
        __m256i z0 = _mm256_maddubs_epi16(                                  \
            q4s[0], _mm256_loadu_si256(                                     \
                        (const __m256i *)(yt->qs + qoff + 0)));             \
        __m256i z1 = _mm256_maddubs_epi16(                                  \
            q4s[1], _mm256_loadu_si256(                                     \
                        (const __m256i *)(yt->qs + qoff + 32)));            \
        __m256i z2 = _mm256_maddubs_epi16(                                  \
            q4s[2], _mm256_loadu_si256(                                     \
                        (const __m256i *)(yt->qs + qoff + 64)));            \
        __m256i z3 = _mm256_maddubs_epi16(                                  \
            q4s[3], _mm256_loadu_si256(                                     \
                        (const __m256i *)(yt->qs + qoff + 96)));            \
        z0 = _mm256_madd_epi16(scs[0], z0);                                 \
        z1 = _mm256_madd_epi16(scs[1], z1);                                 \
        z2 = _mm256_madd_epi16(scs[2], z2);                                 \
        z3 = _mm256_madd_epi16(scs[3], z3);                                 \
        sumi##T = _mm256_add_epi32(sumi##T,                                 \
                                   _mm256_add_epi32(z0, z1));               \
        sumi##T = _mm256_add_epi32(sumi##T,                                 \
                                   _mm256_add_epi32(z2, z3));               \
    } while (0)
            Q6_ACC(0); Q6_ACC(1); Q6_ACC(2); Q6_ACC(3);
            Q6_ACC(4); Q6_ACC(5); Q6_ACC(6); Q6_ACC(7);
#undef Q6_ACC
        }

#define Q6_FINAL(T)                                                         \
    do {                                                                    \
        sumi##T = _mm256_sub_epi32(sumi##T, q8sclsub##T);                   \
        acc##T = _mm256_fmadd_ps(d##T, _mm256_cvtepi32_ps(sumi##T),         \
                                 acc##T);                                   \
    } while (0)
        Q6_FINAL(0); Q6_FINAL(1); Q6_FINAL(2); Q6_FINAL(3);
        Q6_FINAL(4); Q6_FINAL(5); Q6_FINAL(6); Q6_FINAL(7);
#undef Q6_FINAL
    }
    out[0] = hsum_f32_8(acc0);
    out[1] = hsum_f32_8(acc1);
    out[2] = hsum_f32_8(acc2);
    out[3] = hsum_f32_8(acc3);
    out[4] = hsum_f32_8(acc4);
    out[5] = hsum_f32_8(acc5);
    out[6] = hsum_f32_8(acc6);
    out[7] = hsum_f32_8(acc7);
}

