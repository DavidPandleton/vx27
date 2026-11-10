#ifndef QL_QUANT_AVX2_H
#define QL_QUANT_AVX2_H

#include "quant.h"

#include <stddef.h>

/* Quantize nblocks*256 floats into Q8_K blocks (single scale per 256). */
void quantize_q8_k(const float *x, block_q8_k *y, int nblocks);

/* Integer dot products. Inputs must be 32-byte aligned or unaligned-safe. */
float dot_q4_k_q8_k_avx2(const block_q4_k *b, const block_q8_k *y,
                         size_t nblocks);
float dot_q6_k_q8_k_avx2(const block_q6_k *b, const block_q8_k *y,
                         size_t nblocks);

/* Batched prefill: one weight row against eight Q8_K activations.
 * y[t] is at y + t*stride. Results match eight single-dot calls. */
void dot_q4_k_q8_k_x8(const block_q4_k *b, const block_q8_k *y,
                      size_t stride, size_t nblocks, float out[8]);
void dot_q6_k_q8_k_x8(const block_q6_k *b, const block_q8_k *y,
                      size_t stride, size_t nblocks, float out[8]);
/* Hand-written AVX2 kernel (src/quant_avx2.asm). */
void dot_q4_k_q8_k_x8_asm(const block_q4_k *b, const block_q8_k *y,
                          size_t stride, size_t nblocks, float out[8]);
void dot_q6_k_q8_k_x8_asm(const block_q6_k *b, const block_q8_k *y,
                          size_t stride, size_t nblocks, float out[8]);

#endif /* QL_QUANT_AVX2_H */
