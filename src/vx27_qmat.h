#ifndef VX27_QMAT_H
#define VX27_QMAT_H
#include "quant.h"
#include <stddef.h>

// Quantized matvec row: W is nblocks Q4_K, x is float[256*nblocks].
// Ref: dequant row to float then dot. Prod: Q8 quant + AVX2 dot (ql kernel).
float vx27_qmat_q4_ref(const block_q4_k *W, const float *x, size_t nblocks);
float vx27_qmat_q4(const block_q4_k *W, const float *x, size_t nblocks,
                   block_q8_k *q8tmp);

#endif
