#include "vx27_qmat.h"
#include "quant_avx2.h"
#include <stdlib.h>

float vx27_qmat_q4_ref(const block_q4_k *W, const float *x, size_t nblocks) {
    // dequant on the fly per block then dot (uses scalar ref dot)
    double s = 0;
    float *tmp = (float*)malloc(256 * sizeof(float));
    for (size_t b = 0; b < nblocks; b++) {
        dequant_q4_k(&W[b], tmp);
        for (int i = 0; i < 256; i++) s += (double)tmp[i] * x[b*256+i];
    }
    free(tmp);
    return (float)s;
}

float vx27_qmat_q4(const block_q4_k *W, const float *x, size_t nblocks,
                   block_q8_k *q8tmp) {
    quantize_q8_k(x, q8tmp, (int)nblocks);
    return dot_q4_k_q8_k_avx2(W, q8tmp, nblocks);
}
