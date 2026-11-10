#ifndef VX27_PMAT_H
#define VX27_PMAT_H
#include "threadpool.h"
#include <stdint.h>

// Packed-weight matvec: W row-major [rows x cols], x float[cols] -> y float[rows].
// dtype: 0=F32, 2=Q8_K (block 256: float d + int8[256] + bsums, 292B),
//        4=BP3_128 (3x u32 planes per 32 + fp16 scale => 50B/128).
// Bobot tetap ter-packing (mmap langsung); aktivasi float. Tanpa ekspansi FP32.
//
// Varian in-register (produksi) vs skalar (referensi test); diuji identik.
void vx27_pmat(threadpool *tp, const void *W, uint32_t dtype,
               const float *x, float *y, int rows, int cols);
// Batched matvec: B vektor sekaligus, SATU pass bobot (hemat read memori).
// xb[b*cols+c] -> yb[b*rows+r]. CPU-only. 1<=B<=8. Hasil bit-identik dgn
// Bx vx27_pmat berurutan (urutan operasi per vektor dipertahankan).
// Dipakai batch-forward teacher-forcing (prefill + verify MTP): bobot 11GB
// dibaca 1x utk B token, bukan Bx.
void vx27_pmat_batch(threadpool *tp, const void *W, uint32_t dtype,
               const float *xb, float *yb, int rows, int cols, int B);
float vx27_pmat_bp3_scalar(const void *row, const float *x, int cols);
float vx27_pmat_bp3_reg(const void *row, const float *x, int cols);

#endif
