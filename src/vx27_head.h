#ifndef VX27_HEAD_H
#define VX27_HEAD_H
#include <stdint.h>

// lm_head 5120x248320 = 1.27B params (untied) + MTP 1 layer.
// Full matrix never materialized in FP32 in tests; we check sizing/sharding
// and numeric correctness on tiny (hidden=16,vocab=32) proxies.

uint64_t vx27_lmhead_params(void);          // 1271398400
double vx27_lmhead_gb(double bits_per_param);
int vx27_lmhead_rows_gpu(double bits, double budget_gb, uint64_t *rows_gpu);

// float matvec ref vs AVX2 (row-major W[rows*cols], x[cols] -> y[rows]).
void vx27_matvec_ref(const float *W, const float *x, float *y, int rows, int cols);
void vx27_matvec_avx2(const float *W, const float *x, float *y, int rows, int cols);

// MTP 1-layer drafter (simplified, documented):
//   h1 = rmsnorm(h + ffn_mtp(h)) ; draft = h1 @ W_lm
// where ffn_mtp is SwiGLU(hidden->ffn_tiny->hidden). Tested on tiny dims
// with independent ref (double precision) vs float forward.
void vx27_mtp_forward_ref(const float *h, const float *Wg, const float *Wu,
                          const float *Wd, const float *Wlm,
                          const float *norm_w, float *draft,
                          int hidden, int ffn_tiny, int vocab);
void vx27_mtp_forward(const float *h, const float *Wg, const float *Wu,
                      const float *Wd, const float *Wlm,
                      const float *norm_w, float *draft,
                      int hidden, int ffn_tiny, int vocab);

#endif
