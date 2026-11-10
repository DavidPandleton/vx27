#ifndef VX27_FORWARD64_H
#define VX27_FORWARD64_H
#include "vx27_block.h"

// 64-layer end-to-end proxy: 16x [3x linear + 1x full] with tiny dims.
// Uses caller-allocated weights/state; runs single token through all 64.
// Returns checksum (sum of final hidden) for determinism testing.
float vx27_forward64_proxy(const vx27_block_cfg *bc, threadpool *tp,
    const float *x0, float *x_final,
    const vx27_block_weights *w_full, const vx27_block_weights *w_lin,
    float *kcache_all, float *vcache_all, float *S_all);

#endif
