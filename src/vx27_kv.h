#ifndef VX27_KV_H
#define VX27_KV_H
#include <stdint.h>

// KV-cache sizing for the 16 full-attention layers only.
// per token: 16 * 2 * 4 * 256 = 32768 elems.
uint64_t vx27_kv_elems(int ctx);
double vx27_kv_gb(int ctx, double bytes_per_elem); // BF16/FP32=2/4, FP8=1, Q4=0.5
int vx27_kv_fits(int ctx, double bytes_per_elem, double budget_gb);

#endif
