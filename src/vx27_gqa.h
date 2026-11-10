#ifndef VX27_GQA_H
#define VX27_GQA_H

// GQA 24Q/4KV hd256 + QK RMSNorm for Qwen3.8-27B full layers.
// kv_of_q(h) = h * 4 / 24 = h / 6 (same integer map as ql model.c).
// rmsnorm: y = x / sqrt(mean(x^2)+eps) * w.

int vx27_gqa_kv_for_q(int q_head);
void vx27_rmsnorm(const float *x, const float *w, float *y, int n, float eps);
// Single-query attention ref for one layer (float, no cache quant):
// q[24*256], kcache/vcache [pos * 4*256], out[24*256]. scale=1/sqrt(256).
void vx27_attn_1q(const float *q, const float *kcache, const float *vcache, int n_pos,
                  float *out);

#endif
