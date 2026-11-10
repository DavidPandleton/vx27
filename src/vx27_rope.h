#ifndef VX27_ROPE_H
#define VX27_ROPE_H

// Partial RoPE + Mrope + YaRN (simplified, text-only) for Qwen3.8-27B full layers.
// head_dim=256, rope_dim=64 (partial_rotary_factor 0.25), theta=10M.
// Mrope sections [11,11,10] sum to 32 pairs = 64 dims; for text all sections
// share the same position, so it reduces to standard RoPE. Documented simplification.
// YaRN: factor 4.0 extends 262K->1M; we use linear position interpolation
// angle = (pos/factor) * theta^(-2i/rope_dim). factor=1 inside native ctx.

void vx27_rope_cos_sin(int pos, float *cosv, float *sinv, int rope_dim,
                       float theta, float yarn_factor);
// Apply to first rope_dim of each head (NeoX half-split like ql), rest passthrough.
void vx27_rope_apply(float *v, int n_heads, int head_dim, int rope_dim,
                     const float *cosv, const float *sinv);
// Mrope text helper: same pos for all sections.
void vx27_mrope_cos_sin_text(int pos, float *cosv, float *sinv);

#endif
