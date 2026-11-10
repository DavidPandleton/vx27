#ifndef VX27_FUSED_H
#define VX27_FUSED_H
#include "threadpool.h"

// Forward 1 layer Gated DeltaNet fused ala Qwen3.8 (layout asli safetensors).
// Aproksimasi terdokumentasi mengikuti formulasi FLA GatedDeltaNet:
//   qkv = proj(x) -> split q/k/v; conv1d depthwise k=4 + SiLU (state nol utk 1 token);
//   beta = sigmoid(b + dt_bias); alpha = exp(-exp(A_log + a));
//   per grup g (16 grup QK, tiap grup 3 V-head):
//     Sk = S^T k; S = alpha*S + beta*outer(v - alpha*Sk, k); o = S^T q
//   o = rmsnorm(o) * silu(z); out = o @ Wo + x; lalu FFN SwiGLU + residual.
// Single-token; conv/state di-pass caller agar bisa multi-token nanti.

typedef struct {
    int H;            // hidden (5120)
    int nqk, nv;      // grup QK (16), V-head (48)
    int hd;           // head dim (128)
    int F;            // ffn (17408)
    const float *attn_norm_w, *post_norm_w;
    const float *Wqkv;   // [(nqk+ nqk + nv)*hd, H] row-major, urutan q|k|v
    const float *Wz;     // [nv*hd, H]
    const float *Wa, *Wb;// [nv, H]  (48 grup V)
    const float *Alog, *dt_bias; // [nv]
    const float *conv;   // [(nqk+nqk+nv)*hd, 4] depthwise weights
    const float *qknorm; // [hd] or NULL
    const float *Wo;     // [H, nv*hd]
    const float *Wg, *Wu;// [F, H]
    const float *Wd;     // [H, F]
} vx27_fused_w;

typedef struct {
    float *S;     // [nqk*hd*hd]
    float *conv_state; // [(nqk+nqk+nv)*hd*3]
} vx27_fused_state;

void vx27_fused_delta_forward(const vx27_fused_w *w, vx27_fused_state *st,
                              const float *x, float *out, threadpool *tp);

#endif
