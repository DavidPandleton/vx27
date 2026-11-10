#ifndef VX27_BLOCK_H
#define VX27_BLOCK_H
#include "threadpool.h"

// Proxy block dims for testing (scaled-down Qwen3.8).
// Production: hidden=5120, ffn=17408, n_q=24, n_kv=4, hd=256, rope=64.
// Test: hidden=32, ffn=64, n_q=4, n_kv=2, hd=8, rope=8, lin_qk=2, lin_v=2, lin_hd=8.
typedef struct {
    int hidden, ffn;
    int n_q, n_kv, hd, rope_dim;
    int lin_qk, lin_v, lin_hd;
    float rope_theta, yarn_factor;
} vx27_block_cfg;

// Weights (row-major). Full block: Wq[H*n_q*hd], Wk, Wv, Wo[n_q*hd*H], Wg/Wu[H*ffn], Wd[ffn*H].
// Linear block reuses same struct with lin dims for q/k/v/o.
typedef struct {
    const float *attn_norm_w, *ffn_norm_w;
    const float *Wq, *Wk, *Wv, *Wo;
    const float *Wg, *Wu, *Wd;
    const float *qk_norm_w; // per-head-dim norm (hd), may be NULL
} vx27_block_weights;

// Full block forward, single token, KV cache holds n_past+1 positions.
// kcache/vcache: [(n_past+1)*n_kv*hd]. S is scratch. tp==NULL => single thread.
void vx27_full_block_forward(const vx27_block_cfg *bc, threadpool *tp,
    const float *x, float *kcache, float *vcache, int n_past,
    float *state_out, const vx27_block_weights *w);
// Linear block forward, single token, recurrent state S[lin_qk*lin_hd*lin_hd].
void vx27_linear_block_forward(const vx27_block_cfg *bc, threadpool *tp,
    const float *x, float *S, float *out, const vx27_block_weights *w);

// Real-dim estimate: FLOPs and bytes per token for one block (for bench extrapolation).
void vx27_block_estimate(int is_full, double *flops, double *weight_bytes_q4);

#endif
