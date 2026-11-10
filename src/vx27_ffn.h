#ifndef VX27_FFN_H
#define VX27_FFN_H

// SwiGLU FFN for Qwen3.8-27B: hidden=5120, ffn=17408.
// hidden[i] = silu(gate[i]) * up[i], silu(x)=x/(1+exp(-x)).
// out = hidden @ W_down. Matvec itself reuses quant kernels; here we test
// the activation (numerically sensitive) ref vs AVX2.

void vx27_swiglu_forward_ref(const float *gate_in, const float *up,
                             float *hidden, int ffn);
void vx27_swiglu_forward_avx2(const float *gate_in, const float *up,
                              float *hidden, int ffn);

#endif
