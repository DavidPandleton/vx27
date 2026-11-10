#ifndef VX27_DELTANET_H
#define VX27_DELTANET_H
#include <stddef.h>

// Gated DeltaNet recurrent state sizing for Qwen3.8-27B.
// Per linear layer: QK_HEADS=16 groups, each with HD=128.
// State per group is HD x HD; V has 3x heads per group (48/16) sharing the mixer.
// Reference: state_elems_per_layer = QK_HEADS * HD * HD.

int vx27_deltanet_state_per_layer(void);   // 262144
int vx27_deltanet_state_total(void);       // *48
double vx27_deltanet_state_mb(int bytes_per_elem);

// Scalar reference update for ONE head (HD x HD state, row-major):
//   S = decay * S + outer(k, v)
//   o = S^T q   (o[j] = sum_i S[i*HD+j] * q[i])
void vx27_deltanet_update_ref(float *S, const float *k, const float *v,
                              float decay, int hd);
void vx27_deltanet_retrieve_ref(const float *S, const float *q, float *o, int hd);

// Tiled variant (must match reference within 1e-5).
void vx27_deltanet_update_tiled(float *S, const float *k, const float *v,
                                float decay, int hd);
void vx27_deltanet_retrieve_tiled(const float *S, const float *q, float *o, int hd);

// AVX2 kernels (hd must be multiple of 8; 128 in production).
// Same math as ref; FMA ordering may differ by ~1e-6.
void vx27_deltanet_update_avx2(float *S, const float *k, const float *v,
                               float decay, int hd);
void vx27_deltanet_retrieve_avx2(const float *S, const float *q, float *o, int hd);

// L2 norm konvensi FLA: y = x * rsqrt(sum(x^2) + eps) (TANPA /n,
// berbeda dengan rmsn). Dipakai untuk q/k di jalur DeltaNet referensi.
void vx27_deltanet_l2norm(const float *x, float *y, int n);

// Satu langkah exact gated delta rule, transkripsi langsung referensi
// torch_recurrent_gated_delta_rule (use_qk_l2norm_in_kernel=True):
//   q,k dil2norm + q /= sqrt(hd); beta in (0,1); g = log-decay (<=0).
//   decay = exp(g); kv = S^T k; d = (v-kv)*beta;
//   S = S*decay + k⊗d; o = S^T q.
// S row-major [hd*hd], diupdate in-place. Semua float; bandingkan dgn
// oracle double di test_deltanet_exact (toleransi <1e-4).
void vx27_deltanet_step_exact(float *S, const float *q, const float *k, const float *v,
                              double beta, double g_log, float *o, int hd);

// Parameter gerbang referensi (skalar, presisi ganda untuk uji):
//   beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
// (Catatan: dt_bias TIDAK masuk beta; softplus stabil untuk |x| besar.)
void vx27_deltanet_gate_params(double a, double A_log, double b, double dt_bias,
                               double *beta, double *g_log);

// Depthwise causal conv1d k=4 + state carryover (referensi causal_conv1d_update,
// bias=None, aktivasi SiLU). w row-major [C*4], tap j = w[c*4+j] dengan
// j=0 sampel TERLAMA (konsisten F.conv1d padding kiri K-1: token pertama
// memakai tap w[3]). state = 3 input terakhir per channel, zero-init.
// out[c] = silu(sum_j w[c*4+j]*win[j]) bila apply_silu != 0.
void vx27_conv1d_step(float *state /*[C*3] inout*/, const float *x /*[C]*/,
                      const float *w /*[C*4]*/, float *y /*[C]*/,
                      int C, int apply_silu);

#endif
