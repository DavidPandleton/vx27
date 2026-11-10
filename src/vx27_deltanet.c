#include "vx27_deltanet.h"
#include "vx27_config.h"
#include <math.h>

int vx27_deltanet_state_per_layer(void) {
    return VX27_LIN_QK_HEADS * VX27_LIN_HEAD_DIM * VX27_LIN_HEAD_DIM;
}
int vx27_deltanet_state_total(void) {
    return vx27_deltanet_state_per_layer() * VX27_N_LINEAR;
}
double vx27_deltanet_state_mb(int bytes_per_elem) {
    return (double)vx27_deltanet_state_total() * bytes_per_elem / 1e6;
}

void vx27_deltanet_update_ref(float *S, const float *k, const float *v,
                              float decay, int hd) {
    for (int i = 0; i < hd; i++)
        for (int j = 0; j < hd; j++)
            S[i * hd + j] = decay * S[i * hd + j] + k[i] * v[j];
}
void vx27_deltanet_retrieve_ref(const float *S, const float *q, float *o, int hd) {
    for (int j = 0; j < hd; j++) {
        float s = 0.0f;
        for (int i = 0; i < hd; i++) s += S[i * hd + j] * q[i];
        o[j] = s;
    }
}

// Tiled 16x16 to mirror future AVX2 blocking; math identical, loop order differs.
void vx27_deltanet_update_tiled(float *S, const float *k, const float *v,
                                float decay, int hd) {
    const int T = 16;
    for (int i0 = 0; i0 < hd; i0 += T)
        for (int j0 = 0; j0 < hd; j0 += T)
            for (int i = i0; i < i0 + T && i < hd; i++)
                for (int j = j0; j < j0 + T && j < hd; j++)
                    S[i * hd + j] = decay * S[i * hd + j] + k[i] * v[j];
}
void vx27_deltanet_retrieve_tiled(const float *S, const float *q, float *o, int hd) {
    const int T = 16;
    for (int j = 0; j < hd; j++) o[j] = 0.0f;
    for (int i0 = 0; i0 < hd; i0 += T)
        for (int j = 0; j < hd; j++) {
            float s = 0.0f;
            int i1 = i0 + T < hd ? i0 + T : hd;
            for (int i = i0; i < i1; i++) s += S[i * hd + j] * q[i];
            o[j] += s;
        }
}

void vx27_deltanet_l2norm(const float *x, float *y, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float s = 1.0f / sqrtf((float)ss + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = x[i] * s;
}

static double vx27_sp(double x) {
    // softplus stabil: referensi memakai F.softplus (stabil).
    if (x > 20.0) return x;
    if (x < -20.0) return exp(x);
    return log(1.0 + exp(x));
}

void vx27_deltanet_gate_params(double a, double A_log, double b, double dt_bias,
                               double *beta, double *g_log) {
    if (beta) *beta = 1.0 / (1.0 + exp(-b));
    if (g_log) *g_log = -exp(A_log) * vx27_sp(a + dt_bias);
}

void vx27_deltanet_step_exact(float *S, const float *q, const float *k, const float *v,
                              double beta, double g_log, float *o, int hd) {
    // Transkripsi torch_recurrent_gated_delta_rule (satu token, satu V-head):
    // q,k l2norm -> q/=sqrt(hd); S*=decay; kv=S^T k; d=(v-kv)*beta;
    // S+=k⊗d; o=S^T q. Buffer stack untuk hd<=256.
    float qn[256], kn[256];
    vx27_deltanet_l2norm(q, qn, hd);
    vx27_deltanet_l2norm(k, kn, hd);
    float qs = 1.0f / sqrtf((float)hd);
    for (int i = 0; i < hd; i++) qn[i] *= qs;
    double decay = exp(g_log);
    for (int i = 0; i < hd; i++) { float *row = S + (size_t)i * hd; for (int j = 0; j < hd; j++) row[j] = (float)((double)row[j] * decay); }
    double kv[256];
    for (int j = 0; j < hd; j++) {
        double s = 0.0;
        for (int i = 0; i < hd; i++) s += (double)S[i * hd + j] * kn[i];
        kv[j] = s;
    }
    for (int i = 0; i < hd; i++) {
        double ki = kn[i];
        float *row = S + (size_t)i * hd;
        for (int j = 0; j < hd; j++) {
            double d = ((double)v[j] - kv[j]) * beta;
            row[j] = (float)((double)row[j] + ki * d);
        }
    }
    for (int j = 0; j < hd; j++) {
        double s = 0.0;
        for (int i = 0; i < hd; i++) s += (double)S[i * hd + j] * qn[i];
        o[j] = (float)s;
    }
}

static float vx27_silu_f(float v) { return v / (1.0f + expf(-v)); }

void vx27_conv1d_step(float *state /*[C*3]*/, const float *x /*[C]*/,
                      const float *w /*[C*4]*/, float *y /*[C]*/,
                      int C, int apply_silu) {
    // j=0 sampel TERLAMA (konsisten F.conv1d padding kiri K-1:
    // token pertama memakai tap w[3]).
    for (int c = 0; c < C; c++) {
        float s0 = state[(size_t)c * 3 + 0];
        float s1 = state[(size_t)c * 3 + 1];
        float s2 = state[(size_t)c * 3 + 2];
        float acc = w[(size_t)c * 4 + 0] * s0 + w[(size_t)c * 4 + 1] * s1
                  + w[(size_t)c * 4 + 2] * s2 + w[(size_t)c * 4 + 3] * x[c];
        y[c] = apply_silu ? vx27_silu_f(acc) : acc;
        state[(size_t)c * 3 + 0] = s1;
        state[(size_t)c * 3 + 1] = s2;
        state[(size_t)c * 3 + 2] = x[c];
    }
}
