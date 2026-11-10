#include "vx27_rope.h"
#include "vx27_config.h"
#include <math.h>

void vx27_rope_cos_sin(int pos, float *cosv, float *sinv, int rope_dim,
                       float theta, float yarn_factor) {
    if (yarn_factor < 1.0f) yarn_factor = 1.0f;
    float ps = (float)pos / yarn_factor;
    int half = rope_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(theta, -2.0f * (float)i / (float)rope_dim);
        float angle = ps * freq;
        cosv[i] = cosf(angle);
        sinv[i] = sinf(angle);
    }
}

void vx27_rope_apply(float *v, int n_heads, int head_dim, int rope_dim,
                     const float *cosv, const float *sinv) {
    int half = rope_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *p = v + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float x0 = p[i], x1 = p[i + half];
            p[i] = x0 * cosv[i] - x1 * sinv[i];
            p[i + half] = x0 * sinv[i] + x1 * cosv[i];
        }
    }
}

void vx27_mrope_cos_sin_text(int pos, float *cosv, float *sinv) {
    vx27_rope_cos_sin(pos, cosv, sinv, VX27_ROPE_DIM, VX27_ROPE_THETA, 1.0f);
}
