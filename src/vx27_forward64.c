#include "vx27_forward64.h"
#include <stddef.h>

float vx27_forward64_proxy(const vx27_block_cfg *bc, threadpool *tp,
    const float *x0, float *x_final,
    const vx27_block_weights *w_full, const vx27_block_weights *w_lin,
    float *kcache_all, float *vcache_all, float *S_all) {
    int H = bc->hidden;
    float *x_cur = x_final;
    for (int i = 0; i < H; i++) x_cur[i] = x0[i];
    // scratch per block
    float *tmp = x_final; // in-place chain: use second buffer ping-pong
    static float buf[2][256];
    for (int i = 0; i < H && i < 256; i++) buf[0][i] = x0[i];
    float *ping = buf[0], *pong = buf[1];
    int nkvhd = bc->n_kv * bc->hd;
    int s_per_lin = bc->lin_qk * bc->lin_hd * bc->lin_hd;
    for (int L = 0; L < 64; L++) {
        int is_full = ((L + 1) % 4 == 0);
        if (is_full) {
            int fidx = L / 4; // 0..15
            float *kc = kcache_all + (size_t)fidx * 4 * nkvhd;
            float *vc = vcache_all + (size_t)fidx * 4 * nkvhd;
            vx27_full_block_forward(bc, tp, ping, kc, vc, 0, pong, w_full);
        } else {
            int lidx = L - L / 4 - (L % 4 == 3 ? 0 : 0); // count of prior linears
            // simpler: linear index = L - number of fulls so far
            int nfull_sofar = (L + 1) / 4;
            lidx = L - nfull_sofar;
            float *S = S_all + (size_t)lidx * s_per_lin;
            vx27_linear_block_forward(bc, tp, ping, S, pong, w_lin);
        }
        float *t = ping; ping = pong; pong = t;
    }
    for (int i = 0; i < H; i++) x_final[i] = ping[i];
    (void)tmp;
    double chk = 0; for (int i = 0; i < H; i++) chk += x_final[i];
    return (float)chk;
}
