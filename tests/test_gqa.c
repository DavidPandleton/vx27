#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../src/vx27_gqa.h"
#include "../src/vx27_config.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    CHECK(vx27_gqa_kv_for_q(0) == 0, "q0->kv0");
    CHECK(vx27_gqa_kv_for_q(5) == 0, "q5->kv0");
    CHECK(vx27_gqa_kv_for_q(6) == 1, "q6->kv1");
    CHECK(vx27_gqa_kv_for_q(11) == 1, "q11->kv1");
    CHECK(vx27_gqa_kv_for_q(12) == 2, "q12->kv2");
    CHECK(vx27_gqa_kv_for_q(23) == 3, "q23->kv3");

    float x[2] = {3.0f, 4.0f}, w[2] = {1.0f, 1.0f}, y[2];
    vx27_rmsnorm(x, w, y, 2, 0.0f);
    printf("  rmsnorm [3,4] -> [%.4f,%.4f]\n", y[0], y[1]);
    CHECK(fabsf(y[0] - 0.8485f) < 1e-3f && fabsf(y[1] - 1.1314f) < 1e-3f, "rmsnorm values");

    // 1-pos attention: out must equal that pos value per GQA group
    static float q[24*256], kc[1*4*256], vc[1*4*256], out[24*256];
    for (int i = 0; i < 24*256; i++) q[i] = (float)(i % 7) * 0.01f;
    for (int i = 0; i < 4*256; i++) { kc[i] = (float)(i % 5) * 0.02f; vc[i] = (float)(i % 11) * 0.03f + (i / 256); }
    vx27_attn_1q(q, kc, vc, 1, out);
    int ok = 1;
    for (int h = 0; h < 24 && ok; h++) {
        int kvh = h / 6;
        const float *v = vc + (size_t)kvh * 256;
        float *o = out + (size_t)h * 256;
        for (int i = 0; i < 256; i++) if (fabsf(o[i] - v[i]) > 1e-4f) { ok = 0; break; }
    }
    CHECK(ok, "1-pos attn out==value per group");

    // 2-pos identical keys: out == mean of values
    static float kc2[2*4*256], vc2[2*4*256];
    for (int p = 0; p < 2; p++) { memcpy(kc2 + p*4*256, kc, sizeof(kc)); memcpy(vc2 + p*4*256, vc, sizeof(vc)); }
    // make values differ per pos to check average
    for (int i = 0; i < 4*256; i++) vc2[4*256 + i] = vc[i] + 2.0f;
    vx27_attn_1q(q, kc2, vc2, 2, out);
    ok = 1;
    for (int h = 0; h < 24 && ok; h++) {
        int kvh = h / 6;
        float *o = out + (size_t)h * 256;
        for (int i = 0; i < 256; i++) {
            float want = (vc[(size_t)kvh*256+i] + vc2[(size_t)(4*256)+(size_t)kvh*256+i]) * 0.5f;
            if (fabsf(o[i] - want) > 1e-3f) { ok = 0; break; }
        }
    }
    CHECK(ok, "2-pos identical keys -> mean values");
    if (fails) { printf("test_gqa: %d FAIL\n", fails); return 1; }
    printf("test_gqa: PASS\n");
    return 0;
}
