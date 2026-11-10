#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../src/vx27_rope.h"
#include "../src/vx27_config.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    float cosv[64], sinv[64];
    vx27_rope_cos_sin(0, cosv, sinv, 64, 10000000.0f, 1.0f);
    int ok = 1;
    for (int i = 0; i < 32; i++) if (fabsf(cosv[i] - 1.0f) > 1e-6f || fabsf(sinv[i]) > 1e-6f) ok = 0;
    CHECK(ok, "pos0 cos=1 sin=0");

    // determinism + yarn linear: pos32/factor4 == pos8/factor1
    float c1[64], s1[64], c2[64], s2[64];
    vx27_rope_cos_sin(8, c1, s1, 64, 10000000.0f, 1.0f);
    vx27_rope_cos_sin(32, c2, s2, 64, 10000000.0f, 4.0f);
    float md = 0; for (int i = 0; i < 32; i++) { float d = fabsf(c1[i]-c2[i]); if (d>md) md=d; d=fabsf(s1[i]-s2[i]); if(d>md) md=d; }
    printf("  yarn pos32/f4 vs pos8/f1 maxdiff %.3e\n", (double)md);
    CHECK(md < 1e-5f, "yarn linear pos/factor");

    // partial: dims beyond 64 unchanged, first 64 norm preserved
    float v[256], w[256];
    for (int i = 0; i < 256; i++) { v[i] = (float)(i % 13) * 0.05f - 0.3f; w[i] = v[i]; }
    vx27_rope_cos_sin(7, cosv, sinv, 64, 10000000.0f, 1.0f);
    vx27_rope_apply(v, 1, 256, 64, cosv, sinv);
    ok = 1;
    for (int i = 64; i < 256; i++) if (v[i] != w[i]) ok = 0;
    CHECK(ok, "partial passthrough dims>=64");
    double n0 = 0, n1 = 0;
    for (int i = 0; i < 64; i++) { n0 += (double)w[i]*w[i]; n1 += (double)v[i]*v[i]; }
    printf("  norm before %.6f after %.6f\n", sqrt(n0), sqrt(n1));
    CHECK(fabs(n0 - n1) / (n0 + 1e-9) < 1e-5, "rotation preserves norm");

    // mrope text == standard
    float cm[64], sm[64];
    vx27_mrope_cos_sin_text(7, cm, sm);
    md = 0; for (int i = 0; i < 32; i++) { float d=fabsf(cm[i]-cosv[i]); if(d>md) md=d; d=fabsf(sm[i]-sinv[i]); if(d>md) md=d; }
    CHECK(md < 1e-7f, "mrope text == standard");
    if (fails) { printf("test_rope: %d FAIL\n", fails); return 1; }
    printf("test_rope: PASS\n");
    return 0;
}
