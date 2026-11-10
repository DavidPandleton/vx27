#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../src/vx27_ffn.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // tiny + production-width slice
    const int tiny = 128;
    float *g = malloc(tiny*4), *u = malloc(tiny*4), *h1 = malloc(tiny*4), *h2 = malloc(tiny*4);
    for (int i = 0; i < tiny; i++) { g[i] = (float)(i%17)*0.2f-1.6f; u[i] = (float)(i%9)*0.3f-1.2f; }
    vx27_swiglu_forward_ref(g, u, h1, tiny);
    vx27_swiglu_forward_avx2(g, u, h2, tiny);
    float md = 0; for (int i = 0; i < tiny; i++) { float d=fabsf(h1[i]-h2[i]); if(d>md) md=d; }
    printf("  swiglu tiny maxdiff %.3e\n", (double)md);
    CHECK(md < 1e-6f, "swiglu avx2 == ref tiny");

    // ffn=17408 slice with non-multiple-of-8 tail (17408%8==0, test 17409 for tail)
    const int big = 17409;
    float *gb = malloc((size_t)big*4), *ub = malloc((size_t)big*4), *hb1 = malloc((size_t)big*4), *hb2 = malloc((size_t)big*4);
    for (int i = 0; i < big; i++) { gb[i]=(float)(i%19)*0.11f-1.0f; ub[i]=(float)(i%23)*0.07f-0.8f; }
    vx27_swiglu_forward_ref(gb, ub, hb1, big);
    vx27_swiglu_forward_avx2(gb, ub, hb2, big);
    md = 0; for (int i = 0; i < big; i++) { float d=fabsf(hb1[i]-hb2[i]); if(d>md) md=d; }
    printf("  swiglu 17409 maxdiff %.3e\n", (double)md);
    CHECK(md < 1e-6f, "swiglu avx2 == ref 17409 tail");
    // silu sanity: silu(0)=0, silu(1)~0.731
    float gs[2]={0.0f,1.0f}, us[2]={5.0f,1.0f}, hs[2];
    vx27_swiglu_forward_ref(gs, us, hs, 2);
    CHECK(fabsf(hs[0])<1e-6f && fabsf(hs[1]-0.7310586f)<1e-5f, "silu sanity");
    free(g); free(u); free(h1); free(h2); free(gb); free(ub); free(hb1); free(hb2);
    if (fails) { printf("test_ffn: %d FAIL\n", fails); return 1; }
    printf("test_ffn: PASS\n");
    return 0;
}
