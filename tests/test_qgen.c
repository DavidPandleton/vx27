#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
#include "../src/vx27_qfwd.h"
#include "../src/vx27_rope.h"
#include "../src/vx27_bp3.h"
#include "../src/quant.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

// dequant 1 baris embed BP3
static void embed_row(const vx27_qt *e, int row, float *out, int H) {
    const vx27_bp3_128 *G = (const vx27_bp3_128*)e->blob + (size_t)row * (H/128);
    for (int k = 0; k < H/128; k++) {
        uint16_t h = G[k].scale_h; uint32_t u = h;
        uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
        if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
        else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
        else bits=sign|((exp+112u)<<23)|(mant<<13);
        float sc; memcpy(&sc,&bits,4);
        for (int i = 0; i < 128; i++) {
            uint32_t uu=((G[k].p[0*4+i/32]>>(i%32))&1)|(((G[k].p[1*4+i/32]>>(i%32))&1)<<1)|(((G[k].p[2*4+i/32]>>(i%32))&1)<<2);
            out[k*128+i] = (float)((uu>=4)?(int)uu-8:(int)uu) * sc;
        }
    }
}

int main(void) {
    char err[512] = {0};
    vx27_entry ec[4096], eh[4096]; int nc = 0, nh = 0;
    CHECK(vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof(err)) == 0, "list core");
    CHECK(vx27_list("build/vx27_head.vx27", eh, 4096, &nh, err, sizeof(err)) == 0, "list head");
    vx27_map mc = {0}, mh = {0};
    CHECK(vx27_mmap_open("build/vx27_core.vx27", &mc, err, sizeof(err)) == 0, "mmap core");
    CHECK(vx27_mmap_open("build/vx27_head.vx27", &mh, err, sizeof(err)) == 0, "mmap head");
    static const char *names[2048]; static vx27_qt tabs[2048]; int N = 0;
    for (int i = 0; i < nc; i++) { names[N]=ec[i].name; tabs[N].blob=vx27_mmap_blob(&mc,i,NULL); tabs[N].dtype=ec[i].dtype; tabs[N].rows=(int)ec[i].ne0; tabs[N].cols=(int)ec[i].ne1; N++; }
    for (int i = 0; i < nh; i++) { names[N]=eh[i].name; tabs[N].blob=vx27_mmap_blob(&mh,i,NULL); tabs[N].dtype=eh[i].dtype; tabs[N].rows=(int)eh[i].ne0; tabs[N].cols=(int)eh[i].ne1; N++; }
    vx27_qtab tab = {N, names, tabs};
    printf("  tabel=%d tensor\n", N);
    threadpool *tp = tp_create(8, err, sizeof(err));
    CHECK(tp != NULL, "tp 8");

    const vx27_qt *emb = vx27_qfind(&tab, "model.language_model.embed_tokens.weight");
    CHECK(emb != NULL, "embed ketemu");
    float *x0 = malloc(5120*4);
    embed_row(emb, 7, x0, 5120);
    double c0 = 0; for (int i = 0; i < 5120; i++) c0 += x0[i];
    printf("  embed[7] checksum=%.3f\n", c0);
    CHECK(c0 != 0.0, "embed tak-nol");

    enum { T = 3 };
    int ids1[3] = {-1,-1,-1}, ids2[3] = {-1,-1,-1};
    double m1 = 0, m2 = 0;
    vx27_qgen(&tab, tp, x0, T, ids1, &m1);
    printf("  gen1: [%d %d %d] %.0f ms (%.0f ms/token)\n", ids1[0], ids1[1], ids1[2], m1, m1/T);
    int ok = 1;
    for (int t = 0; t < T; t++) if (ids1[t] < 0 || ids1[t] >= 248320) ok = 0;
    CHECK(ok, "id token valid 0..248319");
    vx27_qgen(&tab, tp, x0, T, ids2, &m2);
    printf("  gen2: [%d %d %d] %.0f ms\n", ids2[0], ids2[1], ids2[2], m2);
    CHECK(ids1[0]==ids2[0] && ids1[1]==ids2[1] && ids1[2]==ids2[2], "deterministik multi-token");
    printf("  throughput ~%.2f tok/s (single-stream, tanpa MTP)\n", 1000.0/(m1/T));
    free(x0); tp_free(tp);
    vx27_mmap_close(&mc); vx27_mmap_close(&mh);
    if (fails) { printf("test_qgen: %d FAIL\n", fails); return 1; }
    printf("test_qgen: PASS\n");
    return 0;
}
