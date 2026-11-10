#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) { return (double)clock()*1000.0/CLOCKS_PER_SEC; }
#endif
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
#include "../src/vx27_qfwd.h"
#include "../src/vx27_rope.h"
#include "../src/vx27_gpu_reg.h"
#include "../src/vx27_bp3.h"
#include "../src/quant.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

// 1x forward 64 layer penuh; kembalikan checksum + argmax + waktu
static float fwd64(vx27_qtab *tab, threadpool *tp, const float *x0,
                   int *argmax_out, double *ms_out) {
    const int H = 5120;
    float *h1 = malloc(H*4), *h2 = malloc(H*4);
    for (int i = 0; i < H; i++) h1[i] = x0[i];
    float *Sall = calloc((size_t)48*VX27_SSTATE_PER_LAYER, 4);
    float *Call = calloc((size_t)48*VX27_CSTATE_PER_LAYER, 4);
    float *kv = calloc((size_t)16*(size_t)VX27_QGEN_T*4*256*2, 4);
    int full_idx[64]; int fs = 0;
    for (int L = 0; L < 64; L++) { int f = ((L+1)%4==0); full_idx[L] = f ? fs++ : -1; }
    float cosv[32], sinv[32];
    vx27_rope_cos_sin(0, cosv, sinv, 64, 10000000.0f, 1.0f);
    float *Slin[64], *Clin[64]; int li = 0, ci = 0;
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        if (is_full) { Slin[L] = NULL; Clin[L] = NULL; }
        else { Slin[L] = Sall + (size_t)(li++)*VX27_SSTATE_PER_LAYER; Clin[L] = Call + (size_t)(ci++)*VX27_CSTATE_PER_LAYER; }
    }
    double t0 = now_ms();
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        vx27_qlayer(tab, L, is_full, h1, h2, 0, full_idx, Slin[L], kv, cosv, sinv, tp, Clin[L]);
        float *tt = h1; h1 = h2; h2 = tt;
    }
    float *logits = malloc(248320*4);
    int arg = vx27_qlogits(tab, h1, logits, tp);
    double t1 = now_ms();
    double chk = 0; for (int i = 0; i < H; i++) chk += h1[i];
    if (argmax_out) *argmax_out = arg;
    if (ms_out) *ms_out = t1 - t0;
    free(logits); free(h1); free(h2); free(Sall); free(Call); free(kv);
    return (float)chk;
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
    for (int i = 0; i < nc; i++) {
        uint64_t bb = 0; const void *p = vx27_mmap_blob(&mc, i, &bb);
        if (!p) { printf("FAIL: blob core %d\n", i); fails++; continue; }
        names[N] = ec[i].name; tabs[N].blob = p; tabs[N].dtype = ec[i].dtype;
        tabs[N].rows = (int)ec[i].ne0; tabs[N].cols = (int)ec[i].ne1; N++;
    }
    for (int i = 0; i < nh; i++) {
        uint64_t bb = 0; const void *p = vx27_mmap_blob(&mh, i, &bb);
        if (!p) { printf("FAIL: blob head %d\n", i); fails++; continue; }
        names[N] = eh[i].name; tabs[N].blob = p; tabs[N].dtype = eh[i].dtype;
        tabs[N].rows = (int)eh[i].ne0; tabs[N].cols = (int)eh[i].ne1; N++;
    }
    vx27_qtab tab = {N, names, tabs};
    printf("  tabel=%d\n", N);

    // daftarkan tensor BP3 head (kecuali lm_head + mtp) ke VRAM
    static const void *blobs[256]; static int rws[256], cls[256]; int nb = 0;
    for (int i = 0; i < N && nb < 256; i++) {
        if (tabs[i].dtype != 4) continue;
        const char *nm = names[i];
        if (!strncmp(nm, "lm_head", 7) || strstr(nm, "mtp.") || strstr(nm, "model.norm")) continue;
        int ishead = 0;
        if (strstr(nm, "model.language_model.layers.")) {
            int L = atoi(strstr(nm, "layers.") + 7);
            if (L >= 52) ishead = 1;
        } else continue;
        if (!ishead) continue;
        blobs[nb] = tabs[i].blob; rws[nb] = tabs[i].rows; cls[nb] = tabs[i].cols; nb++;
    }
    printf("  head BP3 utk GPU: %d tensor\n", nb);
    CHECK(nb > 100, "cukup tensor head (>100)");
    CHECK(vx27_gpu_reg_init(blobs, rws, cls, nb, err, sizeof(err)) == 0, "upload head ke VRAM");
    if (fails) { printf("test_qfwd_hybrid: SKIP/BATAL (VRAM) %s\n", err); return 1; }

    threadpool *tp = tp_create(8, err, sizeof(err));
    const vx27_qt *emb = vx27_qfind(&tab, "model.language_model.embed_tokens.weight");
    float x0[5120];
    {
        const vx27_bp3_128 *G = (const vx27_bp3_128 *)emb->blob + 7 * (5120/128);
        for (int k = 0; k < 5120/128; k++) {
            uint16_t h = G[k].scale_h; uint32_t u = h;
            uint32_t sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
            if (exp==0) bits=mant? (sign|((exp+1u)<<23)|(mant<<13)):sign;
            else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
            else bits=sign|((exp+112u)<<23)|(mant<<13);
            float sc; memcpy(&sc,&bits,4);
            for (int j = 0; j < 128; j++) {
                uint32_t uu=((G[k].p[0*4+j/32]>>(j%32))&1)|(((G[k].p[1*4+j/32]>>(j%32))&1)<<1)|(((G[k].p[2*4+j/32]>>(j%32))&1)<<2);
                x0[k*128+j] = (float)((uu>=4)?(int)uu-8:(int)uu) * sc;
            }
        }
    }

    // baseline CPU (registry mati)
    int a0 = -1; double m0 = 0;
    float c0 = fwd64(&tab, tp, x0, &a0, &m0);
    printf("  CPU-only: checksum=%.4f argmax=%d %.0f ms\n", c0, a0, m0);

    // hybrid (registry nyala, head di GPU)
    vx27_gpu_reg_on();
    int a1 = -1; double m1 = 0;
    float c1 = fwd64(&tab, tp, x0, &a1, &m1);
    printf("  hybrid CPU+GPU: checksum=%.4f argmax=%d %.0f ms\n", c1, a1, m1);
    vx27_gpu_reg_off();

    double rel = fabs((double)c1 - (double)c0) / (fabs((double)c0) + 1e-9);
    printf("  relerr=%.3e speedup=%.2fx\n", rel, m0 / (m1 + 1e-9));
    CHECK(rel < 0.05, "numerik hybrid ~= CPU (5%)");
    CHECK(a0 == a1, "argmax sama");
    tp_free(tp);
    vx27_mmap_close(&mc); vx27_mmap_close(&mh);
    vx27_gpu_reg_free();
    if (fails) { printf("test_qfwd_hybrid: %d FAIL\n", fails); return 1; }
    printf("test_qfwd_hybrid: PASS\n");
    return 0;
}
