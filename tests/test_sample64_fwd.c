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
#include "../src/vx27_st.h"
#include "../src/vx27_mmap.h"
#include "../src/vx27_block.h"
#include "../src/vx27_bp3.h"
#include "../src/quant_avx2.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

static float *deq(const vx27_entry *e, const void *blob) {
    uint64_t elems = e->ne0 * e->ne1;
    float *f = malloc((size_t)elems*4+64);
    if (e->dtype == 0) memcpy(f, blob, (size_t)elems*4);
    else if (e->dtype == 2) {
        const block_q8_k *q = blob; size_t nb = (size_t)elems/256;
        for (size_t b = 0; b < nb; b++) for (int i = 0; i < 256; i++) f[b*256+i] = (float)q[b].qs[i]*q[b].d;
    } else if (e->dtype == 4) {
        const vx27_bp3_128 *g = blob; size_t ng = (size_t)elems/128;
        for (size_t k = 0; k < ng; k++) {
            uint16_t h = g[k].scale_h; uint32_t u = h, sign=(u&0x8000u)<<16, exp=(u>>10)&0x1Fu, mant=u&0x3FFu, bits;
            if (exp==0) bits=mant? (sign|((exp+1+112u)<<23)|(mant<<13)):sign;
            else if (exp==31) bits=sign|0x7F800000u|(mant<<13);
            else bits=sign|((exp+112u)<<23)|(mant<<13);
            float sc; memcpy(&sc,&bits,4);
            for (int i = 0; i < 128; i++) {
                uint32_t uu=((g[k].p[0*4+i/32]>>(i%32))&1)|(((g[k].p[1*4+i/32]>>(i%32))&1)<<1)|(((g[k].p[2*4+i/32]>>(i%32))&1)<<2);
                int q=(uu>=4)?(int)uu-8:(int)uu;
                f[k*128+i]=q*sc;
            }
        }
    } else { free(f); return NULL; }
    return f;
}
static float *findf(vx27_entry **E, float **all, int N, const char *name) {
    for (int i = 0; i < N; i++) if (!strcmp(E[i]->name, name)) return all[i];
    return NULL;
}
#define FINDF(nm) findf(E, all, N, nm)
int main(void) {
    int rc = system("build\\vx_forge.exe forge --in build\\sample_heretic64.safetensors --out-core build\\sample64_core.vx27 --out-head build\\sample64_head.vx27 --bits 8");
    CHECK(rc == 0, "forge 64-layer bits=8");
    vx27_map mc = {0}, mh = {0}; char err[512] = {0};
    CHECK(vx27_mmap_open("build/sample64_core.vx27", &mc, err, sizeof(err)) == 0, "mmap core64");
    CHECK(vx27_mmap_open("build/sample64_head.vx27", &mh, err, sizeof(err)) == 0, "mmap head64");
    vx27_entry ec[700], eh[700]; int nce = 0, nhe = 0;
    CHECK(vx27_list("build/sample64_core.vx27", ec, 700, &nce, err, sizeof(err)) == 0, "list core64");
    CHECK(vx27_list("build/sample64_head.vx27", eh, 700, &nhe, err, sizeof(err)) == 0, "list head64");
    printf("  core=%d head=%d total=%d\n", nce, nhe, nce+nhe);
    CHECK(nce == 495 && nhe == 117, "split 495/117");
    // kumpulkan semua entri+blob
    vx27_entry *E[800]; const void *B[800]; int N = 0;
    for (int i = 0; i < nce; i++) { uint64_t bb=0; const void *p = vx27_mmap_blob(&mc, i, &bb); E[N]=&ec[i]; B[N]=p; N++; }
    for (int i = 0; i < nhe; i++) { uint64_t bb=0; const void *p = vx27_mmap_blob(&mh, i, &bb); E[N]=&eh[i]; B[N]=p; N++; }
    // dequant per-layer on demand dalam loop 64 (hemat memori): cache float per tensor via array
    // Sederhananya dequant semua yang dibutuhkan sekaligus (2M floats ~8MB, aman).
    vx27_block_cfg bc = {64, 128, 4, 2, 8, 8, 2, 2, 8, 10000.0f, 1.0f};
    char werr[256]={0}; (void)werr;
    threadpool *tp = tp_create(8, err, sizeof(err));
    CHECK(tp != NULL, "tp 8");
    // embed + norm
    float *all[800]={0};
    for (int i = 0; i < N; i++) { all[i]=deq(E[i],B[i]); if(!all[i]){printf("deq fail %s\n",E[i]->name);fails++;} }
    CHECK(fails==0, "dequant all 612");
    float h[64]; const float *emb = FINDF("model.embed_tokens.weight");
    for (int i = 0; i < 64; i++) h[i] = emb[7*64+i];
    float *Sall = calloc((size_t)48*2*8*8, 4);
    float *kall = calloc((size_t)16*4*2*8, 4), *vall = calloc((size_t)16*4*2*8, 4);
    // siapkan bobot per layer (pointer ke float terdequant)
    double t0 = now_ms();
    float cur[64], nxt[64]; for (int i=0;i<64;i++) cur[i]=h[i];
    int s_li = 0, f_li = 0;
    char nm[128];
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        if (!is_full) {
            snprintf(nm,sizeof nm,"model.layers.%d.linear_attn.q_proj.weight",L); float *Wq=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.linear_attn.k_proj.weight",L); float *Wk=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.linear_attn.v_proj.weight",L); float *Wv=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.linear_attn.o_proj.weight",L); float *Wo=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.gate_proj.weight",L); float *Wg=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.up_proj.weight",L); float *Wu=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.down_proj.weight",L); float *Wd=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.input_layernorm.weight",L); float *an=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.post_attention_layernorm.weight",L); float *fn=FINDF(nm);
            vx27_block_weights w={an,fn,Wq,Wk,Wv,Wo,Wg,Wu,Wd,NULL};
            float *S = Sall + (size_t)s_li*2*8*8; s_li++;
            vx27_linear_block_forward(&bc, NULL, cur, S, nxt, &w);
        } else {
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.q_proj.weight",L); float *Wq=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.k_proj.weight",L); float *Wk=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.v_proj.weight",L); float *Wv=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.o_proj.weight",L); float *Wo=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.q_norm.weight",L); float *qn=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.gate_proj.weight",L); float *Wg=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.up_proj.weight",L); float *Wu=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.down_proj.weight",L); float *Wd=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.input_layernorm.weight",L); float *an=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.post_attention_layernorm.weight",L); float *fn=FINDF(nm);
            vx27_block_weights w={an,fn,Wq,Wk,Wv,Wo,Wg,Wu,Wd,qn};
            float *kc = kall + (size_t)f_li*4*2*8, *vc = vall + (size_t)f_li*4*2*8; f_li++;
            vx27_full_block_forward(&bc, NULL, cur, kc, vc, 0, nxt, &w);
        }
        for (int i = 0; i < 64; i++) cur[i] = nxt[i];
    }
    double t1 = now_ms();
    double chk = 0; for (int i = 0; i < 64; i++) chk += cur[i];
    printf("  fwd64 ref checksum=%.6f ms=%.2f\n", chk, t1-t0);
    // ulangi dengan threadpool untuk determinisme
    memset(Sall,0,(size_t)48*2*8*8*4); memset(kall,0,(size_t)16*4*2*8*4); memset(vall,0,(size_t)16*4*2*8*4);
    for (int i = 0; i < 64; i++) cur[i] = h[i];
    s_li = 0; f_li = 0;
    t0 = now_ms();
    for (int L = 0; L < 64; L++) {
        int is_full = ((L+1)%4==0);
        if (!is_full) {
            snprintf(nm,sizeof nm,"model.layers.%d.linear_attn.q_proj.weight",L); float *Wq=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.linear_attn.k_proj.weight",L); float *Wk=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.linear_attn.v_proj.weight",L); float *Wv=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.linear_attn.o_proj.weight",L); float *Wo=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.gate_proj.weight",L); float *Wg=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.up_proj.weight",L); float *Wu=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.down_proj.weight",L); float *Wd=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.input_layernorm.weight",L); float *an=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.post_attention_layernorm.weight",L); float *fn=FINDF(nm);
            vx27_block_weights w={an,fn,Wq,Wk,Wv,Wo,Wg,Wu,Wd,NULL};
            float *S = Sall + (size_t)s_li*2*8*8; s_li++;
            vx27_linear_block_forward(&bc, tp, cur, S, nxt, &w);
        } else {
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.q_proj.weight",L); float *Wq=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.k_proj.weight",L); float *Wk=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.v_proj.weight",L); float *Wv=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.o_proj.weight",L); float *Wo=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.self_attn.q_norm.weight",L); float *qn=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.gate_proj.weight",L); float *Wg=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.up_proj.weight",L); float *Wu=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.down_proj.weight",L); float *Wd=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.input_layernorm.weight",L); float *an=FINDF(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.post_attention_layernorm.weight",L); float *fn=FINDF(nm);
            vx27_block_weights w={an,fn,Wq,Wk,Wv,Wo,Wg,Wu,Wd,qn};
            float *kc = kall + (size_t)f_li*4*2*8, *vc = vall + (size_t)f_li*4*2*8; f_li++;
            vx27_full_block_forward(&bc, tp, cur, kc, vc, 0, nxt, &w);
        }
        for (int i = 0; i < 64; i++) cur[i] = nxt[i];
    }
    t1 = now_ms();
    double chk2 = 0; for (int i = 0; i < 64; i++) chk2 += cur[i];
    printf("  fwd64 tp checksum=%.6f ms=%.2f\n", chk2, t1-t0);
    CHECK(fabs(chk-chk2) < 1e-3, "64-layer tp == ref");
    for (int i = 0; i < N; i++) free(all[i]);
    free(Sall); free(kall); free(vall); tp_free(tp);
    vx27_mmap_close(&mc); vx27_mmap_close(&mh);
    if (fails) { printf("test_sample64_fwd: %d FAIL\n", fails); return 1; }
    printf("test_sample64_fwd: PASS\n");
    return 0;
}
