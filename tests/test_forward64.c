#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
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
#include "../src/vx27_forward64.h"
#include "../src/vx27_block.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)
static float frand(unsigned *s) { *s = *s*1664525u+1013904223u; return ((int)(*s>>9)&0xffff)/32768.0f-1.0f; }

int main(void) {
    vx27_block_cfg bc = {32, 64, 4, 2, 8, 8, 2, 2, 8, 10000.0f, 1.0f};
    int H = bc.hidden, F = bc.ffn;
    unsigned seed = 777;
    // full weights
    size_t nWq=(size_t)H*bc.n_q*bc.hd, nWk=(size_t)H*bc.n_kv*bc.hd, nWo=(size_t)bc.n_q*bc.hd*H;
    float *an=malloc(H*4), *fn=malloc(H*4), *qn=malloc((size_t)bc.hd*4);
    float *Wq=malloc(nWq*4), *Wk=malloc(nWk*4), *Wv=malloc(nWk*4), *Wo=malloc(nWo*4);
    float *Wg=malloc((size_t)H*F*4), *Wu=malloc((size_t)H*F*4), *Wd=malloc((size_t)F*H*4);
    for (int i=0;i<H;i++){an[i]=1;fn[i]=1;} for(int i=0;i<bc.hd;i++)qn[i]=1;
    for(size_t i=0;i<nWq;i++) Wq[i]=frand(&seed)*0.1f;
    for(size_t i=0;i<nWk;i++){Wk[i]=frand(&seed)*0.1f;Wv[i]=frand(&seed)*0.1f;}
    for(size_t i=0;i<nWo;i++)Wo[i]=frand(&seed)*0.1f;
    for(size_t i=0;i<(size_t)H*F;i++){Wg[i]=frand(&seed)*0.1f;Wu[i]=frand(&seed)*0.1f;}
    for(size_t i=0;i<(size_t)F*H;i++)Wd[i]=frand(&seed)*0.1f;
    vx27_block_weights wf={an,fn,Wq,Wk,Wv,Wo,Wg,Wu,Wd,qn};
    // linear weights
    size_t nLq=(size_t)H*bc.lin_qk*bc.lin_hd, nLv=(size_t)H*bc.lin_v*bc.lin_hd, nLo=(size_t)bc.lin_v*bc.lin_hd*H;
    float *Lq=malloc(nLq*4),*Lk=malloc(nLq*4),*Lv=malloc(nLv*4),*Lo=malloc(nLo*4);
    for(size_t i=0;i<nLq;i++){Lq[i]=frand(&seed)*0.1f;Lk[i]=frand(&seed)*0.1f;}
    for(size_t i=0;i<nLv;i++)Lv[i]=frand(&seed)*0.1f;
    for(size_t i=0;i<nLo;i++)Lo[i]=frand(&seed)*0.1f;
    vx27_block_weights wl={an,fn,Lq,Lk,Lv,Lo,Wg,Wu,Wd,NULL};

    float *x0=malloc(H*4), *o1=malloc(H*4), *o2=malloc(H*4);
    for(int i=0;i<H;i++)x0[i]=frand(&seed);
    int nkvhd=bc.n_kv*bc.hd, sper=bc.lin_qk*bc.lin_hd*bc.lin_hd;
    float *kc1=calloc((size_t)16*4*nkvhd,4), *vc1=calloc((size_t)16*4*nkvhd,4), *S1=calloc((size_t)48*sper,4);
    float *kc2=calloc((size_t)16*4*nkvhd,4), *vc2=calloc((size_t)16*4*nkvhd,4), *S2=calloc((size_t)48*sper,4);

    char err[256]={0};
    threadpool *tp=tp_create(8,err,sizeof(err));
    CHECK(tp!=NULL,"threadpool 8");
    float c1=vx27_forward64_proxy(&bc,NULL,x0,o1,&wf,&wl,kc1,vc1,S1);
    float c2=vx27_forward64_proxy(&bc,tp,x0,o2,&wf,&wl,kc2,vc2,S2);
    printf("  checksum ref=%.6f tp=%.6f\n",c1,c2);
    CHECK(fabsf(c1-c2)<1e-3f,"64-layer tp == ref");
    float c3; { float *kc=calloc((size_t)16*4*nkvhd,4),*vc=calloc((size_t)16*4*nkvhd,4),*S=calloc((size_t)48*sper,4),*o=malloc(H*4);
        c3=vx27_forward64_proxy(&bc,tp,x0,o,&wf,&wl,kc,vc,S); free(kc);free(vc);free(S);free(o); }
    CHECK(c2==c3,"64-layer deterministic");

    const int iters=20;
    double t0=now_ms();
    for(int it=0;it<iters;it++){ float *kc=calloc((size_t)16*4*nkvhd,4),*vc=calloc((size_t)16*4*nkvhd,4),*S=calloc((size_t)48*sper,4),*o=malloc(H*4);
        vx27_forward64_proxy(&bc,tp,x0,o,&wf,&wl,kc,vc,S); free(kc);free(vc);free(S);free(o); }
    double t1=now_ms();
    printf("  bench 64-layer proxy x%d: %.2f ms (%.3f ms/pass)\n",iters,t1-t0,(t1-t0)/iters);
    printf("  note: proxy H32/F64; real H5120/F17408 is ~25700x matvec FLOPs (see ARCH)\n");
    tp_free(tp);
    free(an);free(fn);free(qn);free(Wq);free(Wk);free(Wv);free(Wo);free(Wg);free(Wu);free(Wd);
    free(Lq);free(Lk);free(Lv);free(Lo);free(x0);free(o1);free(o2);free(kc1);free(vc1);free(S1);free(kc2);free(vc2);free(S2);
    if(fails){printf("test_forward64: %d FAIL\n",fails);return 1;}
    printf("test_forward64: PASS\n");
    return 0;
}
