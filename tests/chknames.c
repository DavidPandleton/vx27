#include <stdio.h>
#include <string.h>
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"
int main(void) {
    vx27_entry ec[4096], eh[4096]; int nc=0, nh=0; char err[512]={0};
    vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof(err));
    vx27_list("build/vx27_head.vx27", eh, 4096, &nh, err, sizeof(err));
    const char *all[2048]; int N=0;
    for (int i=0;i<nc;i++) all[N++]=ec[i].name;
    for (int i=0;i<nh;i++) all[N++]=eh[i].name;
    int has(const char *nm){ for(int i=0;i<N;i++) if(!strcmp(all[i],nm)) return 1; return 0; }
    char nm[192]; int miss=0;
    const char *lin[] = {"input_layernorm.weight","linear_attn.in_proj_qkv.weight","linear_attn.in_proj_z.weight","linear_attn.in_proj_a.weight","linear_attn.in_proj_b.weight","linear_attn.A_log","linear_attn.dt_bias","linear_attn.conv1d.weight","linear_attn.norm.weight","linear_attn.out_proj.weight","post_attention_layernorm.weight","mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight",0};
    const char *ful[] = {"input_layernorm.weight","self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight","self_attn.q_norm.weight","self_attn.k_norm.weight","post_attention_layernorm.weight","mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight",0};
    for (int L=0;L<64;L++) {
        int f=((L+1)%4==0);
        const char **lst = f?ful:lin;
        for (int k=0;lst[k];k++) {
            snprintf(nm,sizeof nm,"model.language_model.layers.%d.%s",L,lst[k]);
            if(!has(nm)){printf("MISS L=%d %s\n",L,lst[k]);miss++;}
        }
    }
    const char *g[] = {"model.language_model.embed_tokens.weight","model.language_model.norm.weight","lm_head.weight",0};
    for (int k=0;g[k];k++) if(!has(g[k])){printf("MISS %s\n",g[k]);miss++;}
    printf(miss?"TOTAL MISS %d\n":"ALL PRESENT\n",miss);
    return miss!=0;
}
