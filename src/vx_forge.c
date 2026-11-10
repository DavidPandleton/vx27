#include "vx27_config.h"
#include "vx27_st.h"
#include "vx27_vxfile.h"
#include "vx27_bp3.h"
#include "quant.h"
#include "quant_avx2.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

// Param inventory (estimates, documented as such):
// token_embd + lm_head untied; per-layer from dims; MTP ~1 layer FFN+proj.
static void inventory(double *backbone, double *emb_head, double *mtp) {
    double tok = (double)VX27_VOCAB * VX27_HIDDEN; // 1.271B
    double lm = tok;
    // full layer: Q(5120*6144)+K(5120*1024)+V(5120*1024)+O(6144*5120)+FFN(3*5120*17408)
    double full = 5120.0*6144 + 5120.0*1024*2 + 6144.0*5120 + 3.0*5120.0*17408;
    // linear layer: Q(5120*2048)+K(5120*2048)+V(5120*6144)+O(6144*5120)+FFN(same)
    double lin = 5120.0*2048*2 + 5120.0*6144 + 6144.0*5120 + 3.0*5120.0*17408;
    double bb = VX27_N_FULL * full + VX27_N_LINEAR * lin;
    *backbone = bb;
    *emb_head = tok + lm;
    *mtp = 5120.0*17408 + 5120.0*5120; // rough: 1 FFN-ish + proj
}

static void usage(void) {
    printf("usage:\n");
    printf("  vx_forge.exe plan [--out manifest.json]\n");
    printf("  vx_forge.exe scan --in model.safetensors [--out manifest.json]\n");
    printf("  vx_forge.exe forge (--in F | --in-list L.txt) --out-core core.vx27 --out-head head.vx27 --bits 8|3 [--split-at 52] [--keep-vision]\n");
}

static int cmd_plan(const char *out) {
    double bb = 0, eh = 0, mtp = 0;
    inventory(&bb, &eh, &mtp);
    double total_text = bb + eh + mtp;
    // split: L0-51 core = 52 layers proportionally (39 lin + 13 full, since every 4th is full)
    // layers 0..51 contain full layers 3,7,...,51 -> 13 full + 39 lin
    double full = 5120.0*6144 + 5120.0*1024*2 + 6144.0*5120 + 3.0*5120.0*17408;
    double lin = 5120.0*2048*2 + 5120.0*6144 + 6144.0*5120 + 3.0*5120.0*17408;
    double core_params = 39 * lin + 13 * full;
    double head_backbone = bb - core_params;
    double head_params = head_backbone + eh + mtp;

    printf("VORTEX-27 forge plan (text-only, vision dropped)\n");
    printf("backbone_est=%.3fB emb+head=%.3fB mtp_est=%.3fB total_text=%.3fB\n",
        bb / 1e9, eh / 1e9, mtp / 1e9, total_text / 1e9);
    printf("core L0-51: %.3fB params -> budget %.1fGB needs <=%.2f bits/param\n",
        core_params / 1e9, VX27_CORE_GB, VX27_CORE_GB * 8e9 / core_params);
    printf("head L52-63+emb/mtp: %.3fB params -> budget %.1fGB needs <=%.2f bits/param\n",
        head_params / 1e9, VX27_HEAD_GB, VX27_HEAD_GB * 8e9 / head_params);
    printf("note: card total 27.36B incl 0.46B vision; text-only target ~26.9B\n");

    if (out) {
        FILE *f = fopen(out, "w");
        if (!f) { printf("cannot write %s\n", out); return 1; }
        fprintf(f, "{\n");
        fprintf(f, "  \"model\": \"Qwen3.8-27B Heretic-Abliterated text-only\",\n");
        fprintf(f, "  \"hidden\": %d, \"layers\": %d, \"full_interval\": %d,\n",
            VX27_HIDDEN, VX27_LAYERS, VX27_FULL_INTERVAL);
        fprintf(f, "  \"vocab\": %d, \"ffn\": %d, \"mtp_layers\": %d,\n",
            VX27_VOCAB, VX27_FFN, VX27_MTP_LAYERS);
        fprintf(f, "  \"core_layers\": \"0-51\", \"core_budget_gb\": %.1f,\n", VX27_CORE_GB);
        fprintf(f, "  \"head_layers\": \"52-63+lm_head+mtp\", \"head_budget_gb\": %.1f,\n", VX27_HEAD_GB);
        fprintf(f, "  \"kv_budget_gb\": %.1f, \"ctx_default\": %d, \"ctx_stretch\": %d,\n",
            VX27_KV_BUDGET_GB, VX27_CTX_DEFAULT, VX27_CTX_STRETCH);
        fprintf(f, "  \"backbone_est_B\": %.3f, \"core_params_B\": %.3f, \"head_params_B\": %.3f\n",
            bb / 1e9, core_params / 1e9, head_params / 1e9);
        fprintf(f, "}\n");
        fclose(f);
        printf("wrote %s\n", out);
    }
    return 0;
}

static int cmd_scan(const char *in, const char *out) {
    vx27_st_entry ents[4096];
    int n = 0; uint64_t total = 0; char err[256] = {0};
    if (vx27_st_open(in, ents, 4096, &n, &total, err, sizeof(err)) != 0) {
        printf("scan FAIL: %s\n", err); return 1;
    }
    printf("scan %s: %d tensors, %.3f GB data\n", in, n, (double)total / 1e9);
    int shown = n < 10 ? n : 10;
    for (int i = 0; i < shown; i++)
        printf("  %s %s elems=%llu bytes=%llu\n", ents[i].name, ents[i].dtype,
            (unsigned long long)ents[i].elems, (unsigned long long)ents[i].bytes);
    if (n > shown) printf("  ... +%d more\n", n - shown);
    if (out) {
        FILE *f = fopen(out, "w");
        if (!f) { printf("cannot write %s\n", out); return 1; }
        fprintf(f, "{\"input\":\"%s\",\"tensors\":%d,\"data_gb\":%.3f}\n",
            in, n, (double)total / 1e9);
        fclose(f);
        printf("wrote %s\n", out);
    }
    return 0;
}

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, 4); return f;
}

static int is_vision_tensor(const char *name) {
    return strstr(name, "visual") != NULL || strstr(name, "vision") != NULL ||
           strstr(name, "patch") != NULL || strstr(name, "tower") != NULL;
}
static int layer_of(const char *name) {
    const char *p = strstr(name, "layers.");
    if (!p) return -1;
    return atoi(p + 7);
}
// splitAt=52: core = embed + layers with L < 52 (full-model / real-shard mode).
// splitAt=1: core = embed + layers.0 saja (sampel 2-layer pedagogis).
static int is_core_tensor_n(const char *name, int splitAt) {
    if (strstr(name, "embed") != NULL) return 1;
    int L = layer_of(name);
    if (L < 0) return 0; // lm/mtp/norm -> head
    return L < splitAt;
}

static int cmd_forge_multi(const char **ins, int n_in, const char *out_core, const char *out_head, int bits, int splitAt, int skipVision);

static int cmd_forge(const char *in, const char *out_core, const char *out_head, int bits, int splitAt, int skipVision) {
    return cmd_forge_multi(&in, 1, out_core, out_head, bits, splitAt, skipVision);
}

static int cmd_forge_multi(const char **ins, int n_in, const char *out_core, const char *out_head, int bits, int splitAt, int skipVision) {
    // per-tensor outputs (1199 total across 18 shards; heap, bukan stack)
    enum { CAP = 2048 };
    const char **cn = malloc(CAP * sizeof(*cn)), **hn = malloc(CAP * sizeof(*hn));
    uint32_t *cd = malloc(CAP * sizeof(*cd)), *hd = malloc(CAP * sizeof(*hd));
    uint64_t *cn0 = malloc(CAP * sizeof(*cn0)), *cn1 = malloc(CAP * sizeof(*cn1));
    uint64_t *hn0 = malloc(CAP * sizeof(*hn0)), *hn1 = malloc(CAP * sizeof(*hn1));
    const void **cb = malloc(CAP * sizeof(*cb)), **hb = malloc(CAP * sizeof(*hb));
    uint64_t *cbb = malloc(CAP * sizeof(*cbb)), *hbb = malloc(CAP * sizeof(*hbb));
    void **owned = malloc(CAP * 2 * sizeof(*owned)); int nowned = 0;
    char **names = malloc(CAP * 2 * sizeof(*names)); int nnames = 0;
    if (!cn || !hn || !cd || !hd || !cn0 || !cn1 || !hn0 || !hn1 || !cb || !hb || !cbb || !hbb || !owned || !names)
        { printf("forge FAIL: oom\n"); return 1; }
    int nc = 0, nh = 0, nskip = 0, ntot = 0;

    for (int fi = 0; fi < n_in; fi++) {
    const char *in = ins[fi];
    vx27_st_entry *ents = malloc(4096 * sizeof(*ents));
    int n = 0; uint64_t total = 0; char err[256] = {0};
    if (vx27_st_open(in, ents, 4096, &n, &total, err, sizeof(err)) != 0) {
        printf("forge FAIL %s: %s\n", in, err); return 1;
    }
    FILE *f = fopen(in, "rb");
    if (!f) { printf("forge FAIL: cannot open %s\n", in); return 1; }
    uint8_t lb[8]; if (fread(lb, 1, 8, f) != 8) { fclose(f); return 1; }
    uint64_t hl = vx27_st_u64le(lb);
    uint64_t data_start = 8 + hl;

    for (int i = 0; i < n; i++) {
        if (skipVision && is_vision_tensor(ents[i].name)) { nskip++; continue; }
        ntot++;
        uint64_t elems = ents[i].elems;
        // read raw BF16
        uint8_t *raw = malloc((size_t)ents[i].bytes);
#ifdef _WIN32
        _fseeki64(f, (__int64)(data_start + ents[i].offset0), SEEK_SET);
#else
        fseek(f, (long)(data_start + ents[i].offset0), SEEK_SET);
#endif
        if (fread(raw, 1, (size_t)ents[i].bytes, f) != (size_t)ents[i].bytes) {
            printf("forge FAIL: trunc %s\n", ents[i].name); fclose(f); return 1;
        }
        float *fl = malloc((size_t)elems * 4);
        if (!strcmp(ents[i].dtype, "BF16")) {
            uint16_t *u16 = (uint16_t*)raw;
            for (uint64_t k = 0; k < elems; k++) fl[k] = bf16_to_f32(u16[k]);
        } else if (!strcmp(ents[i].dtype, "F32")) {
            memcpy(fl, raw, (size_t)elems * 4);
        } else {
            printf("forge FAIL: dtype %s unsupported in sample\n", ents[i].dtype);
            free(raw); free(fl); fclose(f); return 1;
        }
        free(raw);

        uint32_t dt; const void *blob; uint64_t bb;        void *keep = NULL;
        if (elems <= 256) {
            dt = VX27_D_F32; blob = fl; bb = elems * 4; keep = fl;
        } else if (bits == 8) {
            if (elems % 256 != 0) { dt = VX27_D_F32; blob = fl; bb = elems * 4; keep = fl; }
            else {
                size_t nb = (size_t)elems / 256;
                block_q8_k *q = malloc(nb * sizeof(block_q8_k));
                quantize_q8_k(fl, q, (int)nb);
                free(fl);
                dt = VX27_D_Q8; blob = q; bb = nb * sizeof(block_q8_k); keep = q;
            }
        } else {
            if (elems % 128 != 0) { dt = VX27_D_F32; blob = fl; bb = elems * 4; keep = fl; }
            else {
                size_t ng = (size_t)elems / 128;
                vx27_bp3_128 *g = malloc(ng * sizeof(vx27_bp3_128));
                for (size_t k = 0; k < ng; k++) vx27_bp3_128_pack(fl + k*128, &g[k]);
                free(fl);
                dt = VX27_D_BP3; blob = g; bb = ng * sizeof(vx27_bp3_128); keep = g;
            }
        }
        owned[nowned++] = keep;
        { char *cp = malloc(260); snprintf(cp, 260, "%s", ents[i].name); names[nnames++] = cp; }
        if (nc >= 2048 - 1 || nh >= 2048 - 1) { printf("forge FAIL: cap\n"); fclose(f); free(ents); return 1; }
        if (is_core_tensor_n(ents[i].name, splitAt)) {
            cn[nc] = names[nnames-1]; cd[nc] = dt; cn0[nc] = ents[i].s0; cn1[nc] = ents[i].s1;
            cb[nc] = blob; cbb[nc] = bb; nc++;
        } else {
            hn[nh] = names[nnames-1]; hd[nh] = dt; hn0[nh] = ents[i].s0; hn1[nh] = ents[i].s1;
            hb[nh] = blob; hbb[nh] = bb; nh++;
        }
    }
    fclose(f);
    free(ents);
    } // end per-file loop
    // vx_write copies blob bytes synchronously; names are stable mallocs
    char werr[256] = {0};
    if (vx27_write(out_core, cn, cd, cn0, cn1, cb, cbb, nc, werr, sizeof(werr)) != 0) {
        printf("forge FAIL core: %s\n", werr); return 1;
    }
    if (vx27_write(out_head, hn, hd, hn0, hn1, hb, hbb, nh, werr, sizeof(werr)) != 0) {
        printf("forge FAIL head: %s\n", werr); return 1;
    }
    uint64_t cb_tot = 0, hb_tot = 0;
    for (int i = 0; i < nc; i++) cb_tot += cbb[i];
    for (int i = 0; i < nh; i++) hb_tot += hbb[i];
    printf("forge %d files bits=%d splitAt=%d: %d tensors (%d vision skipped) -> core %d (%.3f MB) + head %d (%.3f MB)\n",
        n_in, bits, splitAt, ntot, nskip, nc, (double)cb_tot/1e6, nh, (double)hb_tot/1e6);
    for (int i = 0; i < nowned; i++) free(owned[i]);
    for (int i = 0; i < nnames; i++) free(names[i]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    if (!strcmp(argv[1], "plan")) {
        const char *out = NULL;
        for (int i = 2; i < argc; i++)
            if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        return cmd_plan(out);
    }
    if (!strcmp(argv[1], "scan")) {
        const char *in = NULL, *out = NULL;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--in") && i + 1 < argc) in = argv[++i];
            else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        }
        if (!in) { usage(); return 1; }
        return cmd_scan(in, out);
    }
    if (!strcmp(argv[1], "forge")) {
        const char *in = NULL, *inlist = NULL, *oc = NULL, *oh = NULL;
        int bits = 8, splitAt = 52, skipV = 1;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--in") && i + 1 < argc) in = argv[++i];
            else if (!strcmp(argv[i], "--in-list") && i + 1 < argc) inlist = argv[++i];
            else if (!strcmp(argv[i], "--out-core") && i + 1 < argc) oc = argv[++i];
            else if (!strcmp(argv[i], "--out-head") && i + 1 < argc) oh = argv[++i];
            else if (!strcmp(argv[i], "--bits") && i + 1 < argc) bits = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--split-at") && i + 1 < argc) splitAt = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--keep-vision") && i + 1 < argc) skipV = !atoi(argv[++i]);
            else if (!strcmp(argv[i], "--keep-vision")) skipV = 0;
        }
        if ((!in && !inlist) || !oc || !oh || (bits != 8 && bits != 3)) { usage(); return 1; }
        if (inlist) {
            static char paths[64][1024]; static const char *pp[64]; int np = 0;
            FILE *lf = fopen(inlist, "r");
            if (!lf) { printf("cannot open list %s\n", inlist); return 1; }
            while (np < 64 && fgets(paths[np], sizeof(paths[0]), lf)) {
                size_t L = strlen(paths[np]);
                while (L && (paths[np][L-1] == '\n' || paths[np][L-1] == '\r' || paths[np][L-1] == ' ')) paths[np][--L] = 0;
                if (!L) continue;
                pp[np] = paths[np]; np++;
            }
            fclose(lf);
            if (!np) { printf("empty list %s\n", inlist); return 1; }
            return cmd_forge_multi(pp, np, oc, oh, bits, splitAt, skipV);
        }
        return cmd_forge(in, oc, oh, bits, splitAt, skipV);
    }
    usage();
    return 1;
}
