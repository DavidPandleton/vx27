#include "vx27_cfg.h"
#include "vx27_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// find "key" then ':' then number (int or float)
static int find_num(const char *js, const char *key, double *out) {
    char pat[128]; snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) return 0;
    p = strchr(p + strlen(pat), ':');
    if (!p) return 0;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    char *e = NULL;
    double v = strtod(p, &e);
    if (e == p) return 0;
    *out = v;
    return 1;
}
static int find_int(const char *js, const char *key, int *out, int def) {
    double v = 0;
    if (!find_num(js, key, &v)) { *out = def; return 0; }
    *out = (int)v;
    return 1;
}

int vx27_cfg_load(const char *path, vx27_cfg *c, char *err, size_t errlen) {
    memset(c, 0, sizeof(*c));
    FILE *f = fopen(path, "rb");
    if (!f) { if (err) snprintf(err, errlen, "cannot open %s", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (1 << 20)) { fclose(f); if (err) snprintf(err, errlen, "bad size"); return -1; }
    char *js = (char *)malloc((size_t)sz + 1);
    if (!js) { fclose(f); return -1; }
    if (fread(js, 1, (size_t)sz, f) != (size_t)sz) { free(js); fclose(f); return -1; }
    js[sz] = 0; fclose(f);

    double v = 0;
    find_int(js, "hidden_size", &c->hidden, 0);
    if (!c->hidden) find_int(js, "hidden", &c->hidden, 0);
    find_int(js, "num_hidden_layers", &c->layers, 0);
    find_int(js, "num_attention_heads", &c->n_q, 0);
    find_int(js, "num_key_value_heads", &c->n_kv, 0);
    // head dims: prefer explicit, else derive
    find_int(js, "head_dim", &c->head_dim, 0);
    if (!c->head_dim) find_int(js, "attention_head_dim", &c->head_dim, 0);
    if (!c->head_dim && c->hidden && c->n_q) c->head_dim = c->hidden / c->n_q; // fallback (won't match 256, caught by check)
    find_int(js, "intermediate_size", &c->ffn, 0);
    if (!c->ffn) find_int(js, "ffn", &c->ffn, 0);
    find_int(js, "vocab_size", &c->vocab, 0);
    if (!c->vocab) find_int(js, "vocab", &c->vocab, 0);
    find_int(js, "max_position_embeddings", &c->max_pos, 0);
    if (find_num(js, "rope_theta", &v)) c->rope_theta = (float)v;
    if (find_num(js, "rope_theta_base", &v)) c->rope_theta = (float)v;
    // rope dim from partial factor or explicit
    if (find_num(js, "partial_rotary_factor", &v) && c->head_dim)
        c->rope_dim = (int)(c->head_dim * v + 0.5);
    find_int(js, "rotary_dim", &c->rope_dim, c->rope_dim);
    find_int(js, "rope_dim", &c->rope_dim, c->rope_dim);
    // linear attention
    find_int(js, "linear_num_key_heads", &c->lin_qk, 0);
    find_int(js, "linear_num_value_heads", &c->lin_v, 0);
    find_int(js, "linear_key_head_dim", &c->lin_hd, c->lin_hd);
    if (!c->lin_hd) find_int(js, "linear_value_head_dim", &c->lin_hd, 0);
    find_int(js, "linear_conv_kernel_dim", &c->conv_dim, 0);
    find_int(js, "full_attention_interval", &c->full_interval, 0);
    find_int(js, "mtp_num_hidden_layers", &c->mtp_layers, 0);
    // yarn factor (default 1)
    c->yarn_factor = 1.0f;
    if (find_num(js, "yarn_factor", &v)) c->yarn_factor = (float)v;
    else if (find_num(js, "factor", &v) && strstr(js, "yarn")) c->yarn_factor = (float)v;
    // mrope_section [a,b,c]
    const char *ms = strstr(js, "mrope_section");
    if (ms) {
        const char *br = strchr(ms, '[');
        if (br) { int a = 0, b = 0, cc = 0;
            if (sscanf(br, "[%d,%d,%d", &a, &b, &cc) == 3) { c->mrope_sec[0] = a; c->mrope_sec[1] = b; c->mrope_sec[2] = cc; } }
    }
    free(js);
    return 0;
}

int vx27_cfg_check_locked(const vx27_cfg *c, char *err, size_t errlen) {
#define CHK(f, want) do { if ((c->f) != (want)) { if (err) snprintf(err, errlen, #f " %d != locked %d", (int)(c->f), (int)(want)); return -1; } } while (0)
    CHK(hidden, VX27_HIDDEN);
    CHK(layers, VX27_LAYERS);
    CHK(n_q, VX27_ATTN_Q_HEADS);
    CHK(n_kv, VX27_ATTN_KV_HEADS);
    CHK(head_dim, VX27_ATTN_HEAD_DIM);
    CHK(rope_dim, VX27_ROPE_DIM);
    CHK(ffn, VX27_FFN);
    CHK(vocab, VX27_VOCAB);
    CHK(lin_qk, VX27_LIN_QK_HEADS);
    CHK(lin_v, VX27_LIN_V_HEADS);
    CHK(lin_hd, VX27_LIN_HEAD_DIM);
    CHK(conv_dim, VX27_LIN_CONV_DIM);
    CHK(full_interval, VX27_FULL_INTERVAL);
    CHK(mtp_layers, VX27_MTP_LAYERS);
    if (c->rope_theta != VX27_ROPE_THETA) { if (err) snprintf(err, errlen, "rope_theta %.0f != %.0f", (double)c->rope_theta, (double)VX27_ROPE_THETA); return -1; }
    if (c->max_pos != VX27_CTX_NATIVE) { if (err) snprintf(err, errlen, "max_pos %d != %d", c->max_pos, VX27_CTX_NATIVE); return -1; }
    if (!(c->mrope_sec[0] == 11 && c->mrope_sec[1] == 11 && c->mrope_sec[2] == 10)) { if (err) snprintf(err, errlen, "mrope_section != [11,11,10]"); return -1; }
    return 0;
#undef CHK
}
