#ifndef VX27_CFG_H
#define VX27_CFG_H
#include <stddef.h>

// Minimal config.json loader for model_type qwen3_5_text (Qwen3.8-27B).
// Scans flat "key": value ints/floats/arrays without a JSON lib and validates
// against locked VX27_* constants. text_config nesting is flattened by scanning
// the whole file, so both top-level and nested keys work.

typedef struct {
    int hidden, layers, n_q, n_kv, head_dim, rope_dim, ffn, vocab, max_pos;
    int lin_qk, lin_v, lin_hd, conv_dim, full_interval, mtp_layers;
    float rope_theta, yarn_factor;
    int mrope_sec[3];
} vx27_cfg;

int vx27_cfg_load(const char *path, vx27_cfg *c, char *err, size_t errlen);
int vx27_cfg_check_locked(const vx27_cfg *c, char *err, size_t errlen);

#endif
