#ifndef VX27_WEIGHTS_H
#define VX27_WEIGHTS_H
#include <stdint.h>

// 27B text-only inventory (from locked dims) + GGUF probe reuse.
// No 54GB download needed: probe reads only GGUF header + one tensor.

uint64_t vx27_27b_params_text(void);   // ~24.97B est (backbone+emb/mtp, no vision)
double vx27_27b_gb(double bits);       // params*bits/8/1e9
// Open GGUF (ql 4B file) and report tensor bytes/type for Q4 integration proof.
int vx27_gguf_tensor_info(const char *path, const char *tensor,
                           uint64_t *n_bytes, uint32_t *type,
                           char *err, size_t errlen);

#endif
