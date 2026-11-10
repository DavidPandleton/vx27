#include "vx27_weights.h"
#include "vx27_config.h"
#include "gguf.h"
#include <stdio.h>

uint64_t vx27_27b_params_text(void) {
    double tok = (double)VX27_VOCAB * VX27_HIDDEN;
    double full = 5120.0*6144 + 5120.0*1024*2 + 6144.0*5120 + 3.0*5120.0*17408;
    double lin = 5120.0*2048*2 + 5120.0*6144 + 6144.0*5120 + 3.0*5120.0*17408;
    double bb = 16*full + 48*lin;
    double mtp = 5120.0*17408 + 5120.0*5120;
    return (uint64_t)(bb + 2*tok + mtp);
}
double vx27_27b_gb(double bits) { return (double)vx27_27b_params_text() * bits / 8.0 / 1e9; }

int vx27_gguf_tensor_info(const char *path, const char *tensor,
                           uint64_t *n_bytes, uint32_t *type,
                           char *err, size_t errlen) {
    gguf_file g;
    if (gguf_open(path, &g, err, errlen) != GGUF_OK) return -1;
    const gguf_tensor *t = gguf_find_tensor(&g, tensor);
    if (!t) {
        if (err) snprintf(err, errlen, "missing tensor %s", tensor);
        gguf_close(&g); return -1;
    }
    if (n_bytes) *n_bytes = t->n_bytes;
    if (type) *type = t->type;
    gguf_close(&g);
    return 0;
}
