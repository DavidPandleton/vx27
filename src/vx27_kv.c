#include "vx27_kv.h"
#include "vx27_config.h"

uint64_t vx27_kv_elems(int ctx) {
    return (uint64_t)VX27_KV_ELEMS_PER_TOK * (uint64_t)ctx;
}
double vx27_kv_gb(int ctx, double bytes_per_elem) {
    return (double)vx27_kv_elems(ctx) * bytes_per_elem / 1e9;
}
int vx27_kv_fits(int ctx, double bytes_per_elem, double budget_gb) {
    return vx27_kv_gb(ctx, bytes_per_elem) <= budget_gb;
}
