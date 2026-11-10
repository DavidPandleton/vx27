#ifndef VX27_BP3_H
#define VX27_BP3_H
#include <stdint.h>

// Prototipe bit-planar 3-bit v0 (sintetik, bukan format final .vx27 v1).
// 32 bobot per grup: level int -4..3 (uniform, scale=max/3.5).
// Disimpan sebagai 3 plane u32 (bit0,bit1,bit2) + float scale.
// Unpack via LUT nibble-free (scalar ref) vs AVX2 plane gather (prod).

typedef struct { uint32_t p0, p1, p2; float scale; } vx27_bp3_grp; // 32 w

void vx27_bp3_pack(const float *w32, vx27_bp3_grp *g);
void vx27_bp3_unpack_ref(const vx27_bp3_grp *g, float *w32);
float vx27_bp3_dot_ref(const vx27_bp3_grp *G, const float *x, int ngroups);
float vx27_bp3_dot_avx2(const vx27_bp3_grp *G, const float *x, int ngroups);

// v1 final: grup 128 + skala FP16 -> 3*128 bit + 16 bit = 400 bit = 3.125 bit/param.
// Cocok untuk head 6.84B -> 2.67GB FIT 2.8GB (lihat plan14).
typedef struct { uint32_t p[12]; uint16_t scale_h; } vx27_bp3_128; // 50 B
void vx27_bp3_128_pack(const float *w128, vx27_bp3_128 *g);
float vx27_bp3_128_dot_ref(const vx27_bp3_128 *G, const float *x, int ng128);
float vx27_bp3_128_dot_avx2(const vx27_bp3_128 *G, const float *x, int ng128);

#endif
