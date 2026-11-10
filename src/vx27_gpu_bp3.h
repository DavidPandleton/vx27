#ifndef VX27_GPU_BP3_H
#define VX27_GPU_BP3_H
#include <stddef.h>
// Device-resident BP3 matvec: upload bobot (.vx27 BP3) sekali ke VRAM,
// dispatch berkali-kali. X float (host-visible), Y float.
// Cocok untuk matriks yg dipakai tiap token (FFN, lm_head, MTP).
typedef struct vx27_bctx vx27_bctx;
int vx27_bctx_init(vx27_bctx **out, const void *Wbp3, int rows, int cols, char *err, size_t errlen);
int vx27_bctx_dispatch(vx27_bctx *c, const float *xB, int B, float *y_rowmajor, double *ms, char *err, size_t errlen);
void vx27_bctx_free(vx27_bctx *c);
#endif
