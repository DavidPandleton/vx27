#ifndef VX27_GPU_PERSIST_H
#define VX27_GPU_PERSIST_H
#include <stddef.h>
#include <stdint.h>
// Bobot Q4 device-resident persisten: upload sekali (staging->DEVICE_LOCAL),
// dispatch berkali-kali tanpa recreate instance/device.
// Terbatas untuk satu bentuk (rows,cols,B) per ctx, shader tiled v1.
typedef struct vx27_gpu_ctx vx27_gpu_ctx;
int vx27_gpu_ctx_init(vx27_gpu_ctx **ctx, const void *Wq4, int rows, int cols, char *err, size_t errlen);
int vx27_gpu_ctx_dispatch(vx27_gpu_ctx *ctx, const float *xB, float *y_rowmajor, double *ms, char *err, size_t errlen);
void vx27_gpu_ctx_free(vx27_gpu_ctx *ctx);
#endif
