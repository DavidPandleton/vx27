#ifndef VX27_GPU_REG_H
#define VX27_GPU_REG_H
#include <stddef.h>
#include <stdint.h>
// Registry matvec GPU device-resident: upload sekali semua tensor BP3 head,
// dispatch per matriks tanpa re-upload. pmat() memakainya transparan bila
// vx27_gpu_reg_on() dipanggil; default MATI (perilaku lama tidak berubah).
typedef struct vx27_greg vx27_greg;
int vx27_gpu_reg_init(const void **blobs, const int *rows, const int *cols, int n, char *err, size_t errlen);
void vx27_gpu_reg_on(void);
void vx27_gpu_reg_off(void);
// 1=di-handle GPU, 0=tidak terdaftar (pemanggil fallback CPU), <0 error (fallback CPU).
int vx27_gpu_reg_matvec(const void *W, int rows, int cols, const float *x, float *y, double *ms);
void vx27_gpu_reg_free(void);
#endif
