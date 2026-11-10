#ifndef VX27_GPU_DL_H
#define VX27_GPU_DL_H
#include <stddef.h>
// Staging -> DEVICE_LOCAL upload probe (Tahap 2 PLAN).
// Copies `bytes` through a staging buffer; returns bandwidth.
int vx27_gpu_dl_upload(size_t bytes, double *ms, double *bw_gbs, char *err, size_t errlen);
#endif
