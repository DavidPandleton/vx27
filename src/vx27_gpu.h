#ifndef VX27_GPU_H
#define VX27_GPU_H
#include <stddef.h>

// Vulkan GPU backend for vx27 (RTX 3050 4GB).
// Reuses ql's proven Q4_K matvec shader (one thread per row*token).
// Probe PASS on this laptop; matvec MATCH required before claiming speedup.

int vx27_gpu_device_count(void); // >=1 on TUF A15
// Bench Q4_K (rows x cols, cols%256==0) B tokens:
// W = Q4_K rows*nblocks, xfloat = rows input per token batch.
// Returns 0 on MATCH, fills gpu_ms, cpu_ms, gvals. Negative if GPU unavailable.
// y_rowmajor_out nullable: if given, full GPU y (rows*B, row*B+t) is copied out.
int vx27_gpu_q4_bench(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B,
                      double *gpu_ms, double *cpu_ms, double *gvals,
                      float *y_gpu0, float *y_cpu0, char *err, size_t errlen);
int vx27_gpu_q4_bench_full(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen);
// Tiled v1: workgroup(64)=1 row, shared reduction. Same MATCH contract.
int vx27_gpu_q4_bench_tiled(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B,
                      double *gpu_ms, double *cpu_ms, double *gvals,
                      float *y_gpu0, float *y_cpu0, char *err, size_t errlen);
int vx27_gpu_q4_bench_tiled_full(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen);
int vx27_gpu_q4_bench_tiled8_full(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen);
// Q8_K x Q8_K (format .vx27 Q8, mis. gate_proj Q8 dari forge bits=8).
// W = Q8_K rows*nblocks, xfloat di-quantize ke Q8_K per token.
int vx27_gpu_q8_bench_tiled_full(const void *Wq8, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen);

#endif
