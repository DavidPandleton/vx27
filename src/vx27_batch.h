#ifndef VX27_BATCH_H
#define VX27_BATCH_H
#include "threadpool.h"

// matvec_batch GPU device-resident penuh (y GPU transpose) + fallback CPU.
// Autotune v0/v1-persist/v2 sekali per bentuk, cache tercepat.
int vx27_matvec_batch(int gpu_ok, threadpool *tp,
    const void *Wq4, const float *xB, float *yBx,
    int rows, int cols, int B, double *ms);
int vx27_batch_best(void); // 0=v0 1=persist-tiled 2=v2, -1=CPU/belum tune

// Packed Q8 batch (format .vx27, dtype=2): autotune CPU-8T vs GPU-Q8 sekali
// per bentuk, cache pemenang. Return 1=gpu, 0=cpu. *ms = waktu pemenang.
int vx27_matvec_batch_q8(threadpool *tp,
    const void *Wq8, const float *xB, float *yBx,
    int rows, int cols, int B, double *ms, char *err, size_t errlen);
int vx27_batch_q8_best(void); // 1=gpu 0=cpu -1=belum tune

#endif
