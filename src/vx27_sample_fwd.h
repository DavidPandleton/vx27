#ifndef VX27_SAMPLE_FWD_H
#define VX27_SAMPLE_FWD_H
#include <stddef.h>
// Forward 2-layer dari .vx27 sampel via mmap (bobot Q8/BP3 nyata).
// Forward 2-layer dari .vx27 sampel via mmap (bobot Q8/BP3 nyata).
// Sample: H=64 F=128, L0 linear (lin_qk=2 lin_v=2 lin_hd=8), L1 full (nq=4 nkv=2 hd=8 rope=8).
// Bobot di-dequant dari mmap blob ke float sekali (publish sebagai referensi),
// lalu block forward ref vs threadpool. Return checksum.
float vx27_sample_fwd(const char *core_vx27, const char *head_vx27,
                      int use_tp, double *ms, char *err, size_t errlen);
#endif
