#ifndef VX27_QFWD_H
#define VX27_QFWD_H
#include "threadpool.h"
#include <stdint.h>

// Forward 64-layer Qwen3.8 (H5120) dengan bobot TETAP ter-packing (mmap langsung).
// dtype: 0=F32, 2=Q8_K, 4=BP3_128. Aktivasi float. Tanpa ekspansi FP32.
// Linear path mengikuti referensi gated delta rule (l2norm q/k, q/=sqrt(hd),
// beta/gate, conv carryover, gated RMSNorm output). Single token.

typedef struct {
    const void *blob; uint32_t dtype;
    int rows, cols; // ne0, ne1 dari .vx27
} vx27_qt;

typedef struct {
    int n; // jumlah tensor
    const char **name;
    vx27_qt *t;
} vx27_qtab;

const vx27_qt *vx27_qfind(const vx27_qtab *tab, const char *name);
#define VX27_QGEN_T 512  // maks token per run (KV-cache per slot); cukup utk prompt+generate run/chat
// State DeltaNet per layer linear: 48 V-head x 128 x 128 (per V-head, referensi).
#define VX27_SSTATE_PER_LAYER ((size_t)48*128*128)
// State konvolusi per layer linear: conv_dim(10240) x 3 (3 input terakhir).
#define VX27_CSTATE_PER_LAYER ((size_t)10240*3)

// satu layer penuh (is_full: 1=GQA, 0=Gated DeltaNet fused). n_pos = indeks posisi
// token saat ini (0-based); KV-cache kvc berukuran [16 slot][T][nkv*hd] x2 (k lalu v).
// S = state DeltaNet per layer [48*128*128] (per V-head, referensi), NULL utk full.
// conv = state konvolusi per layer [10240*3] (3 input terakhir), NULL utk full.
void vx27_qlayer(const vx27_qtab *tab, int L, int is_full,
                 const float *x, float *out, int n_pos,
                 const int *full_idx /*16x layer->kvcache slot, -1 bila lin*/,
                 float *S, float *kvc,
                 const float *rope_cos, const float *rope_sin,
                 threadpool *tp, float *conv);
// Teacher-forced batch satu layer: B token (posisi n_pos0+b) diproses
// lapis-per-lapis; proyeksi di-batch via pmat_batch (1 pass bobot).
// State rekuren TUNGGAL per sekuens (S/conv milik caller) dimajukan
// sekuensial antar b (urutan == step sekuensial -> bit-identik).
// KV prefix dibagi; tail n_pos0.. ditulis berurutan (rantai trunk+tail).
// Sckpt/Cckpt [B][state-layer-ini] opsional (NULL=tidak): salinan state
// setelah tiap token b, utk commit parsial verify spekulatif.
void vx27_qlayer_batch(const vx27_qtab *tab, int L, int is_full,
                 const float *xs, float *outs, int B, int n_pos0,
                 const int *full_idx, float *S, float *kvc,
                 threadpool *tp, float *conv,
                 float *Sckpt, float *Cckpt);
// rms akhir + lm_head -> logits[vocab]. Mengembalikan argmax.
int vx27_qlogits(const vx27_qtab *tab, const float *h, float *logits, threadpool *tp);
// Batch: hs[B][H] -> lgs[B][V], args[B]. Logit per posisi teacher-forced.
void vx27_qlogits_batch(const vx27_qtab *tab, const float *hs, float *lgs,
                 uint32_t *args, int B, threadpool *tp);
// Multi-token decode: T langkah, state DeltaNet + KV-cache + RoPE dibawa antar token.
// x0 = embedding token awal; tiap langkah argmax -> embed baris berikutnya.
// out_ids[T]: id token; ms_total: waktu penuh. Deterministik.
void vx27_qgen(const vx27_qtab *tab, threadpool *tp, const float *x0,
               int T, int *out_ids, double *ms_total);

#endif
