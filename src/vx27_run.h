#ifndef VX27_RUN_H
#define VX27_RUN_H
#include <stddef.h>
#include <stdint.h>
#include "vx27_qfwd.h"
#include "threadpool.h"

// Mesin inferensi persisten: mmap core+head sekali, KV + DeltaNet state hidup
// lintas token, threadpool 8. Dipakai `vx27 run` / `vx27 chat`.
typedef struct vx27_engine vx27_engine;

// max_t = jumlah token maksimum total (prompt + generate), 1..VX27_QGEN_T.
int vx27_engine_open(vx27_engine **e, const char *core, const char *head,
                     int max_t, char *err, size_t errlen);
void vx27_engine_reset(vx27_engine *e);
void vx27_engine_close(vx27_engine *e);
// Embed baris id -> out[5120]. 0 = ok.
int vx27_engine_embed(vx27_engine *e, uint32_t id, float *out, char *err, size_t errlen);
// 1 langkah forward penuh (64 layer) utk token di posisi n_pos. 0 = ok.
int vx27_engine_step(vx27_engine *e, const float *x, float *out, int n_pos, char *err, size_t errlen);
// Logits + argmax.
uint32_t vx27_engine_logits(vx27_engine *e, const float *h, float *logits, char *err, size_t errlen);
// Batch teacher-forced 64-layer: xs[B][5120] (posisi n_pos0+b) -> hs[B][5120].
// State milik engine dimajukan in-place; Sckpt [B][48][786432] /
// Cckpt [B][48][30720] opsional (NULL=tidak) utk commit parsial spekulatif.
// KV: trunk prefix dibagi + tail draft berurutan. 1<=B<=8.
int vx27_engine_batch_step(vx27_engine *e, const float *xs, float *hs, int B, int n_pos0,
    float *Sckpt, float *Cckpt, char *err, size_t errlen);
// Batch logits: hs[B][5120] -> lgs[B][248320], args[B].
void vx27_engine_logits_batch(vx27_engine *e, const float *hs, float *lgs,
    uint32_t *args, int B, char *err, size_t errlen);
// Commit parsial: salin ckpt token-ke-m ke state engine (tanpa rollback).
void vx27_engine_commit_ckpt(vx27_engine *e, const float *Sckpt, const float *Cckpt, int m);
// Akses internal utk MTP/uji di atas engine yang sama (qtab gabungan core+head + tp).
vx27_qtab *vx27_engine_qtab(vx27_engine *e);
threadpool *vx27_engine_tp(vx27_engine *e);
// Entry point CLI: vx27 run "teks" [opsi] / vx27 chat. Mengembalikan exit code.
int vx27_run_main(int argc, char **argv);
int vx27_chat_main(int argc, char **argv);

#endif
