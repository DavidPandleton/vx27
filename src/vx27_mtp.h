#ifndef VX27_MTP_H
#define VX27_MTP_H
#include <stddef.h>
#include <stdint.h>
#include "threadpool.h"
#include "vx27_qfwd.h"

// MTP (Multi-Token Prediction) 1 layer, bobot ASLI dari .vx27 hasil forge
// (15 tensor `mtp.*`: fc + 1 decoder layer full-attention + 3 pre/final norm).
//
// Dataflow (didokumentasikan, mengikuti pola Qwen3_5Attention + konvensi
// penamaan tensor; urutan concat fc dikalibrasi di test_mtp):
//   e_n = rmsnorm(emb, pre_fc_norm_embedding)
//   h_n = rmsnorm(h_prev, pre_fc_norm_hidden)
//   x   = fc @ concat([e_n, h_n])          // urutan referensi vLLM (order_he=0);
//                                          // order_he=1 ([h,e]) hanya varian kalibrasi
//   y   = x + attn(rmsnorm(x, input_layernorm))   // full-attn 24Q/4KV hd256 + q/gate,
//                                                // mode single-token (tanpa riwayat KV)
//   y   = x + attn(rmsnorm(x, input_layernorm))   // full-attn 24Q/4KV hd256 + q/gate,
//                                                // mode single-token (tanpa riwayat KV)
//   z   = y + mlp(rmsnorm(y, post_attention_layernorm))
//   o   = rmsnorm(z, mtp.norm);  draft_logits = o @ lm_head
//
// Mode single-token: dengan 1 token di KV, softmax = 1.0 sehingga output
// attention = value (q/k/RoPE tidak berpengaruh). Yang dipakai: v_proj,
// gate (paruh q_proj), o_proj. Ini aproksimasi yang didokumentasikan;
// loop verify menjamin output akhir IDENTIK dengan greedy baseline.

// Cek 15 tensor mtp.* ada + bentuk sesuai. Return jumlah ketemu, <0 bila fatal.
int vx27_mtp_probe(const vx27_qtab *tab, char *err, size_t errlen);

// Satu draft MTP -> draft_logits[248320]. 0 = ok.
// use_gate=1: attention *= sigmoid(gate) (sesuai referensi); =0: tanpa gate.
// ablate bitmask diagnostik: 1=lewati blok attention+MLP (o = norm(fc_out));
// 2=lewati SEMUA norm (identitas); 4=lewati final norm saja;
// 8=lewati pre_fc norm saja. Kombinasi dg OR.
// Norm memakai x*w (baseline stabil); varian ablate dipakai memetakan
// sensitivitas terhadap konvensi norm (temuan: fc tanpa pre_fc norm
// menempatkan main-top1 di rank 3 vs rank 200rb+ dengan norm).
int vx27_mtp_draft(const vx27_qtab *tab, threadpool *tp,
                   const float *h_prev, const float *emb,
                   float *draft_logits, int order_he, int use_gate, int ablate,
                   char *err, size_t errlen);

// Langkah decoder MTP dgn KV-cache kausal (fidelity vLLM/EAGLE):
// attention penuh 24Q/4KV hd256 + q_norm/k_norm (1+w) + RoPE rd=64 +
// softmax kausal atas 0..pos (BUKAN aproksimasi single-token).
// kvc: [2][T][1024] float (k lalu v), T=VX27_QGEN_T; append k/v pd pos.
// feat_out[5120] (boleh NULL): fitur o utk chaining draft berikut ala EAGLE
//   (draft k+1 memakai feat draft k sebagai h_prev, bukan hidden utama).
// logits_out[248320] (boleh NULL): draft logits.
// 0 = ok. Tail di luar panjang logis ditimpa (posisi eksplisit, tanpa rollback).
int vx27_mtp_step_kv(const vx27_qtab *tab, threadpool *tp,
                   const float *h_prev, const float *emb, float *kvc, int pos, int T,
                   float *feat_out, float *logits_out, int order_he, int use_gate,
                   char *err, size_t errlen);
#endif
