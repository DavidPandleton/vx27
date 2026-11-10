# ARCH — VORTEX-27

## 1. Model kunci

`Qwen3.8-27B` dense hybrid, text-only (vision drop):

- 64 block: `16x [3x (Gated DeltaNet+FFN) + 1x (Gated Attention+FFN)]`
- DeltaNet/layer: QK 16x128, V 48x128, conv kernel 4, state/layer 16*128*128=262144 elems
- Full/layer (16x): Q 24x256, KV 4x256 (GQA 6:1), RoPE parsial 64, theta 10M, Mrope [11,11,10]
- FFN SwiGLU 17408, RMSNorm, hidden 5120, vocab 248320 padded, untied embed/lm_head
- MTP 1 hidden layer, ctx 262144 native, YaRN factor 4.0 untuk 1M (kami pakai 64K default / 128K stretch)

Sumber: HF `Qwen/Qwen3.8-27B` README + `unsloth config.json` + vLLM recipes
(`model_type qwen3_5_text`, `full_attention_interval 4`).

## 2. Zona memori TUF A15 (Ryzen 7435HS + RTX3050 4GB + 24GB DDR5 8+16GB)

- Zona 0 cache/register: aktivasi + state DeltaNet aktif. State total 12.58M elems
  (~25MB BF16, ~6MB Q4) — muat di luar register, di L3 + streaming. Bukan nol-stall,
  tapi tidak tumbuh dengan ctx (O(1)).
- Zona 1 dual-channel bawah 16GB: `vx27_core.bin` L0-51. Anggaran 11.2GB.
  Butuh rata-rata <=4.94 bit/param (18.13B params) — realistis dengan campuran 4-bit attn + 3-bit FFN.
- Zona 2 VRAM 3.9GB usable: `vx27_head.bin` L52-63 + lm_head 1.27B + MTP.
  Anggaran 2.8GB untuk 6.84B params perlu <=3.27 bit/param — ketat, butuh lm_head 3-bit
  atau split lm_head CPU/GPU. Ini risiko terbesar, sudah dicetak oleh `vx_forge plan`.
- Zona 3 single-channel atas: KV-cache hanya 16 layer full.
  32768 elems/token. 64K Q4 1.07GB FIT, 128K Q4 2.15GB FIT (budget 2.5GB).
  BF16 tidak muat (64K 4.29GB). Jadi KV Q4/FP8 wajib.

## 3. Reuse dari `ql`

- `threadpool.c`: `GetLogicalProcessorInformation` -> 1 thread/physical core.
  `ql` dan `vx27` sama-sama 8 worker di 7435HS. Jangan hardcode mask `0,2,4...`.
- `quant.c/h`, `quant_avx2.c/h/asm`: layout QK_K=256, Q4_K 144B, Q6_K 210B, Q8_K 256+scales.
  Dot integer `VPMADDUBSW`-style, x8 batch untuk prefill. `vx27` link langsung,
  terbukti oleh `test_st`/`info` (zero-block dot 0.0).
- Tidak reuse penuh: `model.c` 36-layer full-attn
  (butuh DeltaNet + GQA 24/4 hd256), `tokenizer.c` Qwen2-4B (vocab 248320 beda).
  `gguf.c` direuse hanya untuk probe Q4 nyata 4B (27B asli pakai safetensors).

## 4. Modul baru `vx27`

- `vx27_st`: baca u64 LE header_len + scan `"dtype"/"shape"/"data_offsets"`.
  lookup `BF16/F16=2, F32=4, U8/I8=1`. `vx27_st_validate` cek bytes==elems*ds,
  bytes==off1-off0, in-bounds file_data, non-overlap, no-duplikat, total==max_end==file_data.
  Diuji 5 kasus di `test_st` (good + overlap + badsize + trailing + gap).
- `vx27_deltanet`: sizing + `update S=decay*S+outer(k,v)`, `retrieve o=S^T q`.
  Varian tiled 16x16 + avx2 8-wide FMA. `test_deltanet` tiled <1e-5, AVX2 <1e-4, bench 9x.
- `vx27_deltanet exact` (paket fidelity, terverifikasi vs referensi HF `torch_recurrent_gated_delta_rule` + `causal_conv1d_update`):
  q/k l2norm konvensi FLA (eps di DALAM sqrt, tanpa /n) + q/=sqrt(hd); beta=sigmoid(b) TANPA dt_bias; g=-exp(A_log)*softplus(a+dt_bias);
  delta-rule dengan beta-di-k (`k_beta`, `v_beta`) implisit via rumus ekuivalen; conv depthwise k=4 state carryover 3 tap (token pertama pakai tap w[3], sesuai padding kiri F.conv1d — BUKAN tap-0).
  `test_deltanet_exact` PASS: gate mapping bit-identik 0.0; exact-vs-oracle 40x(1+5 langkah) maxdiff out 6e-09 state 6e-08; conv carryover vs naif maxdiff 1.5e-08.
  Divergensi terukur vs kernel lama pada input sama: 0.53 (diharapkan BESAR — inilah hidden-space mismatch yg menjelaskan MTP 0%).
  Status: kernel exact + oracle TERBUKTI dan SUDAH TER-WIRE ke `vx27_qlayer` (state conv per layer 48x10240x3 + beta/gate per V-head + (1+w) norm); re-baseline selesai (checksum -193.433064, MTP rank 38/220).
- `vx27_rope`: partial 64/256 NeoX half-split (seperti `ql`), theta 10M,
  `angle=(pos/yarn)*theta^(-2i/64)`. Mrope text = standard. `test_rope` cek pos0, yarn, passthrough, norm.
- `vx27_gqa`: map `kvh=q*4/24`, rmsnorm, `attn_1q` float ref (scale 1/sqrt256).
  `test_gqa` cek map, rmsnorm [3,4], 1-pos==value, 2-pos==mean.
- `vx27_cfg`: scan flat `"key":value`, derive `rope_dim=head_dim*partial_factor`, `check_locked` 15 field.
  Diuji good/bad di `test_cfg` + **config asli `Qwen/Qwen3.8-27B` PASS** (rope nested `rope_parameters` tetap ketemu scanner).
- `vx27_block`: forward 1-token proxy (H32/F64) full (norm>QKV>QK-norm>RoPE>attn>Wo>res>FFN) dan
  linear (norm>QKV>DeltaNet state>Wo>res>FFN). `matvec_tp` bagi rows via `tp_run` (<64 rows single).
  `test_block` tp vs ref 0.0 + bench + `estimate` real-dim (44.6 GFLOP/token).
- `vx27_ffn`: `hidden=silu(gate)*up`. Ref vs AVX2 (silu skalar + mul 8-wide). `test_ffn` maxdiff 0.0.
- `vx27_qmat`: wrapper Q4_K(144B/blok) x Q8_K AVX2. Ref dequant+dot vs prod Q8+`dot_q4_k_q8_k_avx2`.
  `test_qmat` probe GGUF 4B nyata + sintetik 2 blok (<2%).
- `vx27_weights`: inventory 27B text ~25B, `gb=bits/8/1e9`, `gguf_tensor_info` (header-only).
- `vx27_forward64`: 64 proxy `16x[3lin+1full]` in-place ping-pong, KV 16x + state 48x. `test_forward64` tp==ref + bench.
- `vx27_vxfile`: container v0 magic+version+entries+blob, `write/list` + `plan_14gb` + forge Q8/BP3 nyata + `vx27_mmap` MapViewOfFile. `test_forge` PASS (sampel 24T).
- `vx27_sample_fwd`: 2-layer (L0 lin + L1 full) dari `.vx27` mmap Q8/BP3, dequant + block forward. `test_sample_fwd` Q8 -0.3146 ref==tp.
- `test_sample64_fwd`: 64-layer H64 612T via mmap, split 495/117, checksum -0.0652 ref==tp, ref 2.07ms.
- `vx27_fused`: forward 1 layer Gated DeltaNet fused asli (in_proj_qkv/a/b/z, conv k=4, A_log/dt_bias, delta-rule, z-gate, FFN). Aproksimasi FLA, bukan bit-identik HF.
  `test_fused0` L0 real H5120: checksum 16.186 finite, tp==ref 0.0, ref 264ms tp 39ms.
- `vx27_pmat`: matvec langsung Q8/BP3/F32 x F32 tanpa ekspansi. BP3 in-register
  (`srlv` 8-wide, `blendv` sign, tanpa buffer int8) 6.1x vs skalar, identik (diff ≤3e-08).
  `test_pmat_reg` PASS sintetik + baris nyata 5120-wide.
- `vx27_qfwd`: forward 64-layer penuh H5120 via mmap `.vx27` (packed, tanpa dequant global) + RoPE/GQA + output-gate attention + **qgen multi-token** (KV/DeltaNet/RoPE carried, deterministik).
  `test_qfwd64`: checksum 171.721496 finite, deterministik 0.0, 1.426s/pass (22.29ms/layer), argmax 517 valid; probe mengunci `q_proj` 12288x5120 dan `o_proj` 5120x6144.
  `test_qgen`: 3 token [517 218776 536] deterministik antar-run, ~0.85 tok/s single-stream tanpa MTP.
- `test_real_shard`: shard asli 3.7GB (392T, 333 vision skip) -> core 59 (1.74GB Q8), biggest down_proj [5120,17408] 101MB mmap, stream 22.6GB/s.
- `test_real_layer`: 1 tensor H5120/F17408 (89M) forge Q8 101MB + BP3 36MB via mmap, stream 63GB/s.
- `vx27_gpu_persist`: DEVICE_LOCAL persisten, dispatch reuse (fix reset-flag 38ms->5.5ms). `test_persist` PASS.
- `vx27_bp3`: 3 plane u32 + scale, pack round(), unpack 2-bit sign extend. `test_bp3` SNR 17.4dB, dot 0.0, relerr 3.15%.
- `vx27_batch`: autotune v0/v1-persist/v2 sekali per bentuk + persisten reuse + transpose t-major. `test_batch` best dinamis (0/1) 4-5ms full-y 7e-07.
  Varian Q8 (`matvec_batch_q8`): autotune CPU-8T (pmat) vs GPU-Q8; di RTX 3050 ini CPU menang 2x (18.1 vs 36.2ms) krn buffer host-visible — GPU dicatat sbg fallback teruji, bukan klaim menang.
- `vx27_gpu_full`: runner generik + v0/v1/v2 full-y MATCH + autotune tercepat. `test_gpu_full` PASS.
- `vx27_bp3 v1`: grup 128 + FP16 (50B/128=3.125bit). Head 6.84B->2.67GB FIT 2.8GB. `test_bp3` PASS.
- `vx27_gpu_dl`: staging host-visible -> device-local copy timed. `test_dl` 14MB 3.58ms 4.1GB/s.
- `vx27_gpu`: Vulkan init + Q4_K matvec v0 + tiled v1/v2 (workgroup=1 row, shared reduction, dispatch rows).
  `test_gpu` 45.3Gval/s MATCH; `test_gpu_tiled` v0 vs v1 MATCH.
- `vx27_gpu_bp3`: shader BP3 tiled + ctx device-resident per-matvec (upload sekali, dispatch N kali).
  `test_gpu_bp3` PASS: B=1 4.5ms / B=8 33.0ms (21.6Gval/s), MATCH bit-exact vs CPU; cpu-8T menang (host-visible).
- `vx27_gpu_reg`: registry 126 tensor head BP3 device-resident (upload sekali, dispatch per matvec, hook transparan di `pmat`).
  `test_qfwd_hybrid` (pasca-fidelity): forward 64 penuh dgn layer 52-63 di GPU — checksum -193.4331 vs CPU -193.4328 (relerr 1.3e-06),
  argmax 16 sama, speedup ~1.02x (GPU kalah tipis). Profil jujur: bottleneck = PCIe ping-pong aktivasi
  per matvec (x upload + y readback tiap call), bukan komputasi. Perbaikan berikut = aktivasi residen di
  device antar layer head (satu transfer masuk + satu keluar), bukan per-matvec.
- `vx27_tokbin` + `tokenizer/nfc/unicode_tables` (port utuh mesin BPE `ql`): muat vocab/merges/types Qwen3.8 (248320)
  dari binari `tools/make_tokbin.py`; nol perubahan di jalur encode/decode proven. `test_tok` 14/14 golden vs lib resmi + roundtrip.
- `sampler` (port utuh `ql`, model-independent): greedy/top-k/top-p/temperature + xoshiro. `test_sampler` PASS.
- `vx27_run`: engine persisten (mmap core+head sekali, KV/DeltaNet state + scratch hidden milik engine, `reset` zerokan state) + `run`/`chat` CLI (tokenize template ChatML Qwen3, prefill sekuensial, decode loop + sampler + detoken streaming + stop `im_end`/`endoftext`, statistik tok/s ke stderr). KV stride = `VX27_QGEN_T` (512) — engine menolak `max_t` lebih besar. `test_run_smoke` PASS.
- `vx27_mtp`: draft MTP 1 layer dari bobot asli (fc 10240→5120 + decoder full-attn 24Q/4KV hd256 + FFN + lm_head; q/gate dipecah per-head sesuai `Qwen3NextAttention`/`vLLM qwen3_5_mtp`; concat `[emb, hidden]` sesuai referensi; softmax 1-token = value). `test_mtp` PASS: 15 tensor, finite, deterministik, loop Eagle-style K=2 exact vs greedy; acceptance 0% non-degenerat.
  Diagnosis dataflow SELESAI — status final (bukan tebakan, semua terukur):
  (a) concat order dibetulkan ke `[e,h]`; urutan masih 0%.
  (b) input draft + loop ditulis ulang ke semantik Eagle; tetap 0% (accept tercatat hanya token-0 degenerat).
  (c) matriks offset sistematis 8 sel x 7 posisi: semua 0; varian C/E/F/G + top-10: semua 0.
  (d) peringkat silang penuh anti-korelasi (213rb/183rb vs acak 124rb).
  (e) ablasi: fc-saja rank 246rb; fc tanpa pre_fc norm rank 3 (draft top-1 = main top-2) — pre_fc norm membalik tanda.
  (f) survei bobot norm riil: `pre_fc_*` mean -0.16/-0.45, norm layer 0.04-0.26, `model.norm` 0.94.
  (g) dequant BP3 divalidasi vs BF16 asli (korel 0.97) — unpack BERSALAH tidak terbukti.
  (h) `(1+w)` di engine+MTP: non-finite posisi >=5 (blowup state rekuren).
  RE-BASELINE PASCA-FIDELITY (ter-wire, suite 38/38 hijau, terukur):
  forward 64 checksum -193.433064 (gantikan 171.721496), argmax 16 (gantikan 517), ~1502ms/pass;
  qgen [16 8 220] ~0.78 tok/s; run CLI ", teman" ~0.52 tok/s; hybrid relerr 1.3e-06 speedup 1.02x.
  Kalibrasi MTP: `[e,h]`=3/7 (offset A s=+0 3/7), rank silang 38/220 (pra-fidelity 213rb/183rb) — hidden-space selaras;
  assisted K=2 exact, accept 1/6 (17%), efektif 0.39 vs 0.39 tok/s (verify sekuensial tidak menang, terukur).
  Batch-forward + KV trunk/tail SUDAH terimplementasi (`test_pmat_batch`, `test_mtp_batch`, suite 40/40):
  `vx27_pmat_batch` (1 pass bobot, bit-identik, 2.62x pd 512x5120 B=3), `vx27_qlayer_batch`
  (teacher-forcing lapis-per-lapis, state tunggal sekuensial, KV trunk+tail rantai, ckpt per-token
  utk commit parsial rollback-free), `vx27_engine_batch_step`/`logits_batch`/`commit_ckpt`,
  prefill batch chunked ≤8 TERPASANG di `run_generate` (run/chat, bit-identik, run_smoke tetap [11 174204]).
  Terukur: prefill 7 token 2146ms vs seq ~6560ms (**3.06x**); CLI short 0.52→0.69 tok/s;
  verify batch B=3 == seq exact tetapi 0.91x/token pd accept 17% (batch 1.9x lbh murah dari 3x
  forward, namun baseline seq optimal 1 forward/token via pending; butuh accept >60% utk menang
  generasi — kejujuran terukur, bukan klaim). Multi-cabang tree ditunda (rantai cukup utk K=2).
  Catatan forge: 12 tensor `mtp.layers.0.*` masuk core (bukan head) karena `layer_of()` cocok dengan "layers.0" — fungsional tak masalah (engine gabung core+head), hanya akuntansi zona.
- `test_refusal`: matcher ID+EN 8/8 + probe observasional read-only (tanpa ubah bobot): template-chat greedy kolaps `!!!!` utk semua prompt → sinyal refusal tak terukur pada forward aproksimasi saat ini.
- `vx27_qlayer` full-attention kini memakai seluruh `q_proj` [12288,5120]: 24 blok head x (256 query + 256 gate), lalu `attn_out *= sigmoid(gate)` sebelum `o_proj`, sesuai `Qwen3NextAttention`. `test_qfwd64` mengunci bentuk `q_proj`/`o_proj` pada bobot asli.
- Jalur DeltaNet fidelity exact SUDAH TER-WIRE ke `vx27_qlayer` (status: TERBUKTI, suite hijau):
  `vx27_deltanet_step_exact` + `gate_params` + `l2norm` + `conv1d_step` (carryover 3 tap/channel,
  tap-0 = w[3] sesuai padding kiri F.conv1d) + state per V-head 48x128x128 (bukan per grup QK) +
  QK-norm `(1+w)` + l2norm kernel + q/=sqrt(hd) + GatedRMSNorm output `(xw, init ones)` +
  `beta=sigmoid(b)` / `g=-exp(A_log)*softplus(a+dt_bias)` per V-head.
  State conv milik engine (`Call`, 48x10240x3) di-zero pada `reset`, dibawa antar token.
  Aturan yang dipakai: SEMUA LIMA sekaligus + re-baseline penuh (tidak parsial) — hasilnya
  checksum -193.433064 deterministik, finite, MTP rank silang 38/220.
  Verify paralel (batch-forward) SUDAH terimplementasi + terukur (`test_mtp_batch` PASS):
  1 batch B=3 = 1.9x lebih murah dari 3x forward sekuensial (bobot dibaca 1x), hasil bit-identik.
  Namun pada accept 17% (K=2) efektif 0.91x/token vs baseline seq — baseline seq optimal
  (1 forward/token via pending) tetap menang; ambang menang ≈ accept >60%.
  Parallel semu (engine clone per draft) tetap ditolak: tidak menghemat FLOP.
  KV tree = trunk prefix dibagi + tail draft rantai + commit ckpt parsial (rollback-free);
  multi-cabang ditunda. Prefill batch = kemenangan nyata (3.06x, terpasang di run/chat).
  MTP-KV kausal + rollout EAGLE berantai (`test_mtp_kv` PASS): q/k_norm + RoPE +
  softmax kausal atas KV MTP sendiri (gantikan aproksimasi single-token softmax=1),
  chaining feat draft (d1 dari feat d0, bukan hidden utama), sapuan order×gate
  (terbaik [e,h]+gate 4/7; varian lain 0-1/7 — konfirmasi dataflow, bukan tebakan).
  Hasil: agreement 4/7 (lama 3/7), loop K=2 accept 2/6=33% exact, loop K=1
  (batch B=2, ambang menang p>0.2) accept 1/3 exact tetapi 0.83x/token —
  V terukur ~1.5F + draft ~0.3F belum mengalahkan greedy 1.0F/token pada p~0.33.
  Atensi: rank silang KV (44461) < single-token (38) — detail atensi MTP (RoPE?/gate?)
  vs referensi belum cocok; single-token (tanpa q/k) secara peringkat lebih selaras.
  Aturan main tetap: semua loop exact vs greedy (bit-identik), angka accept = pengukuran.
- `vx27_head`: lm_head sizing `248320*5120=1.27B`, `gb=bits/8/1e9`, shard `rows=vocab*budget/total`.
  matvec float ref (double akumulasi) vs AVX2 FMA 8-wide. MTP `h1=rmsnorm(h+ffn(h)); draft=h1@Wlm`.
  Diuji di `test_head` (matvec ~7e-09, mtp ~3e-08).
- `vx27_kv`: `elems=32768*ctx`, `gb=elems*bpe/1e9`, `fits=gb<=2.5`. Diuji di `test_kv`.
- `vx_forge plan`: inventory param dari dims (bukan dari file), lalu hitung
  `bits_need = budget*8/params`. Output JSON manifest untuk reproducibility.
- `vx_forge scan`: `vx27_st_open` + total GB + 10 tensor pertama. Untuk cek Heretic
  asli nanti tanpa load 50GB ke RAM (hanya baca header).

## 6. Harness (arah: Claude Code / OpenCode / Hermes CLI, tidak kering)

Prinsip: ide UX-nya diambil, implementasinya tetap C zero-dep (tidak bisa
tempel Node/TUI mereka). Fondasi sudah proven di `ql`, tinggal porting.

| Ide dari | Diadopsi ke `vx27` | Catatan |
|---|---|---|
| Claude Code: slash command, `/compact`, session resume, permission (`--yolo` vs approval), todo list, plan mode, streaming thinking | ya, bertahap | `ql` sudah punya `/context /clear /exit`, session file, approval, compaction, delegate; tambah `/compact`, `/todo`, plan mode |
| OpenCode: session + agents + tool (read/write/edit/grep/shell), sandbox workspace, LSP-ish goto (grep), redaksi secret | ya (subset) | `ql` tools 21/21 + sandbox + redaksi 8/8 dipakai ulang; LSP/server tidak (di luar scope monolit) |
| Hermes CLI: output rapi (markdown/streaming), multiline input, history, model switch | ya (CLI polish) | streaming detokenizer + histori sudah ada di `ql`; tambah render markdown ringan + multiline |
| Theme & looks (khusus diminta): ANSI colors, markdown render, syntax highlight kode, thinking spinner, status bar | ya | semua zero-dep di C, masuk polish harness v1 |
| Tidak diambil | TUI penuh, MCP server, multi-model router, theme, plugin JS | butuh runtime/dependensi, melawan target biner statis kecil |

Urutan: (1) forward 27B hidup dulu — harness tanpa mesin = setir tanpa mobil;
(2) port tokenizer 248320 + sampler dari `ql` (180+21 test);
(3) `vx27 chat/run` minimal + thinking display + approval;
(4) tools + session + compaction (reuse `ql`).

## 7. Verifikasi

`mingw32-make test` = 44 langkah berurutan, semua harus PASS sebelum klaim apapun.
Roofline decode (`test_breakdown` PASS, diukur per-komponen bobot real 8T):
gate 2.75ms/12.65GB/s, down 2.53ms/13.74GB/s, qkv 1.49ms/13.78GB/s,
out 0.89ms/13.83GB/s, q 2.26ms/10.86GB/s, o 1.12ms/10.96GB/s,
lm_head 36.01ms/13.79GB/s. Prediksi jumlah matvec = 761ms/9.9GB per forward
(MLP 68%) vs terukur 930-1500ms — MEMORY-BOUND terbukti (12-14 GB/s efektif),
glue (norm/DeltaNet/atensi/swiglu/malloc/churn pool) ≤~20%.
Langit-langit fisik box ini: 9.9GB ÷ 13GB/s ≈ 761ms/token ≈ 1.3 tok/s (tanpa glue);
greedy kini 0.78-1.07 tok/s (72-82% dari langit). Target 5 TPS butuh ~2.2GB/token
(~0.7 bit/param): TIDAK reachable untuk 27B di box ini (butuh model ≤8B, kuant
≤2-bit + revalidasi penuh, atau backend jauh lebih cepat — GPU terukur kalah:
36ms vs CPU-8T 18ms per matvec gate).
KEPUTUSAN SCOPE (final, berbasis ukur — bukan menyerah tanpa data):
(1) greedy ~1 tok/s = batas box ini; (2) prefill batch 3.06x TERPASANG (TTFT turun,
CLI 0.52→0.69 tok/s); (3) arena scratch DITOLAK karena ROI ≤5% (glue total ≤170ms,
malloc hanya fraksi kecilnya; risiko churn signature tanpa imbalan); (4) spekulasi
diparkir sampai accept >60% (butuh drafter terlatih / bedah bobot — di luar scope
runtime C ini); (5) greedy pending-trick (1 forward/token) dipertahankan sebagai
batas atas; semua loop spekulatif tetap exact-vs-greedy bila diaktifkan lagi.
Angka di README adalah output nyata, bukan proyeksi: KV 1.07/2.15GB, state 25MB,
core 4.94 bit, head 3.27 bit, deltanet AVX2 9x/37GFLOPS, rope/gqa/cfg PASS,
ffn maxdiff 0.0, lmhead 0.636GB/0.477GB, matvec 7e-09, mtp 3e-08,
block tp==ref 0.0, proxy 0.058/0.061ms, real 44.6GFLOP/token,
qmat probe 14MB type12 + diff 0.0, forward64 tp==ref + 6.6ms/pass proxy,
gpu v0 MATCH 45.6Gval/s, vxfile plan Q4core+Q3head FIT, tiled MATCH,
bp3 v0+v1 (3.125bit) PASS, dl 9GB/s, batch full-y 7e-07, gpu_full v0/v1/v2 MATCH + autotune,
forge sampel 24T/64T + bobot H5120 sintetik + shard asli 3.7GB (vision skip) + mmap ok,
persist 5.5ms/dispatch MATCH, sample64 fwd64 -0.0652 ref==tp,
real-dim Q8 101MB BP3 36MB stream 63GB/s, shard 59T 1.74GB stream 22.6GB/s,
fused L0 real H5120 checksum 16.186 tp==ref, ref 264ms tp 39ms.
Implikasi jujur: jalur float-dequant ~40ms/layer (TP) => butuh matvec kuant langsung
(Q8/BP3 x Q8 tanpa ekspansi FP32) untuk capai target bandwidth; float path = referensi kebenaran.
Update: matvec kuant langsung SUDAH jalan (pmat AVX2, Q8/BP3 x F32, tanpa ekspansi) —
test_qfwd64: FULL 64-layer H5120 bobot asli via mmap, checksum 171.721496 finite,
deterministik 0.0, 1.426s/pass (22.29ms/layer),
argmax 517 valid. Bottleneck kini compute-bound unpack+konversi (~80 GFLOP/token);
jalan keluar = matvec_batch GPU persisten utk FFN raksasa dan/atau MTP micro-batch.
Biner `vx27.exe` 46KB (fondasi; akan tumbuh dengan kernel).
