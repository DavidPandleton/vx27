# VORTEX-27 (`vx27`) — monolit inference Qwen3.8-27B Heretic text-only

Target terkunci: `Qwen3.8-27B Heretic-Abliterated` safetensors (BF16/F16), text-only.
Vision tower 0.46B dibuang. 64 layer = 48x Gated DeltaNet + 16x Gated Attention (tiap layer ke-4),
hidden 5120, FFN 17408, vocab 248320, MTP 1 layer, ctx native 262144.

Status: skeleton + forge dry-run + kernel fondasi. Belum full forward 27B.

## Struktur

```
vx27/
  src/vx27_config.h   konstanta kunci Qwen3.8-27B (jangan ubah tanpa ganti model)
  src/vx27.c          CLI `vx27.exe info`
  src/vx_forge.c      CLI `vx_forge.exe plan|scan|forge` (split core/head, manifest; multi-shard `--in-list`, skip vision)
  src/vx27_st.*       parser + validator safetensors (header_len, dtype, shape*size==bytes,
                      offsets==bytes, in-bounds, no-overlap, no-gap, no-trailing)
  src/vx27_deltanet.* sizing + update/retrieve ref vs tiled vs AVX2 (hd=128, 9.3x vs ref, ~37 GFLOPS)
  src/vx27_rope.*     partial RoPE 64/256 + Mrope text + YaRN linear (theta 10M)
  src/vx27_gqa.*      GQA 24Q/4KV hd256 + QK RMSNorm + attn 1-query ref
  src/vx27_cfg.*      loader config.json qwen3_5_text + check locked constants
  src/vx27_ffn.*      SwiGLU 17408 ref vs AVX2 (silu scalar + mul 8-wide)
  src/vx27_head.*     lm_head 1.27B sizing/shard + matvec AVX2 + MTP 1-layer ref vs fwd
  src/vx27_block.*    forward blok penuh (DeltaNet/GQA+FFN) ref vs threadpool + bench proxy + estimasi real-dim
  src/vx27_qmat.*     Q4_K x Q8_K AVX2 wrapper (ref dequant vs prod) + probe GGUF nyata
  src/vx27_weights.*  inventory 27B text + gguf tensor info (tanpa unduh 54GB)
  src/vx27_forward64.* forward 64-layer proxy end-to-end (16x[3lin+1full]) + bench
  src/vx27_gpu.*      Vulkan Q4_K matvec v0 + tiled v1 + bench CPU vs GPU (y0)
  src/vx27_gpu_full.* runner SPV generik + _full v0/v1/v2 (y penuh, MATCH+nrmse) + autotune
  src/vx27_gpu_persist.* bobot Q4 device-local persisten (upload sekali, dispatch N kali, fix reset-flag)
  src/vx27_mmap.*     loader mmap Windows (MapViewOfFile) + blob readback
  src/vx27_vxfile.*   container .vx27 v0 (Q8 nyata + F32) + v1 BP3 + plan 14GB
  src/vx27_sample_fwd.* forward 2-layer sampel via mmap Q8/BP3 (ref vs tp)
  tools/make_sample.py sampel 2-layer (24T, 191KB); make_sample64.py 64-layer (612T, 3.8MB)
  tools/gen_real_gate.c 1 tensor H5120/F17408 sintetik (89M BF16 178MB x2, forge Q8 101MB / BP3 36MB)
  tools/gen_real_gate.c 1 tensor real-dim H5120/F17408 (89M, 178MB BF16 x2)
  src/vx27_bp3.*      prototipe bit-planar 3-bit (3 plane + scale, ref vs AVX2)
  src/vx27_batch.*    matvec_batch device-resident penuh (y GPU transpose) + fallback CPU
  src/vx27_gpu_dl.*   staging -> DEVICE_LOCAL upload probe
  gpu/q4k_matvec.comp shader v0 (1 thread per row*token)
  gpu/q4k_matvec_tiled.comp shader v1 (workgroup 64 = 1 row, shared reduction)
  gpu/q4k_matvec_tiled8.comp shader v2 (8-acc unroll, W reuse parsial)
  src/vx27_kv.*       hitungan KV-cache 16 layer full saja
  src/threadpool.*    reuse ql: 8 worker + affinity physical core
  src/quant.*         reuse ql: Q4_K/Q6_K layout + skalar ref
  src/quant_avx2.*    reuse ql: Q8 quant + dot AVX2 + x8 batch + NASM
  tests/              42 suite (tambah test_breakdown) semua PASS
  build/              vx27.exe 46KB, vx_forge.exe, manifest.json
```

Reuse dari `C:\Users\ASUS A15\ql`: threadpool, quant, quant_avx2 terbukti
(`ql` 20/20 bench, 180/180 tokenizer, 600 trial bit-identik). Tidak reuse
`gguf.c/model.c/tokenizer.c` karena Qwen3.8 butuh safetensors + forward hybrid baru.

## Build

Toolchain portable (tanpa admin), sama seperti `ql`:

- `C:/Users/Public/mingw64` (GCC 16.2 UCRT)
- `C:/Users/Public/nasm-*` (NASM 3.01 ok, dipakai `-f win64`)
- `C:/Users/Public/vulkan` (belum dipakai di vx27 tahap ini)

```bash
export PATH="/c/Users/Public/mingw64/bin:$PATH"
cd vx27
mingw32-make test
```

## Test

`mingw32-make test` menjalankan, berurutan:

1. `test_st` — good file 2 tensor 40B; overlap, bytes-vs-offsets, trailing, gap semua ditolak dengan pesan.
2. `test_deltanet` — state/layer 262144, total 48 layer ~25MB BF16;
   tiled vs ref <1e-5; AVX2 update maxdiff 0.0, retrieve ~2.4e-08 (<1e-4);
   bench hd=128 x2000: AVX2 ~3.5ms (1.75us/iter, ~37GFLOPS), ref ~32.5ms (9.3x).
3. `test_deltanet_exact` — paket fidelity DeltaNet terverifikasi vs referensi HF (`torch_recurrent_gated_delta_rule` + `causal_conv1d_update`): gate mapping bit-identik; exact-vs-oracle 40x(1+5 langkah) maxdiff out 6e-09/state 6e-08; conv carryover vs konvolusi naif 1.5e-08; divergensi kernel lama 0.53 (dokumentasi, bukan assert).
4. `test_kv` — 32768 elems/token, 64K Q4 FIT, 128K Q4 FIT, 64K BF16 OVER, 8K BF16 ~0.54GB.
5. `test_rope` — pos0 cos=1/sin=0; yarn pos32/f4==pos8/f1 maxdiff 0; partial passthrough >=64; norm preserved; mrope==standard.
6. `test_gqa` — map q/6==kv (0-5->0 ... 23->3); rmsnorm [3,4]->[0.8485,1.1314]; 1-pos out==value; 2-pos identical keys -> mean.
7. `test_cfg` — good parsed + locked PASS; bad layers=32 ditolak; **config asli `Qwen3.8-27B` (4KB) locked PASS** (hidden/layers/rope/mrope cocok).
8. `test_ffn` — swiglu tiny maxdiff 0.0; 17409 (tail) maxdiff 0.0; silu(0)=0, silu(1)=0.73105.
9. `test_head` — lmhead 1.27B, 4-bit 0.636GB FIT 2.8GB, shard 0.5GB->195312 rows; matvec tiny maxdiff ~7e-09; mtp draft maxdiff ~3e-08 finite.
10. `test_block` — full/linear proxy (H32/F64) tp vs ref maxdiff 0.0 deterministik; bench 200x full 0.058ms/blok, linear 0.061ms/blok; estimasi real-dim 44.6 GFLOP/token, Q4 full-pass full 2.73GB + linear 8.43GB.
11. `test_qmat` — probe GGUF nyata `blk.0.ffn_gate.weight` 14008320B type12; 27B Q4 12.49GB/Q3 9.36GB; Q4 ref vs AVX2 diff 0.0 (<2%).
12. `test_forward64` — 64-layer proxy checksum ref==tp, deterministik; bench 20x 6.64ms/pass proxy; catat 25700x ke real-dim.
13. `test_gpu` — vulkan 1 device (RTX 3050); `blk.0.ffn_gate` 9728x2560 B=8 MATCH; cpu 14.4ms vs gpu 4.37ms 45.5Gval/s.
14. `test_vxfile` — tulis/list `.vx27` Q8+F32; plan14 core Q4 9.06GB FIT + head Q3 2.57GB FIT.
15. `test_gpu_tiled` — v0 vs v1 tiled MATCH; v1 correctness ok, speedup belum konsisten (+8.3% / -25% antar-run).
16. `test_bp3` — v0 SNR 17.42dB 4.0bit diff 0.0 relerr 3.15%; v1 128+FP16 3.125bit diff 1.4e-07 relerr <10%; 0.24us/dot.
17. `test_dl` — staging->DEVICE_LOCAL 14MB 1.6-3.6ms 4.1-9.4GB/s (<50ms).
18. `test_batch` — device-resident penuh: GPU y transpose, full-y maxdiff 7e-07 MATCH; fallback CPU.
19. `test_gpu_full` — v0/v1/v2 full-y MATCH maxdiff 9.5e-07; autotune best dinamis (v0/v1/v2 4.2-5.2ms).
20. `test_forge` — sampel 24T -> core 10 + head 14 (bits8 Q8 / bits3 BP3); mmap list+blob ok.
21. `test_persist` — upload sekali, dispatch1 4.8ms dispatch2 4.0ms deterministik MATCH cpu; bench 20x 5.5ms/dispatch.
22. `test_sample_fwd` — forward 2-layer dari `.vx27` via mmap (Q8 checksum -0.3146 + BP3 -0.7403); ref==tp; bench Q8 ref 0.058ms/tp 0.48ms.
23. `test_sample64_fwd` — sampel 64L 612T 3.8MB -> core 495 (1.78MB) + head 117 (0.43MB); forward 64-layer mmap ref==tp checksum -0.0652; ref 2.07ms/tp 9.5ms (proxy H64).
24. `test_real_layer` — 1 tensor H5120/F17408 sintetik (89M): forge Q8 101MB + BP3 36MB (3.25bit); mmap blob ok; stream 100MB x20 63GB/s.
25. `test_real_shard` — shard-1 asli 3.7GB 392T (333 vision skip) -> core 59 teks (1.74GB Q8); biggest down_proj [5120,17408] 101MB mmap ok; stream 1.7GB x10 34GB/s.
26. `test_fused0` — forward 1 layer REAL H5120 fused (qkv 52M + FFN 3x89M) via mmap: checksum 16.186 finite, tp==ref maxdiff 0.0, ref 264ms tp 39ms.
27. `test_qfwd64` — FULL 64-layer H5120 bobot asli via mmap (jalur fidelity exact ter-wire): checksum -193.433064 finite, deterministik 0.0, ~1502ms/pass (23.47ms/layer), argmax 16 valid; probe mengunci `q_proj` 12288x5120 dan `o_proj` 5120x6144. Baseline lama pra-fidelity (171.721496/argmax 517) digantikan — perubahan hidden-space yang diharapkan (divergensi kernel lama vs exact 0.53 terukur di `test_deltanet_exact`).
28. `test_pmat_reg` — BP3 in-register (`srlv` 8-wide, tanpa buffer int8) == skalar: diff 0.0 sintetik / 3e-08 bobot nyata; **6.1x** (13.6→2.2ms/row-5120).
29. `test_pmat_batch` — batched matvec 1 pass bobot (B<=8): BP3/F32/Q8 **bit-identik** vs Bx single (maxdiff 0.0 sintetik + baris real gate_proj); bench real 512x5120 B=3: 3x-seq 0.32ms vs batch 0.12ms (**2.62x**).
30. `test_qgen` — GENERASI 3 token H5120 bobot asli via mmap (pasca-fidelity): [16 8 220] deterministik antar-run, id valid, ~0.78 tok/s single-stream tanpa MTP (1261-1287ms/token).
31. `test_gpu_q8` — shader Q8_K tiled + bench gate 17408x5120 B=8: MATCH, gpu 36.2ms (19.7Gval/s) vs cpu-1T 89.9ms vs cpu-8T 18.1ms; autotune pilih CPU (GPU kalah 2x host-visible), deterministik.
32. `test_gpu_bp3` — shader BP3 tiled + upload device-resident sekali: B=1 gpu 4.5ms / B=8 gpu 33ms (21.6Gval/s), MATCH bit-exact (maxdiff 0.0) vs CPU; cpu-8T 23.3ms tetap menang → CPU default, GPU fallback terverifikasi.
33. `test_qfwd_hybrid` — forward 64-layer dgn 12 layer head (52-63) di GPU device-resident + aktivasi host (pasca-fidelity): checksum -193.4331 vs CPU -193.4328 (relerr 1.3e-06), argmax sama 16, speedup ~1.02x (GPU kalah tipis: PCIe ping-pong per-matvec, bukan komputasi).
34. `test_tok` — tokenizer Qwen3.8 248320: load bin (vocab+types+merges HF) + 14/14 golden oracle lib resmi + roundtrip + special/padding handling.
35. `test_sampler` — greedy argmax, determinisme seed, top_k=1, range id vocab penuh.
36. `test_run_smoke` — teks-ujung-ke-ujung (pasca-fidelity: "Halo" -> 2 token [11 174204] deterministik, decode 7 byte tak-kosong) via engine+tokenizer+sampler.
37. `vx27.exe run` — CLI teks-masuk-teks-keluar (pasca-fidelity + prefill batch): `run "Halo" --max-tokens 2 --no-template` -> ", teman" deterministik, ~0.69 tok/s (prefill batch chunked ≤8, bit-identik; dulu 0.52; qgen isolasi ~0.78 tok/s); template chat didukung (`--no-template` utk mentah). Teks mentah masih artefak aproksimasi engine (bukan kualitas bobot asli).
38. `test_mtp` — MTP 1 layer bobot asli, dataflow sesuai referensi vLLM (`[e,h]`, q/gate per-head, input Eagle): 15 tensor bentuk sesuai; draft finite + deterministik bit-identik (~62ms/draft); loop K=2 **exact** vs greedy. Kalibrasi ulang PASCA-FIDELITY: agreement `[e,h]`=3/7 (offset A s=+0: 3/7, s=-1: 1/6), peringkat silang main-top1 di draft=38 + draft-top1 di main=220 (vs pra-fidelity 213rb/183rb anti-korelasi) — hidden-space kini selaras. Assisted K=2 exact vs baseline, accept 1/6 (17%); efektif baseline 0.39 tok/s vs MTP 0.39 tok/s — verify SEQUENSIAL tidak menang (terukur). Batch-verify kini TERIMPLEMENTASI (`test_mtp_batch` no. 39): 1 batch B=3 = 1.9x lebih murah dari 3x forward, TETAPI baseline seq optimal (1 forward/token via pending) tetap menang pada accept 17% (0.91x terukur) — butuh accept >60% utk menang generasi. Prefill batch menang selalu (3.06x). Dequant BP3 tervalidasi vs BF16 asli (korel 0.97).
39. `test_mtp_batch` — batch-forward teacher-forcing + verify spekulatif K=2: batch B=3 **bit-identik** vs 3x sequential (maxdiff 0.0, termasuk logits); prefill batch 7 token 2146ms vs seq ~6560ms (**3.06x**, bit-identik); loop N=4 seq vs batch+commit hasil **identik** (exact); verify batch 0.91x per-token pada accept 17% (terukur jujur — batch 1.9x lebih murah dari 3x forward, tapi baseline seq optimal 1 forward/token via pending; butuh accept >60% utk menang generasi; prefill menang selalu).
40. `test_mtp_kv` — MTP decoder kausal dgn KV-cache sendiri + rollout berantai EAGLE (feat chaining): q/k_norm + RoPE + softmax kausal (gantikan aproksimasi single-token); prefill MTP-KV; sapuan order×gate (terbaik [e,h]+gate 4/7, varian lain 0-1/7); agreement 4/7 (lama 3/7); loop N=4 K=2 accept 2/6=33% exact; loop K=1 (batch B=2) accept 1/3 exact tetapi 0.83x/token — draft+verify batch belum mengalahkan greedy 1-forward/token pd p~0.33 (ambang K=1: p>0.2 dgn V~1.0F; V terukur ~1.5F). Semua bit-identik/deterministik.
41. `test_breakdown` — roofline forward per-komponen (bobot real, 8T): gate 2.75ms/12.65GB/s, down 2.53ms, qkv 1.49ms, out 0.89ms, q 2.26ms, o 1.12ms, lm_head 36.01ms/13.79GB/s; prediksi 761ms/9.9GB per forward (MLP 68%) vs terukur 930-1500ms — memory-bound terbukti (12-14GB/s), glue ≤~20%.
42. `test_refusal` — matcher frasa refusal ID+EN 8/8 + probe observasional 3 kasus (2 disallowed + 1 benign, greedy 4 token, template chat): semua keluaran kolaps `!!!!` (konsisten dgn temuan template-collapse) → tidak ada sinyal refusal terukur; verdict engine informasional, tanpa ubah bobot.
43. `test_full_model` — `.vx27` penuh 18 shard: core 702T/21.4B param/8.7GB + head 164T/5.89B/2.39GB, no-vision, split benar, FIT 11.2/2.8GB, mmap stream 8.8GB/s.
44. `vx27.exe info` — config, 8 physical core, AVX2, threadpool, quant link, tabel KV + budget.
45. `vx_forge.exe plan --out build/manifest.json` — inventory + kebutuhan bit/param.

Hasil terakhir: semua PASS (lihat `build/manifest.json`).

## Budget terukur (bukan klaim)

- `vx27.exe info`: KV/token 32768 elems = 64KB BF16.
  64K Q4 1.07GB FIT, 128K Q4 2.15GB FIT (budget single-channel 2.5GB).
  64K BF16 4.29GB OVER — jadi Q4/FP8 wajib untuk 64K+.
- `vx_forge.exe plan`: backbone est 22.3B + emb/head 2.54B + MTP 0.11B = 24.97B text-only
  (card 27.36B termasuk vision 0.46B + overhead gating/norm).
  Core L0-51 18.13B perlu <=4.94 bit/param agar muat 11.2GB.
  Head L52-63+emb/mtp 6.84B perlu <=3.27 bit/param agar muat 2.8GB VRAM.
  Artinya head wajib ~3-bit + lm_head terkompresi agresif — ini risiko utama berikutnya.

## Batasan

1. Belum ada forward 27B end-to-end (baru sizing + DeltaNet 1-head float ref).
2. Belum ada kuantisasi bit-planar 3-bit custom (masih pakai layout Q4_K/Q6_K `ql` sebagai fondasi).
3. Belum ada backend GPU (Vulkan `ql` sudah probe PASS 3964 MiB, tapi belum diintegrasikan ke `vx27`).
4. `vx_st` hanya scanner header, bukan validator safetensors penuh (belum cek overlap offset/data size).
5. Windows-only `-march=znver3`, sama seperti `ql`.

## Next

- `safetensors` validator penuh + `config.json qwen3_5_text` loader.
- Kernel DeltaNet AVX2 128-dim (update + retrieve) + bench vs ref.
- GQA 24Q/4KV hd256 + RoPE parsial 64 + Mrope + YaRN factor 4.
- PTX/Vulkan `lm_head 5120x248320` + MTP drafter.
- `forge` kuantisasi nyata core/head `.vx27` + uji load 14GB di TUF A15.

Lisensi kode: tambah sebelum distribusi (seperti `ql`). Bobot: Apache 2.0 Qwen + modifikasi Heretic/abliterasi komunitas.
