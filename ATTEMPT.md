# VORTEX-27 — attempt log (custom 27B local-inference runtime in C)

This repo is an **attempt** at building a zero-dependency local inference runtime
for a 27B model, written from scratch in C11 + AVX2. It is parked, not finished.
This file states honestly what was achieved, what was measured, and what failed.

## Goal

Run **Qwen3.8-27B (Heretic, text-only)** locally on a laptop
(Ryzen 7 7435HS, RTX 3050 4GB, 24GB RAM) as a single static binary:
`vx27.exe`, no third-party runtime deps.

## Lineage

`vx27` reuses proven pieces ported from the sibling project **`ql`**
(4B agent runtime in a separate repo): BPE tokenizer engine, sampler,
bit-planar quant kernels, threadpool + affinity, Vulkan harness, CLI/UX ideas.
Everything model-architecture-specific (Gated DeltaNet, MTP, 64-layer forward)
was written new here.

## What was achieved (all measured, suite green)

- `mingw32-make test` = **42/42 PASS, 0 warning/error/FAIL**.
- Full 64-layer H5120 forward on real weights: finite, deterministic
  (checksum -193.433064, rerun maxdiff 0.0); `q_proj` 12288x5120 and
  `o_proj` 5120x6144 shapes locked by probe.
- DeltaNet fidelity package verified vs HF reference oracle
  (gate mapping bit-identical; step maxdiff 6e-09; conv carryover 1.5e-08).
- Custom BP3_128 weights (3.125 bit): dequant validated vs BF16 (corr 0.97);
  batched matvec 1 weight pass for B vectors, bit-identical, 2.62x.
- Batch-forward teacher forcing + KV trunk/tail + snapshot-free partial commit:
  bit-identical vs sequential; **prefill 3.06x** (shipped in `run`/`chat`).
- Tokenizer 248320 vocab 14/14 golden vs official; sampler; `run`/`chat` CLI.
- Docs: `README.md` (all suite numbers are real outputs) + `ARCH.md`.

## What failed / was parked (honest)

- **Throughput: ~0.5–1 tok/s single-stream, target 5 TPS NOT reached.**
  `test_breakdown` proves the forward is **memory-bound**: 9.9GB weights/token
  over 12–14 GB/s bus = ~761ms/token floor (~1.3 tok/s ceiling even with zero
  glue). 5 TPS needs ~0.7 bit/param or different hardware. GPU measured slower
  than CPU (36ms vs 18ms per gate matvec). Speculative MTP: accept 33%,
  batch-verify 0.83–0.91x/token (loses to greedy). Full roofline in `ARCH.md`.
- **End-to-end text quality unverified.** Template chat sometimes collapses to
  `!!!!`; short outputs look off. Probable causes: 3-bit quant + template/stop
  handling, but no perplexity/logit-vs-reference eval was ever run.
  Do not claim this model is "smart" on the basis of this repo alone.
- Batch verify, KV-tree speculation, refusal probing: implemented, measured,
  kept exact-vs-greedy, but not performance-positive. Parked, not removed.

## Reproduce (weights NOT in this repo — see below)

```bash
export PATH="/c/Users/Public/mingw64/bin:$PATH"
cd vx27
# 1. put 18x Qwen3.8-27B .safetensors shards into build/
# 2. forge: build/vx_forge.exe forge --in build/model-*.safetensors ...
#    (produces build/vx27_core.vx27 ~8.2GB + build/vx27_head.vx27 ~2.3GB)
# 3. mingw32-make test   # 42 suites + info + manifest
./build/vx27.exe run "Halo" --max-tokens 6
```

Set `-include` paths / `VULKAN` in `Makefile` to match the machine.

## What is deliberately NOT pushed

- `build/model-*-of-00018.safetensors` (~52GB, re-downloadable from HF)
- `build/vx27_{core,head}.vx27` (~11GB, re-forgeable from the shards)
- `build/*.exe`, `*.o`, `*.spv` (rebuild via Makefile)

Deleting the local folder loses the ability to run without re-downloading.
Code + tests + docs + measured numbers are all here.

## License

MIT (see `LICENSE`). Model weights (not included) follow their own licenses.
Quant kernel ideas follow GGML/llama.cpp (MIT).
