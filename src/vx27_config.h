#ifndef VX27_CONFIG_H
#define VX27_CONFIG_H

// VORTEX-27 locked target: Qwen3.8-27B text-only (vision tower dropped)
// Sources: Qwen/Qwen3.8-27B model card + vLLM recipes + unsloth config.json
// Forge input: Heretic-Abliterated safetensors (BF16/FP16). GGUF only as ref.

#define VX27_HIDDEN            5120
#define VX27_LAYERS            64
#define VX27_FULL_INTERVAL     4      // every 4th layer is full attention
#define VX27_N_FULL            16
#define VX27_N_LINEAR          48

// Gated DeltaNet (linear) per layer
#define VX27_LIN_QK_HEADS      16
#define VX27_LIN_V_HEADS       48
#define VX27_LIN_HEAD_DIM      128
#define VX27_LIN_CONV_DIM      4

// Gated Attention (full) per full layer
#define VX27_ATTN_Q_HEADS      24
#define VX27_ATTN_KV_HEADS     4
#define VX27_ATTN_HEAD_DIM     256
#define VX27_ROPE_DIM          64
#define VX27_ROPE_THETA        10000000.0f

// FFN SwiGLU
#define VX27_FFN               17408

// Vocab / embeddings (padded, untied)
#define VX27_VOCAB             248320
// 1.27B params in lm_head alone: 5120*248320

// MTP head (built-in drafter)
#define VX27_MTP_LAYERS        1

// Context
#define VX27_CTX_NATIVE        262144
#define VX27_CTX_DEFAULT       65536
#define VX27_CTX_STRETCH       131072  // needs YaRN factor 4.0 + Q4 KV

// Vision tower (dropped in vx27 text-only): 27 blocks, hidden 1152, ~0.46B
#define VX27_VISION_DROPPED    1

// Split budget (blueprint, to be validated by forge)
#define VX27_CORE_GB           11.2   // L1-52 CPU dual-channel lock
#define VX27_HEAD_GB           2.8    // L53-64 + lm_head + MTP on RTX3050
#define VX27_KV_BUDGET_GB      2.5    // single-channel zone

// KV math per token: 16 full layers * 2 * 4 * 256 = 32768 elems
#define VX27_KV_ELEMS_PER_TOK  32768

#endif
