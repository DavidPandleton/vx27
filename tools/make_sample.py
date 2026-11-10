"""Buat sampel safetensors mini ala Heretic (Qwen3.8 text-only, dims kecil).
Tensor names mengikuti konvensi Qwen3.8 (model.layers.N.*) agar forge nyata bisa dipetakan.
Dims kecil: hidden=64, ffn=128, n_q=4 hd=8 (q=32), n_kv=2 (kv=16), lin_qk=2 lin_v=2 lin_hd=8.
Dtype BF16 (top-16 bits of FP32). File < 1MB.
"""
import struct, json, os

H, F = 64, 128
NQH, NKVH, HD = 4, 2, 8
LQK, LV, LHD = 2, 2, 8

def bf16_bytes(vals):
    out = bytearray()
    for v in vals:
        f32 = struct.pack('<f', v)
        out += f32[2:]  # top 16 bits little-endian? BF16 = high 16 of FP32
    return bytes(out)

def seq(n, s=0.01):
    return [((i % 37) - 18) * s for i in range(n)]

tensors = {}
def add(name, shape, vals):
    tensors[name] = (shape, bf16_bytes(vals))

add("model.embed_tokens.weight", [256, H], seq(256*H))
add("model.norm.weight", [H], seq(H))
for L in range(2):
    p = f"model.layers.{L}."
    is_full = (L == 1)
    if is_full:
        add(p+"self_attn.q_proj.weight", [NQH*HD, H], seq(NQH*HD*H))
        add(p+"self_attn.k_proj.weight", [NKVH*HD, H], seq(NKVH*HD*H))
        add(p+"self_attn.v_proj.weight", [NKVH*HD, H], seq(NKVH*HD*H))
        add(p+"self_attn.o_proj.weight", [H, NQH*HD], seq(H*NQH*HD))
        add(p+"self_attn.q_norm.weight", [HD], seq(HD))
        add(p+"self_attn.k_norm.weight", [HD], seq(HD))
    else:
        add(p+"linear_attn.q_proj.weight", [LQK*LHD, H], seq(LQK*LHD*H))
        add(p+"linear_attn.k_proj.weight", [LQK*LHD, H], seq(LQK*LHD*H))
        add(p+"linear_attn.v_proj.weight", [LV*LHD, H], seq(LV*LHD*H))
        add(p+"linear_attn.o_proj.weight", [H, LV*LHD], seq(LV*LHD*H))
    add(p+"mlp.gate_proj.weight", [F, H], seq(F*H))
    add(p+"mlp.up_proj.weight", [F, H], seq(F*H))
    add(p+"mlp.down_proj.weight", [H, F], seq(H*F))
    add(p+"input_layernorm.weight", [H], seq(H))
    add(p+"post_attention_layernorm.weight", [H], seq(H))
add("lm_head.weight", [256, H], seq(256*H))
add("mtp.head.weight", [H, H], seq(H*H))

# header
off = 0
hdr = {}
for name, (shape, blob) in tensors.items():
    hdr[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off+len(blob)]}
    off += len(blob)
js = json.dumps(hdr, separators=(",", ":")).encode()
out = "build/sample_heretic.safetensors"
with open(out, "wb") as f:
    f.write(struct.pack("<Q", len(js)))
    f.write(js)
    for _, (_, blob) in tensors.items():
        f.write(blob)
print(f"wrote {out}: {len(tensors)} tensors, header {len(js)}B, data {off}B, total {8+len(js)+off}B")
for name in list(tensors)[:5]:
    print(" ", name, tensors[name][0])
