"""Ekstrak tokenizer.json Qwen3.8 -> format binari sederhana utk tok_load_vx.
Format file (little-endian):
  tok_vocab.bin : u32 n (=248320); per id: u32 len + <len> byte (string GPT-2-mapped UTF-8 apa adanya)
  tok_types.bin : u32 n; int32 per id (1=normal, 3=special, 2=unknown/padding)
  tok_merges.bin: u32 n; per merge: u32 len + <len> byte ("kiri kanan" apa adanya)
Aturan:
  - ids 0..248043 dari model.vocab (semua type 1)
  - added_tokens: special=true -> type 3; special=false -> type 1
  - id yg tidak terisi (padding 248077..248319) -> type 2, len 0
Juga tulis golden encode (oracle lib tokenizers resmi) utk test C.
"""
import io, json, struct, sys

SRC = "build/tok.json"
OUT = "build/"

d = json.load(io.open(SRC, encoding="utf-8"))
model = d["model"]
assert model["type"] == "BPE"
N = 248320
vocab = model["vocab"]          # str -> id
merges = model["merges"]        # ["a b", ...]
by_id = [None] * N
typ = [2] * N                   # default unknown/padding
for s, i in vocab.items():
    assert 0 <= i < N, i
    by_id[i] = s.encode("utf-8")
    typ[i] = 1
n_added_special = 0
for t in d["added_tokens"]:
    i = t["id"]
    assert 0 <= i < N, i
    by_id[i] = t["content"].encode("utf-8")
    if t.get("special"):
        typ[i] = 3
        n_added_special += 1
    else:
        typ[i] = 1
n_pad = sum(1 for s in by_id if s is None)
for i in range(N):
    if by_id[i] is None:
        by_id[i] = b""
        typ[i] = 2
print(f"vocab={len(vocab)} merges={len(merges)} special={n_added_special} pad={n_pad} N={N}")
assert len(merges) == 247587, len(merges)

def write_list(path, items):
    with io.open(path, "wb") as f:
        f.write(struct.pack("<I", len(items)))
        for s in items:
            f.write(struct.pack("<I", len(s)))
            f.write(s)

write_list(OUT + "tok_vocab.bin", by_id)
with io.open(OUT + "tok_types.bin", "wb") as f:
    f.write(struct.pack("<I", N))
    for t in typ:
        f.write(struct.pack("<i", t))
write_list(OUT + "tok_merges.bin", [m.encode("utf-8") for m in merges])

# ---- golden oracle (lib resmi) ----
from tokenizers import Tokenizer
tok = Tokenizer.from_file(SRC)
cases = [
    "Hello world",
    "Hello, world! 123",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "Koding AI lokal tanpa sensor.",
    "x=1+2*3",
    "こんにちは世界 🌍",
    "<|im_start|>system<|im_end|>",
    "<|im_start|>user\nSiapa presiden Indonesia?<|im_end|>",
    "a" * 50,
    "   spasi   ganda   ",
    "trailing space ",
    "line1\nline2\nline3",
    "",
    "123456789",
]
with io.open(OUT + "tok_golden.txt", "w", encoding="utf-8") as f:
    for s in cases:
        enc = tok.encode(s)
        f.write(s.encode("utf-8").hex() + "\n")
        f.write(" ".join(map(str, enc.ids)) + "\n")
print("golden cases:", len(cases))
# sanity: special ids
for name in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
    print(name, "->", tok.token_to_id(name))
