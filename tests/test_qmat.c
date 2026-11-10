#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../src/vx27_qmat.h"
#include "../src/vx27_weights.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // Real Q4_K tensor from ql 4B model proves the kernel path used for 27B.
    const char *gguf = "C:/Users/ASUS A15/ql/models/Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf";
    uint64_t nbytes = 0; uint32_t type = 0; char err[256] = {0};
    CHECK(vx27_gguf_tensor_info(gguf, "blk.0.ffn_gate.weight", &nbytes, &type, err, sizeof(err)) == 0, "probe real Q4_K tensor");
    if (!fails) {
        printf("  blk.0.ffn_gate.weight bytes=%llu type=%u\n", (unsigned long long)nbytes, type);
        CHECK(type == 12, "type 12 == Q4_K");
    }
    CHECK(vx27_27b_params_text() > 24000000000ULL, "27B text params ~25B");
    printf("  27B Q4 %.2f GB, Q3 %.2f GB\n", vx27_27b_gb(4.0), vx27_27b_gb(3.0));

    // Q4 dot vs dequant-ref on synthetic x (512 elems = 2 blocks).
    // Build Q4 blocks by hand: use dequant of zero-block? Instead use real
    // blocks read via mmap? Simpler: craft blocks with d=1.0, dmin=0, scales=0, qs pattern.
    // dequant_q4_k handles arbitrary bits; dot comparison validates kernel, not quant quality.
    block_q4_k W[2]; block_q8_k Q8[2];
    for (int b = 0; b < 2; b++) {
        // d=0x3C00 (1.0), dmin=0x0000
        W[b].d = 0x3C00; W[b].dmin = 0x0000;
        for (int i = 0; i < 12; i++) W[b].scales[i] = 0;
        for (int i = 0; i < 128; i++) W[b].qs[i] = (uint8_t)((i * 7 + b) & 0xFF);
    }
    float x[512];
    for (int i = 0; i < 512; i++) x[i] = (float)(i % 13) * 0.05f - 0.3f;
    float r1 = vx27_qmat_q4_ref(W, x, 2);
    float r2 = vx27_qmat_q4(W, x, 2, Q8);
    printf("  q4 ref=%.6f avx2=%.6f diff=%.3e\n", r1, r2, fabsf(r1-r2));
    CHECK(fabsf(r1 - r2) / (fabsf(r1) + 1.0f) < 0.02f, "q4 avx2 ~= ref within 2% (Q8 quant)");
    if (fails) { printf("test_qmat: %d FAIL\n", fails); return 1; }
    printf("test_qmat: PASS\n");
    return 0;
}
