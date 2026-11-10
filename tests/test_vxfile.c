#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../src/vx27_vxfile.h"
#include "../src/quant.h"
#include "../src/quant_avx2.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // Forge nyata v0: float[512] -> Q8_K (2 blok) + F32 passthrough -> .vx27 -> list back
    float xf[512];
    for (int i = 0; i < 512; i++) xf[i] = (float)(i % 29) * 0.03f - 0.4f;
    block_q8_k q8[2];
    quantize_q8_k(xf, q8, 2);
    float wgt[8] = {1,2,3,4,5,6,7,8};
    const char *names[2] = {"blk.0.ffn_down.q8", "tiny.scale.f32"};
    uint32_t dts[2] = {VX27_D_Q8, VX27_D_F32};
    uint64_t n0[2] = {512, 8}, n1[2] = {1, 1};
    const void *blobs[2] = {q8, wgt};
    uint64_t bbs[2] = {sizeof(q8), sizeof(wgt)};
    char err[256] = {0};
    CHECK(vx27_write("build/test_core.vx27", names, dts, n0, n1, blobs, bbs, 2, err, sizeof(err)) == 0, "write .vx27");
    vx27_entry ents[8]; int n = 0;
    CHECK(vx27_list("build/test_core.vx27", ents, 8, &n, err, sizeof(err)) == 0, "list .vx27");
    CHECK(n == 2 && !strcmp(ents[0].name, "blk.0.ffn_down.q8") && ents[0].bytes == sizeof(q8), "entry0 q8");
    CHECK(ents[1].dtype == VX27_D_F32 && ents[1].bytes == 32, "entry1 f32");
    // Q8 roundtrip: dequant via dot vs float? check first val finite
    CHECK(ents[0].offset == 0 && ents[1].offset == sizeof(q8), "offsets contiguous");

    // Rencana 14GB tanpa alokasi 14GB
    int oc = 0, oh = 0; char msg[256] = {0};
    int rc = vx27_plan_14gb(&oc, &oh, msg, sizeof(msg));
    printf("  plan14: %s (rc=%d)\n", msg, rc);
    CHECK(oc && oh && rc == 0, "14GB plan Q4core+Q3head FIT");
    if (fails) { printf("test_vxfile: %d FAIL\n", fails); return 1; }
    printf("test_vxfile: PASS\n");
    return 0;
}
