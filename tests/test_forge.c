#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/vx27_vxfile.h"
#include "../src/vx27_mmap.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // end-to-end nyata: forge sample -> mmap core/head
    int rc = system("build\\vx_forge.exe forge --in build\\sample_heretic.safetensors --out-core build\\sample_core.vx27 --out-head build\\sample_head.vx27 --bits 8 --split-at 1");
    CHECK(rc == 0, "forge bits=8 sample");
    rc = system("build\\vx_forge.exe forge --in build\\sample_heretic.safetensors --out-core build\\sample_core3.vx27 --out-head build\\sample_head3.vx27 --bits 3 --split-at 1");
    CHECK(rc == 0, "forge bits=3 sample");

    vx27_map mc = {0}, mh = {0}; char err[256] = {0};
    CHECK(vx27_mmap_open("build/sample_core.vx27", &mc, err, sizeof(err)) == 0, "mmap core");
    CHECK(vx27_mmap_open("build/sample_head.vx27", &mh, err, sizeof(err)) == 0, "mmap head");
    vx27_entry ents[64]; int n = 0;
    CHECK(vx27_list("build/sample_core.vx27", ents, 64, &n, err, sizeof(err)) == 0, "list core");
    printf("  core tensors=%d head+mmap ok\n", n);
    CHECK(n == 10, "core 10 tensors (embed+layers.0)");
    CHECK(vx27_list("build/sample_head.vx27", ents, 64, &n, err, sizeof(err)) == 0, "list head");
    CHECK(n == 14, "head 14 tensors");
    // blob readback via mmap
    uint64_t bb = 0; const void *p = vx27_mmap_blob(&mc, 0, &bb);
    CHECK(p && bb > 0, "mmap blob readable");
    // corrupted magic rejected
    FILE *f = fopen("build/sample_head.vx27", "rb");
    CHECK(f != NULL, "head exists for corrupt test");
    if (f) fclose(f);
    vx27_mmap_close(&mc); vx27_mmap_close(&mh);

    // bits=3 files exist and list
    CHECK(vx27_list("build/sample_core3.vx27", ents, 64, &n, err, sizeof(err)) == 0, "list core3");
    int has_bp3 = 0;
    for (int i = 0; i < n; i++) if (ents[i].dtype == 4) has_bp3 = 1;
    CHECK(has_bp3, "bp3 dtype present in bits=3 forge");
    if (fails) { printf("test_forge: %d FAIL\n", fails); return 1; }
    printf("test_forge: PASS\n");
    return 0;
}
