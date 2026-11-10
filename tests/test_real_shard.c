#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) { return (double)clock()*1000.0/CLOCKS_PER_SEC; }
#endif
#include "../src/vx27_vxfile.h"
#include "../src/vx27_st.h"
#include "../src/vx27_mmap.h"
#include "../src/vx27_batch.h"
#include "../src/threadpool.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // Forge shard asli (skip vision default, split default 52)
    int rc = system("build\\vx_forge.exe forge --in build\\model-00001-of-00018.safetensors --out-core build\\shard0_core.vx27 --out-head build\\shard0_head.vx27 --bits 8");
    CHECK(rc == 0, "forge real shard bits=8");
    char err[512] = {0};
    vx27_entry ec[1024], eh[8]; int nce = 0, nhe = 0;
    CHECK(vx27_list("build/shard0_core.vx27", ec, 1024, &nce, err, sizeof(err)) == 0, "list shard core");
    CHECK(vx27_list("build/shard0_head.vx27", eh, 8, &nhe, err, sizeof(err)) == 0, "list shard head (empty ok)");
    printf("  core=%d head=%d (vision skipped, layers 0..4 -> core)\n", nce, nhe);
    CHECK(nce == 59, "59 text tensors (embed + layers 0..4)");
    CHECK(nhe == 0, "head empty (no layers>=52/lm in shard 1)");
    // cari tensor terbesar di core (in_proj_qkv 52M)
    int big = -1; uint64_t bmax = 0;
    for (int i = 0; i < nce; i++) {
        uint64_t el = ec[i].ne0 * ec[i].ne1;
        if (el > bmax) { bmax = el; big = i; }
    }
    CHECK(big >= 0, "big tensor found");
    if (big >= 0) printf("  biggest: %s [%llu,%llu] dtype=%u bytes=%llu\n", ec[big].name,
        (unsigned long long)ec[big].ne0, (unsigned long long)ec[big].ne1,
        ec[big].dtype, (unsigned long long)ec[big].bytes);
    // mmap + bench stream (tanpa dequant penuh 200MB)
    vx27_map mc = {0};
    CHECK(vx27_mmap_open("build/shard0_core.vx27", &mc, err, sizeof(err)) == 0, "mmap shard core");
    uint64_t bb = 0; const void *p = vx27_mmap_blob(&mc, big >= 0 ? big : 0, &bb);
    CHECK(p && bb > 0, "mmap biggest blob readable");
    double t0 = now_ms(); volatile double sink = 0;
    for (int it = 0; it < 10; it++) {
        const uint8_t *b = (const uint8_t*)mc.base;
        double s = 0; for (size_t i = 0; i < mc.size; i += 4096) s += b[i];
        sink += s;
    }
    double t1 = now_ms();
    printf("  stream %.1f MB x10: %.2f ms (%.2f GB/s mmap)\n", (double)mc.size/1e6, t1-t0, (mc.size*10.0)/((t1-t0)/1000.0)/1e9);
    (void)sink;
    vx27_mmap_close(&mc);
    if (fails) { printf("test_real_shard: %d FAIL\n", fails); return 1; }
    printf("test_real_shard: PASS\n");
    return 0;
}
