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
#include "../src/vx27_mmap.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

// Validasi .vx27 penuh dari 18 shard: split, budget, no-vision, mmap stream.
int main(void) {
    char err[512] = {0};
    vx27_entry ec[4096], eh[4096]; int nc = 0, nh = 0;
    CHECK(vx27_list("build/vx27_core.vx27", ec, 4096, &nc, err, sizeof(err)) == 0, "list core");
    CHECK(vx27_list("build/vx27_head.vx27", eh, 4096, &nh, err, sizeof(err)) == 0, "list head");
    printf("  core=%d head=%d total=%d\n", nc, nh, nc + nh);
    uint64_t ce = 0, he = 0, cb = 0, hb = 0;
    int bad = 0;
    for (int i = 0; i < nc; i++) {
        ce += ec[i].ne0 * ec[i].ne1; cb += ec[i].bytes;
        if (strstr(ec[i].name, "visual") || strstr(ec[i].name, "vision")) { printf("  vision leak: %s\n", ec[i].name); bad++; }
        const char *p = strstr(ec[i].name, "layers.");
        if (p && atoi(p + 7) >= 52) { printf("  head layer in core: %s\n", ec[i].name); bad++; }
    }
    for (int i = 0; i < nh; i++) {
        he += eh[i].ne0 * eh[i].ne1; hb += eh[i].bytes;
        if (strstr(eh[i].name, "visual") || strstr(eh[i].name, "vision")) { printf("  vision leak: %s\n", eh[i].name); bad++; }
        const char *p = strstr(eh[i].name, "layers.");
        if (p && atoi(p + 7) < 52 && !strstr(eh[i].name, "embed")) { printf("  core layer in head: %s\n", eh[i].name); bad++; }
    }
    CHECK(bad == 0, "no vision, split layers correct");
    printf("  core %.3fB params %.3f GB | head %.3fB params %.3f GB\n",
        (double)ce/1e9, (double)cb/1e9, (double)he/1e9, (double)hb/1e9);
    CHECK(cb <= 11200000000ULL, "core FIT 11.2GB dual-channel");
    CHECK(hb <= 2800000000ULL, "head FIT 2.8GB VRAM");
    vx27_map mc = {0}, mh = {0};
    CHECK(vx27_mmap_open("build/vx27_core.vx27", &mc, err, sizeof(err)) == 0, "mmap core");
    CHECK(vx27_mmap_open("build/vx27_head.vx27", &mh, err, sizeof(err)) == 0, "mmap head");
    double t0 = now_ms(); volatile double sink = 0;
    for (int it = 0; it < 3; it++) {
        const uint8_t *b = mc.base;
        double s = 0; for (size_t i = 0; i < mc.size; i += 4096) s += b[i];
        sink += s;
    }
    double t1 = now_ms();
    printf("  stream core %.1f GB x3: %.2f ms (%.2f GB/s)\n", (double)mc.size/1e9, t1-t0, (mc.size*3.0)/((t1-t0)/1000.0)/1e9);
    (void)sink;
    vx27_mmap_close(&mc); vx27_mmap_close(&mh);
    if (fails) { printf("test_full_model: %d FAIL\n", fails); return 1; }
    printf("test_full_model: PASS\n");
    return 0;
}
