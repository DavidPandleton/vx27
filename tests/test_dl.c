#include <stdio.h>
#include "../src/vx27_gpu_dl.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // Upload 14MB (same tensor as q4 bench) staging -> DEVICE_LOCAL
    double ms = 0, bw = 0; char err[256] = {0};
    int rc = vx27_gpu_dl_upload(14*1024*1024, &ms, &bw, err, sizeof(err));
    CHECK(rc == 0, "dl upload 14MB");
    printf("  dl 14MB: %.3f ms %.2f GB/s\n", ms, bw);
    CHECK(ms < 50.0, "dl <50ms");
    if (fails) { printf("test_dl: %d FAIL\n", fails); return 1; }
    printf("test_dl: PASS\n");
    return 0;
}
