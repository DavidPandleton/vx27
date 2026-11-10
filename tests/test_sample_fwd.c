#include <stdio.h>
#include <math.h>
#include "../src/vx27_sample_fwd.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    char err[512] = {0};
    double m1 = 0, m2 = 0;
    float c1 = vx27_sample_fwd("build/sample_core.vx27", "build/sample_head.vx27", 0, &m1, err, sizeof(err));
    if (err[0]) printf("  fwd err: %s\n", err);
    CHECK(err[0] == 0, "fwd Q8 mmap end-to-end");
    float c2 = vx27_sample_fwd("build/sample_core.vx27", "build/sample_head.vx27", 1, &m2, err, sizeof(err));
    printf("  checksum ref=%.6f tp=%.6f\n", c1, c2);
    CHECK(fabsf(c1 - c2) < 1e-3f, "tp == ref (mmap Q8)");
    printf("  bench ref %.3f ms tp %.3f ms\n", m1, m2);
    double m3 = 0;
    float c3 = vx27_sample_fwd("build/sample_core3.vx27", "build/sample_head3.vx27", 1, &m3, err, sizeof(err));
    printf("  bp3 checksum=%.6f ms=%.3f\n", c3, m3);
    CHECK(err[0] == 0 && c3 == c3, "fwd BP3 mmap end-to-end");
    if (fails) { printf("test_sample_fwd: %d FAIL\n", fails); return 1; }
    printf("test_sample_fwd: PASS\n");
    return 0;
}
