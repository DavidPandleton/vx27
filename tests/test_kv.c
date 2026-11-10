#include <stdio.h>
#include "../src/vx27_kv.h"
#include "../src/vx27_config.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    CHECK(vx27_kv_elems(1) == 32768, "32768 elems/token");
    CHECK(vx27_kv_fits(65536, 0.5, VX27_KV_BUDGET_GB), "64K Q4 fits 2.5GB");
    CHECK(vx27_kv_fits(131072, 0.5, VX27_KV_BUDGET_GB), "128K Q4 fits 2.5GB");
    CHECK(!vx27_kv_fits(65536, 2.0, VX27_KV_BUDGET_GB), "64K BF16 over budget");
    CHECK(vx27_kv_gb(8192, 2.0) > 0.5 && vx27_kv_gb(8192, 2.0) < 0.6, "8K BF16 ~0.54GB");
    if (fails) { printf("test_kv: %d FAIL\n", fails); return 1; }
    printf("test_kv: PASS\n");
    return 0;
}
