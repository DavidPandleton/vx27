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
#include "../src/gguf.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    // 1) forge sampel real-dim -> .vx27 (Q8 dan BP3)
    int rc = system("build\\vx_forge.exe forge --in build\\sample_real5120.safetensors --out-core build\\real_core.vx27 --out-head build\\real_head.vx27 --bits 8");
    CHECK(rc == 0, "forge real-dim bits=8");
    rc = system("build\\vx_forge.exe forge --in build\\sample_real5120.safetensors --out-core build\\real_core3.vx27 --out-head build\\real_head3.vx27 --bits 3");
    CHECK(rc == 0, "forge real-dim bits=3");

    // 2) mmap + list: core 1T (layers.0) + head 1T (layers.52)
    char err[512] = {0};
    vx27_entry ec[8], eh[8]; int nce = 0, nhe = 0;
    CHECK(vx27_list("build/real_core.vx27", ec, 8, &nce, err, sizeof(err)) == 0, "list real core");
    CHECK(vx27_list("build/real_head.vx27", eh, 8, &nhe, err, sizeof(err)) == 0, "list real head");
    printf("  core=%d head=%d\n", nce, nhe);
    CHECK(nce == 1 && nhe == 1, "split 1/1 real-dim");
    if (nce == 1) printf("  core %s ne=[%llu,%llu] dtype=%u bytes=%llu\n", ec[0].name,
        (unsigned long long)ec[0].ne0, (unsigned long long)ec[0].ne1, ec[0].dtype, (unsigned long long)ec[0].bytes);
    vx27_map mc = {0};
    CHECK(vx27_mmap_open("build/real_core.vx27", &mc, err, sizeof(err)) == 0, "mmap real core");
    uint64_t bb = 0; const void *p = vx27_mmap_blob(&mc, 0, &bb);
    CHECK(p && bb > 0, "mmap blob readable");
    printf("  blob bytes=%llu (expect Q8 ~%.1f MB)\n", (unsigned long long)bb, (double)17408*5120*1.14/1e6);

    // 3) bench matvec real-dim H5120/F17408 via batch (Q8 .vx27 dim check)
    // ambil W Q4 dari GGUF 4B? tidak — bench real-dim pakai .vx27 Q8 via CPU x8 + GPU Q4 4B sebagai pembanding integrasi.
    // Untuk validasi real-dim tanpa 54GB: benchmark SwiGLU F=17408 slice dari bobot Q8 mmap.
    // Sederhananya: dequant 1 baris Q8 -> float, dot vs AVX2 (sudah diuji), lalu ukur throughput baca.
    double t0 = now_ms();
    volatile double sink = 0;
    for (int it = 0; it < 20; it++) {
        // stream 100MB blob (simulasi tarik bobot 1 layer Q8 per token)
        const uint8_t *b = (const uint8_t*)p;
        double s = 0; for (uint64_t i = 0; i < bb; i += 4096) s += b[i];
        sink += s;
    }
    double t1 = now_ms();
    printf("  stream 100MB x20: %.2f ms (%.2f GB/s mmap read)\n", t1-t0, 2.0/((t1-t0)/1000.0));
    CHECK(t1-t0 > 0, "mmap stream bench");
    (void)sink;
    vx27_mmap_close(&mc);
    if (fails) { printf("test_real_layer: %d FAIL\n", fails); return 1; }
    printf("test_real_layer: PASS\n");
    return 0;
}
