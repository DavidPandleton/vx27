#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "../src/vx27_st.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

static void write_file(const char *path, const char *js, size_t data_len) {
    size_t hl = strlen(js);
    FILE *f = fopen(path, "wb");
    uint64_t h = (uint64_t)hl;
    uint8_t lb[8]; for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(h >> (8 * i));
    fwrite(lb, 1, 8, f);
    fwrite(js, 1, hl, f);
    for (size_t i = 0; i < data_len; i++) fputc(0, f);
    fclose(f);
}

int main(void) {
    // Good: t1 [0,32) t2 [32,40), data 40B
    write_file("build/test_st.safetensors",
        "{\"t1\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,32]},\"t2\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[32,40]}}", 40);
    vx27_st_entry ents[16]; int n = 0; uint64_t total = 0; char err[256] = {0};
    CHECK(vx27_st_open("build/test_st.safetensors", ents, 16, &n, &total, err, sizeof(err)) == 0, "good file opens");
    CHECK(n == 2 && total == 40, "two tensors total 40B");
    if (n == 2) {
        CHECK(!strcmp(ents[0].dtype, "BF16") && ents[0].elems == 16 && ents[0].bytes == 32, "t1 BF16 4x4 =32B");
        CHECK(!strcmp(ents[1].dtype, "F32") && ents[1].elems == 2 && ents[1].bytes == 8, "t2 F32 [2] =8B");
    }

    // Overlap: t2 starts at 16 inside t1
    write_file("build/test_st_overlap.safetensors",
        "{\"t1\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,32]},\"t2\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]}}", 32);
    CHECK(vx27_st_open("build/test_st_overlap.safetensors", ents, 16, &n, &total, err, sizeof(err)) != 0, "overlap rejected");
    printf("  overlap err: %s\n", err);

    // Size mismatch: shape says 16 elems BF16=32B but offsets only 16B
    write_file("build/test_st_badsize.safetensors",
        "{\"t1\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,16]}}", 16);
    CHECK(vx27_st_open("build/test_st_badsize.safetensors", ents, 16, &n, &total, err, sizeof(err)) != 0, "bytes vs offsets mismatch rejected");
    printf("  badsize err: %s\n", err);

    // Trailing data: offsets end at 32 but file has 40
    write_file("build/test_st_trailing.safetensors",
        "{\"t1\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,32]}}", 40);
    CHECK(vx27_st_open("build/test_st_trailing.safetensors", ents, 16, &n, &total, err, sizeof(err)) != 0, "trailing data rejected");
    printf("  trailing err: %s\n", err);

    // Gap: [0,16) + [24,32) with hole
    write_file("build/test_st_gap.safetensors",
        "{\"t1\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[0,16]},\"t2\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[24,40]}}", 40);
    CHECK(vx27_st_open("build/test_st_gap.safetensors", ents, 16, &n, &total, err, sizeof(err)) != 0, "gap rejected");
    printf("  gap err: %s\n", err);

    CHECK(vx27_dtype_size("BF16") == 2, "dtype BF16=2");
    CHECK(vx27_dtype_size("NOPE") < 0, "unknown dtype rejected");
    if (fails) { printf("test_st: %d FAIL\n", fails); return 1; }
    printf("test_st: PASS\n");
    return 0;
}
