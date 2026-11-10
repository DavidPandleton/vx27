// Generator 1-layer real-dim sintetik: 2 tensor BF16 procedural (LCG, no RNG lib).
// T1 core: model.layers.0.mlp.gate_proj.weight [17408,5120] (~89M, 178MB BF16)
// T2 head: model.layers.52.mlp.gate_proj.weight [17408,5120]
// Streaming write, memori konstan. < 60 dtk di TUF A15.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    const char *path = "build/sample_real5120.safetensors";
    const long R = 17408, C = 5120;
    const char *n1 = "model.layers.0.mlp.gate_proj.weight";
    const char *n2 = "model.layers.52.mlp.gate_proj.weight";
    char h1[256], h2[256];
    snprintf(h1, sizeof(h1), "\"%s\":{\"dtype\":\"BF16\",\"shape\":[%ld,%ld],\"data_offsets\":[0,%ld]}", n1, R, C, (long)R*C*2);
    snprintf(h2, sizeof(h2), "\"%s\":{\"dtype\":\"BF16\",\"shape\":[%ld,%ld],\"data_offsets\":[%ld,%ld]}", n2, R, C, (long)R*C*2, (long)R*C*2*2);
    char js[1024];
    int hl = snprintf(js, sizeof(js), "{%s,%s}", h1, h2);
    FILE *f = fopen(path, "wb");
    if (!f) { printf("cannot create %s\n", path); return 1; }
    uint64_t h = (uint64_t)hl;
    fwrite(&h, 8, 1, f);
    fwrite(js, 1, hl, f);
    uint32_t seed = 0x1234567u;
    uint16_t buf[8192];
    long total = R * C;
    for (int rep = 0; rep < 2; rep++) {
        long left = total;
        while (left > 0) {
            int n = left > 8192 ? 8192 : (int)left;
            for (int i = 0; i < n; i++) {
                seed = seed * 1664525u + 1013904223u;
                float v = ((float)(seed >> 9) / 8388608.0f - 0.5f) * 0.2f;
                uint32_t u; memcpy(&u, &v, 4);
                buf[i] = (uint16_t)(u >> 16);
            }
            if (fwrite(buf, 2, n, f) != (size_t)n) { printf("write fail\n"); fclose(f); return 1; }
            left -= n;
        }
    }
    fclose(f);
    printf("wrote %s R=%ld C=%ld x2 tensors\n", path, R, C);
    return 0;
}
