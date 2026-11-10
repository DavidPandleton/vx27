#include <stdio.h>
#include <string.h>
#include "../src/vx27_cfg.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

static void write_good(const char *p) {
    FILE *f = fopen(p, "w");
    fprintf(f, "{");
    fprintf(f, "\"hidden_size\":5120,");
    fprintf(f, "\"num_hidden_layers\":64,");
    fprintf(f, "\"num_attention_heads\":24,");
    fprintf(f, "\"num_key_value_heads\":4,");
    fprintf(f, "\"head_dim\":256,");
    fprintf(f, "\"intermediate_size\":17408,");
    fprintf(f, "\"vocab_size\":248320,");
    fprintf(f, "\"max_position_embeddings\":262144,");
    fprintf(f, "\"rope_theta\":10000000,");
    fprintf(f, "\"partial_rotary_factor\":0.25,");
    fprintf(f, "\"linear_num_key_heads\":16,");
    fprintf(f, "\"linear_num_value_heads\":48,");
    fprintf(f, "\"linear_key_head_dim\":128,");
    fprintf(f, "\"linear_conv_kernel_dim\":4,");
    fprintf(f, "\"full_attention_interval\":4,");
    fprintf(f, "\"mtp_num_hidden_layers\":1,");
    fprintf(f, "\"rope_type\":\"yarn\",\"factor\":4.0,");
    fprintf(f, "\"mrope_section\":[11,11,10],");
    fprintf(f, "\"text_config\":{\"model_type\":\"qwen3_5_text\"}");
    fprintf(f, "}");
    fclose(f);
}

int main(void) {
    write_good("build/test_cfg_good.json");
    vx27_cfg c; char err[256] = {0};
    CHECK(vx27_cfg_load("build/test_cfg_good.json", &c, err, sizeof(err)) == 0, "load good config");
    if (!fails) {
        CHECK(c.hidden == 5120 && c.layers == 64, "hidden/layers parsed");
        CHECK(c.n_q == 24 && c.n_kv == 4 && c.head_dim == 256, "gqa parsed");
        CHECK(c.rope_dim == 64, "rope_dim 256*0.25=64");
        CHECK(c.lin_qk == 16 && c.lin_v == 48 && c.lin_hd == 128, "linear parsed");
        CHECK(c.mrope_sec[0] == 11 && c.mrope_sec[1] == 11 && c.mrope_sec[2] == 10, "mrope parsed");
        CHECK(vx27_cfg_check_locked(&c, err, sizeof(err)) == 0, "locked check PASS");
    }
    // bad: layers 32
    FILE *f = fopen("build/test_cfg_bad.json", "w");    fprintf(f, "{\"hidden_size\":5120,\"num_hidden_layers\":32,\"num_attention_heads\":24,\"num_key_value_heads\":4,\"head_dim\":256,\"intermediate_size\":17408,\"vocab_size\":248320,\"max_position_embeddings\":262144,\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"linear_num_key_heads\":16,\"linear_num_value_heads\":48,\"linear_key_head_dim\":128,\"linear_conv_kernel_dim\":4,\"full_attention_interval\":4,\"mtp_num_hidden_layers\":1,\"mrope_section\":[11,11,10]}");
    fclose(f);
    vx27_cfg cb;
    CHECK(vx27_cfg_load("build/test_cfg_bad.json", &cb, err, sizeof(err)) == 0, "load bad config");
    CHECK(vx27_cfg_check_locked(&cb, err, sizeof(err)) != 0, "locked check rejects layers=32");
    printf("  bad err: %s\n", err);
    // real Qwen3.8 config.json (4KB, sudah diunduh) — validasi kunci bila ada
    FILE *rf = fopen("build/qwen38_config.json", "rb");
    if (rf) {
        fclose(rf);
        vx27_cfg rc;
        CHECK(vx27_cfg_load("build/qwen38_config.json", &rc, err, sizeof(err)) == 0, "load real qwen3.8 config");
        if (!fails) {
            CHECK(rc.hidden == 5120 && rc.layers == 64, "real hidden/layers");
            CHECK(rc.rope_dim == 64 && rc.rope_theta == 10000000.0f, "real rope");
            CHECK(vx27_cfg_check_locked(&rc, err, sizeof(err)) == 0, "real locked check PASS");
            if (err[0]) printf("  real err: %s\n", err);
        }
    } else {
        printf("  skip real config (not downloaded)\n");
    }
    if (fails) { printf("test_cfg: %d FAIL\n", fails); return 1; }
    printf("test_cfg: PASS\n");
    return 0;
}
