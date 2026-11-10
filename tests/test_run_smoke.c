#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/vx27_run.h"
#include "../src/tokenizer.h"
#include "../src/sampler.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

// Smoke: teks-masuk-teks-keluar via engine+tokenizer+sampler (tanpa CLI).
// Prompt mentah "Halo" (tanpa template, 2 token) -> generate 2 token greedy.
// Syarat: id valid, teks decode tak-kosong, deterministik antar-run.
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    char err[1024] = {0};
    tokenizer *tok = NULL;
    CHECK(tok_load_bin(&tok, "build/tok_vocab.bin", "build/tok_types.bin",
                       "build/tok_merges.bin", err, sizeof err) == 0, "load tokenizer");
    if (!tok) { printf("  err: %s\n", err); return 1; }

    const char *prompt = "Halo";
    uint32_t *ids = NULL; size_t n_ids = 0;
    CHECK(tok_encode(tok, (const uint8_t*)prompt, strlen(prompt), &ids, &n_ids, err, sizeof err) == 0, "encode Halo");
    printf("  prompt ids: %zu\n", n_ids);
    CHECK(n_ids >= 1 && n_ids <= 8, "panjang prompt wajar");

    vx27_engine *E = NULL;
    int max_t = (int)n_ids + 2;
    CHECK(vx27_engine_open(&E, "build/vx27_core.vx27", "build/vx27_head.vx27", max_t, err, sizeof err) == 0, "engine open");
    if (!E) { printf("  err: %s\n", err); return 1; }

    // prefill + 2x decode greedy, 2 run independen (engine direset via reopen state? pakai reset)
    uint32_t gen1[2] = {0}, gen2[2] = {0};
    float *x = malloc(5120*4), *h = malloc(5120*4), *logits = malloc(248320*4);
    sampler smp; sample_params sp; sample_params_default(&sp);
    sp.temp = 0.0f;
    CHECK(sampler_init(&smp, &sp, 12345, err, sizeof err) == 0, "sampler init");
    for (size_t i = 0; i < n_ids; i++) sampler_accept(&smp, ids[i]);

    for (int run = 0; run < 2; run++) {
        vx27_engine_reset(E);
        sampler_reset(&smp);
        for (size_t i = 0; i < n_ids; i++) sampler_accept(&smp, ids[i]);
        int n_pos = 0;
        for (size_t i = 0; i < n_ids; i++) {
            if (vx27_engine_embed(E, ids[i], x, err, sizeof err) != 0) { printf("FAIL: embed %s\n", err); fails++; goto done; }
            if (vx27_engine_step(E, x, h, n_pos, err, sizeof err) != 0) { printf("FAIL: step\n"); fails++; goto done; }
            n_pos++;
        }
        for (int t = 0; t < 2; t++) {
            uint32_t id = vx27_engine_logits(E, h, logits, err, sizeof err);
            id = sampler_sample(&smp, logits, 248320);
            sampler_accept(&smp, id);
            (run == 0 ? gen1 : gen2)[t] = id;
            if (t + 1 < 2) {
                if (vx27_engine_embed(E, id, x, err, sizeof err) != 0) { printf("FAIL: embed gen\n"); fails++; goto done; }
                if (vx27_engine_step(E, x, h, n_pos, err, sizeof err) != 0) { printf("FAIL: step gen\n"); fails++; goto done; }
                n_pos++;
            }
        }
    }
    printf("  gen1: [%u %u] gen2: [%u %u]\n", gen1[0], gen1[1], gen2[0], gen2[1]);
    CHECK(gen1[0] == gen2[0] && gen1[1] == gen2[1], "deterministik");
    CHECK(gen1[0] < 248320 && gen1[1] < 248320, "id valid");
    {
        uint8_t *bytes = NULL; size_t nb = 0;
        uint32_t both[2] = {gen1[0], gen1[1]};
        CHECK(tok_decode(tok, both, 2, &bytes, &nb, err, sizeof err) == 0, "decode");
        printf("  teks: %zu byte\n", nb);
        CHECK(nb > 0, "teks tak-kosong");
        free(bytes);
    }
done:
    sampler_free(&smp);
    free(x); free(h); free(logits);
    vx27_engine_close(E);
    free(ids);
    tok_free(tok);
    if (fails) { printf("test_run_smoke: %d FAIL\n", fails); return 1; }
    printf("test_run_smoke: PASS\n");
    return 0;
}
