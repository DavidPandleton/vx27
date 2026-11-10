#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/sampler.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

int main(void) {
    char err[256] = {0};
    /* greedy: temp=0 -> argmax, deterministik */
    {
        sampler s; sample_params p; sample_params_default(&p);
        p.temp = 0.0f;
        if (sampler_init(&s, &p, 12345, err, sizeof err) != 0) { printf("FAIL init: %s\n", err); return 1; }
        float lg[5] = {0.1f, 2.0f, 0.5f, 1.0f, -1.0f};
        uint32_t a = sampler_sample(&s, lg, 5);
        uint32_t b = sampler_sample(&s, lg, 5);
        CHECK(a == 1 && b == 1, "greedy argmax + stabil");
        sampler_free(&s);
    }
    /* sampling: seed sama -> hasil sama */
    {
        sampler s1, s2; sample_params p; sample_params_default(&p);
        p.temp = 0.7f; p.top_k = 40;
        sampler_init(&s1, &p, 999, err, sizeof err);
        sampler_init(&s2, &p, 999, err, sizeof err);
        float lg[8];
        for (int i = 0; i < 8; i++) lg[i] = (float)i * 0.3f - 1.0f;
        int same = 1;
        for (int t = 0; t < 20; t++) {
            float a[8], b[8];
            for (int i = 0; i < 8; i++) { a[i] = lg[i]; b[i] = lg[i]; }
            if (sampler_sample(&s1, a, 8) != sampler_sample(&s2, b, 8)) { same = 0; break; }
        }
        CHECK(same, "seed sama -> sequence sama");
        sampler_free(&s1); sampler_free(&s2);
    }
    /* top_k=1 -> argmax juga */
    {
        sampler s; sample_params p; sample_params_default(&p);
        p.temp = 1.0f; p.top_k = 1;
        sampler_init(&s, &p, 7, err, sizeof err);
        float lg[4] = {0.0f, 0.0f, 9.0f, 0.0f};
        CHECK(sampler_sample(&s, lg, 4) == 2, "top_k=1 argmax");
        sampler_free(&s);
    }
    /* id selalu dalam range */
    {
        sampler s; sample_params p; sample_params_default(&p);
        p.temp = 1.0f;
        sampler_init(&s, &p, 42, err, sizeof err);
        float *lg = malloc(248320 * sizeof(float));
        for (int i = 0; i < 248320; i++) lg[i] = -((float)(i % 97)) * 0.1f;
        lg[248319] = 5.0f;
        uint32_t id = sampler_sample(&s, lg, 248320);
        CHECK(id < 248320, "id dalam range vocab penuh");
        free(lg);
        sampler_free(&s);
    }
    if (fails) { printf("test_sampler: %d FAIL\n", fails); return 1; }
    printf("test_sampler: PASS\n");
    return 0;
}
