#ifndef QL_SAMPLER_H
#define QL_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float temp;
    float top_p;
    float min_p;
    float repeat_penalty;
    int top_k;
    int repeat_last_n;
} sample_params;

typedef struct {
    sample_params p;
    uint64_t s[4];
    uint32_t *recent;
    int n_recent;
    int pos;
} sampler;

void sample_params_default(sample_params *p);

int sampler_init(sampler *s, const sample_params *p, uint64_t seed, char *err,
                 size_t errlen);
void sampler_free(sampler *s);
void sampler_reset(sampler *s);

/* Applies repetition penalty to logits in place and returns one token. */
uint32_t sampler_sample(sampler *s, float *logits, int vocab);

/* Record a generated token for repetition penalty / cache updates. */
void sampler_accept(sampler *s, uint32_t id);

#endif /* QL_SAMPLER_H */
