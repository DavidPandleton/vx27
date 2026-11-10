#include "sampler.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sample_params_default(sample_params *p) {
    p->temp = 0.3f;
    p->top_p = 0.9f;
    p->min_p = 0.0f;
    p->repeat_penalty = 1.1f;
    p->top_k = 40;
    p->repeat_last_n = 64;
}

static int serr(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int sampler_init(sampler *s, const sample_params *p, uint64_t seed, char *err,
                 size_t errlen) {
    memset(s, 0, sizeof(*s));
    s->p = *p;
    if (s->p.temp < 0.0f) s->p.temp = 0.0f;
    if (s->p.top_p <= 0.0f || s->p.top_p > 1.0f) s->p.top_p = 1.0f;
    if (s->p.min_p < 0.0f || s->p.min_p > 1.0f) s->p.min_p = 0.0f;
    if (s->p.repeat_penalty <= 0.0f) s->p.repeat_penalty = 1.0f;
    if (s->p.repeat_last_n < 0) s->p.repeat_last_n = 0;
    if (s->p.top_k <= 0) s->p.top_k = 40;

    uint64_t x = seed ? seed : 0x123456789ABCDEFULL;
    for (int i = 0; i < 4; i++) s->s[i] = splitmix64(&x);
    if (s->p.repeat_last_n > 0) {
        s->recent =
            (uint32_t *)calloc((size_t)s->p.repeat_last_n, sizeof(uint32_t));
        if (!s->recent) return serr(err, errlen, "out of memory");
    }
    return 0;
}

void sampler_free(sampler *s) {
    if (!s) return;
    free(s->recent);
    s->recent = NULL;
    s->n_recent = 0;
    s->pos = 0;
}

void sampler_reset(sampler *s) {
    s->n_recent = 0;
    s->pos = 0;
}

void sampler_accept(sampler *s, uint32_t id) {
    if (!s->recent || s->p.repeat_last_n <= 0) return;
    s->recent[s->pos] = id;
    s->pos = (s->pos + 1) % s->p.repeat_last_n;
    if (s->n_recent < s->p.repeat_last_n) s->n_recent++;
}

static uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(sampler *s) {
    const uint64_t result = rotl64(s->s[1] * 5, 7) * 9;
    const uint64_t t = s->s[1] << 17;
    s->s[2] ^= s->s[0];
    s->s[3] ^= s->s[1];
    s->s[1] ^= s->s[2];
    s->s[0] ^= s->s[3];
    s->s[2] ^= t;
    s->s[3] = rotl64(s->s[3], 45);
    return result;
}

static float rng_float(sampler *s) {
    return (float)(rng_next(s) >> 40) * (1.0f / 16777216.0f);
}

typedef struct {
    float v;
    int id;
} cand;

static void heap_sift_down(cand *h, int n, int i) {
    for (;;) {
        int l = i * 2 + 1, r = l + 1, m = i;
        if (l < n && h[l].v < h[m].v) m = l;
        if (r < n && h[r].v < h[m].v) m = r;
        if (m == i) break;
        cand t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

static void heap_sift_up(cand *h, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].v <= h[i].v) break;
        cand t = h[i];
        h[i] = h[p];
        h[p] = t;
        i = p;
    }
}

static int cmp_cand_desc(const void *a, const void *b) {
    const cand *x = (const cand *)a;
    const cand *y = (const cand *)b;
    if (x->v < y->v) return 1;
    if (x->v > y->v) return -1;
    return x->id - y->id;
}

uint32_t sampler_sample(sampler *s, float *logits, int vocab) {
    /* Repetition penalty, once per unique token. */
    if (s->p.repeat_penalty > 1.0f && s->recent && s->n_recent > 0) {
        uint32_t seen[256];
        int nseen = 0;
        for (int i = 0; i < s->n_recent; i++) {
            int idx = (s->pos - 1 - i + s->p.repeat_last_n * 2) %
                      s->p.repeat_last_n;
            uint32_t id = s->recent[idx];
            if ((int)id >= vocab) continue;
            int dup = 0;
            for (int k = 0; k < nseen; k++) {
                if (seen[k] == id) {
                    dup = 1;
                    break;
                }
            }
            if (dup) continue;
            if (nseen < (int)(sizeof(seen) / sizeof(seen[0]))) seen[nseen++] = id;
            if (logits[id] > 0.0f) logits[id] /= s->p.repeat_penalty;
            else logits[id] *= s->p.repeat_penalty;
        }
    }

    if (s->p.temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[best]) best = i;
        return (uint32_t)best;
    }

    int k = s->p.top_k;
    if (k > vocab) k = vocab;
    if (k < 1) k = 1;
    if (k > 1024) k = 1024;

    cand *heap = (cand *)malloc((size_t)k * sizeof(cand));
    if (!heap) {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[best]) best = i;
        return (uint32_t)best;
    }

    int hn = 0;
    for (int i = 0; i < vocab; i++) {
        float v = logits[i];
        if (hn < k) {
            heap[hn].v = v;
            heap[hn].id = i;
            heap_sift_up(heap, hn);
            hn++;
        } else if (v > heap[0].v) {
            heap[0].v = v;
            heap[0].id = i;
            heap_sift_down(heap, hn, 0);
        }
    }
    qsort(heap, (size_t)hn, sizeof(cand), cmp_cand_desc);

    /* temperature + softmax */
    float mx = heap[0].v * (1.0f / s->p.temp);
    float sum = 0.0f;
    for (int i = 0; i < hn; i++) {
        heap[i].v = expf(heap[i].v / s->p.temp - mx);
        sum += heap[i].v;
    }

    /* min-p */
    int keep = hn;
    if (s->p.min_p > 0.0f) {
        float thresh = (heap[0].v / sum) * s->p.min_p;
        keep = 1;
        for (int i = 1; i < hn; i++) {
            if (heap[i].v / sum < thresh) break;
            keep = i + 1;
        }
    }

    /* top-p */
    if (s->p.top_p < 1.0f) {
        float cum = 0.0f;
        int n = 0;
        for (int i = 0; i < keep; i++) {
            cum += heap[i].v / sum;
            n = i + 1;
            if (cum >= s->p.top_p) break;
        }
        if (n > 0) keep = n;
    }
    if (keep < 1) keep = 1;

    float total = 0.0f;
    for (int i = 0; i < keep; i++) total += heap[i].v;
    float r = rng_float(s) * total;
    uint32_t out = (uint32_t)heap[keep - 1].id;
    for (int i = 0; i < keep; i++) {
        r -= heap[i].v;
        if (r <= 0.0f) {
            out = (uint32_t)heap[i].id;
            break;
        }
    }
    free(heap);
    return out;
}
