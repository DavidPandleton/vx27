#include "tokenizer.h"

#include "nfc.h"
#include "unicode_tables.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HASH_BASE 0x9E3779B185EBCA87ULL
#define MAX_TOKENS  (4u * 1024u * 1024u)
#define MAX_MERGES  (4u * 1024u * 1024u)

typedef struct {
    uint8_t *bytes;
    uint32_t len;
    uint32_t type;
} tok_entry;

typedef struct {
    const uint8_t *p;
    uint64_t hash;
    uint32_t len;
    uint32_t id;
} vocab_slot;

typedef struct {
    const uint8_t *l;
    const uint8_t *r;
    uint64_t hash;
    uint32_t ll;
    uint32_t rl;
    uint32_t rank;
} merge_slot;

typedef struct {
    const uint8_t *p;
    uint32_t len;
    uint32_t id;
} special_entry;

struct tokenizer {
    tok_entry *tokens;
    uint32_t n_tokens;
    vocab_slot *vocab;
    uint32_t vocab_cap;
    merge_slot *merges;
    uint32_t merge_cap;
    special_entry *special;
    uint32_t n_special;
    uint8_t *arena;
    size_t arena_size;
};

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */

static int terr(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* Hashing                                                             */
/* ------------------------------------------------------------------ */

static uint64_t seq_hash(const uint8_t *p, size_t n) {
    uint64_t h = 0;
    for (size_t i = 0; i < n; i++) h = h * HASH_BASE + p[i];
    return h;
}

static uint64_t pair_hash(uint64_t hl, uint32_t ll, uint64_t hr, uint32_t rl) {
    uint64_t h = hl ^ (hr + 0x9E3779B97F4A7C15ULL + (hl << 6) + (hl >> 2));
    h ^= ((uint64_t)ll << 32) | rl;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    return h;
}

static uint32_t bucket(uint64_t h, uint32_t cap) {
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return (uint32_t)h & (cap - 1);
}

static uint32_t next_pow2(uint64_t v) {
    uint32_t p = 1;
    while ((uint64_t)p < v) p <<= 1;
    return p;
}

/* ------------------------------------------------------------------ */
/* UTF-8                                                               */
/* ------------------------------------------------------------------ */

static bool utf8_next(const uint8_t *s, size_t n, size_t *i, uint32_t *cp,
                      bool strict) {
    if (*i >= n) return false;
    size_t k = *i;
    uint8_t c = s[k];
    if (c < 0x80) {
        *cp = c;
        *i = k + 1;
        return true;
    }

    uint32_t v;
    int len;
    if ((c & 0xE0) == 0xC0) {
        v = c & 0x1F;
        len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        v = c & 0x0F;
        len = 3;
    } else if ((c & 0xF8) == 0xF0) {
        v = c & 0x07;
        len = 4;
    } else {
        if (strict) return false;
        *cp = 0xFFFD;
        *i = k + 1;
        return true;
    }

    if (k + (size_t)len > n) {
        if (strict) return false;
        *cp = 0xFFFD;
        *i = k + 1;
        return true;
    }
    for (int j = 1; j < len; j++) {
        uint8_t cc = s[k + (size_t)j];
        if ((cc & 0xC0) != 0x80) {
            if (strict) return false;
            *cp = 0xFFFD;
            *i = k + 1;
            return true;
        }
        v = (v << 6) | (cc & 0x3F);
    }
    if ((len == 2 && v < 0x80) || (len == 3 && v < 0x800) ||
        (len == 4 && v < 0x10000) || v > 0x10FFFF ||
        (v >= 0xD800 && v <= 0xDFFF)) {
        if (strict) return false;
        *cp = 0xFFFD;
        *i = k + 1;
        return true;
    }
    *cp = v;
    *i = k + (size_t)len;
    return true;
}

/* ------------------------------------------------------------------ */
/* Unicode categories                                                  */
/* ------------------------------------------------------------------ */

static bool in_ranges(const u_range *r, uint32_t n, uint32_t cp) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

static bool uni_letter(uint32_t cp) {
    return in_ranges(UNI_LETTER, UNI_LETTER_count, cp);
}
static bool uni_number(uint32_t cp) {
    return in_ranges(UNI_NUMBER, UNI_NUMBER_count, cp);
}
static bool uni_space(uint32_t cp) {
    return in_ranges(UNI_SPACE, UNI_SPACE_count, cp);
}

/* ------------------------------------------------------------------ */
/* GPT-2 byte-level alphabet                                           */
/* ------------------------------------------------------------------ */

static int cp_to_byte[0x200];
static bool cp_map_ready;

static void init_cp_map(void) {
    if (cp_map_ready) return;
    for (int i = 0; i < 0x200; i++) cp_to_byte[i] = -1;

    int bs[256], cs[256], n = 0;
    for (int b = 0x21; b <= 0x7E; b++) { bs[n] = b; cs[n] = b; n++; }
    for (int b = 0xA1; b <= 0xAC; b++) { bs[n] = b; cs[n] = b; n++; }
    for (int b = 0xAE; b <= 0xFF; b++) { bs[n] = b; cs[n] = b; n++; }

    int extra = 0;
    for (int b = 0; b < 256; b++) {
        bool found = false;
        for (int i = 0; i < n; i++) {
            if (bs[i] == b) { found = true; break; }
        }
        if (!found) {
            bs[n] = b;
            cs[n] = 256 + extra;
            extra++;
            n++;
        }
    }
    for (int i = 0; i < n; i++) cp_to_byte[cs[i]] = bs[i];
    cp_map_ready = true;
}

static bool decode_mapped(const uint8_t *s, size_t n, uint8_t *out,
                          uint32_t *outlen) {
    init_cp_map();
    size_t i = 0;
    uint32_t o = 0;
    while (i < n) {
        uint32_t cp;
        if (!utf8_next(s, n, &i, &cp, true)) return false;
        if (cp >= 0x200 || cp_to_byte[cp] < 0) return false;
        out[o++] = (uint8_t)cp_to_byte[cp];
    }
    *outlen = o;
    return true;
}

/* ------------------------------------------------------------------ */
/* Array iteration over GGUF metadata                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
} arr_iter;

static bool arr_next_str(arr_iter *it, const uint8_t **s, uint32_t *n) {
    if (it->p >= it->end) return false;
    if ((size_t)(it->end - it->p) < 8) return false;
    uint64_t len;
    memcpy(&len, it->p, 8);
    it->p += 8;
    if (len > (uint64_t)(it->end - it->p) || len > UINT32_MAX) return false;
    *s = it->p;
    *n = (uint32_t)len;
    it->p += (size_t)len;
    return true;
}

static bool arr_next_i32(arr_iter *it, int32_t *v) {
    if ((size_t)(it->end - it->p) < 4) return false;
    memcpy(v, it->p, 4);
    it->p += 4;
    return true;
}

/* ------------------------------------------------------------------ */
/* Hash tables                                                         */
/* ------------------------------------------------------------------ */

static bool vocab_insert(tokenizer *t, const uint8_t *p, uint32_t len,
                         uint64_t h, uint32_t id) {
    uint32_t i = bucket(h, t->vocab_cap);
    for (;;) {
        vocab_slot *s = &t->vocab[i];
        if (s->id == UINT32_MAX) {
            s->p = p;
            s->hash = h;
            s->len = len;
            s->id = id;
            return true;
        }
        if (s->hash == h && s->len == len && memcmp(s->p, p, len) == 0) {
            if (id < s->id) s->id = id;
            return true;
        }
        i = (i + 1) & (t->vocab_cap - 1);
    }
}

static int vocab_find(const tokenizer *t, const uint8_t *p, uint32_t len,
                      uint64_t h) {
    uint32_t i = bucket(h, t->vocab_cap);
    for (;;) {
        const vocab_slot *s = &t->vocab[i];
        if (s->id == UINT32_MAX) return -1;
        if (s->hash == h && s->len == len && memcmp(s->p, p, len) == 0)
            return (int)s->id;
        i = (i + 1) & (t->vocab_cap - 1);
    }
}

static bool merge_insert(tokenizer *t, const uint8_t *l, uint32_t ll,
                         const uint8_t *r, uint32_t rl, uint32_t rank) {
    uint64_t hl = seq_hash(l, ll);
    uint64_t hr = seq_hash(r, rl);
    uint64_t h = pair_hash(hl, ll, hr, rl);
    uint32_t i = bucket(h, t->merge_cap);
    for (;;) {
        merge_slot *s = &t->merges[i];
        if (s->rank == UINT32_MAX) {
            s->l = l;
            s->r = r;
            s->ll = ll;
            s->rl = rl;
            s->hash = h;
            s->rank = rank;
            return true;
        }
        if (s->hash == h && s->ll == ll && s->rl == rl &&
            memcmp(s->l, l, ll) == 0 && memcmp(s->r, r, rl) == 0) {
            if (rank < s->rank) s->rank = rank;
            return true;
        }
        i = (i + 1) & (t->merge_cap - 1);
    }
}

static int merge_find(const tokenizer *t, const uint8_t *l, uint32_t ll,
                      uint64_t hl, const uint8_t *r, uint32_t rl,
                      uint64_t hr) {
    uint64_t h = pair_hash(hl, ll, hr, rl);
    uint32_t i = bucket(h, t->merge_cap);
    for (;;) {
        const merge_slot *s = &t->merges[i];
        if (s->rank == UINT32_MAX) return -1;
        if (s->hash == h && s->ll == ll && s->rl == rl &&
            memcmp(s->l, l, ll) == 0 && memcmp(s->r, r, rl) == 0)
            return (int)s->rank;
        i = (i + 1) & (t->merge_cap - 1);
    }
}

/* ------------------------------------------------------------------ */
/* Load                                                                */
/* ------------------------------------------------------------------ */

static void tok_free_inner(tokenizer *t) {
    if (!t) return;
    free(t->tokens);
    free(t->vocab);
    free(t->merges);
    free(t->special);
    free(t->arena);
    free(t);
}

void tok_free(tokenizer *t) { tok_free_inner(t); }

static const gguf_kv *need_array(const gguf_file *g, const char *key,
                                 uint32_t elem_type, char *err, size_t errlen) {
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv || kv->type != GGUF_T_ARRAY) {
        terr(err, errlen, "metadata '%s' is not an array", key);
        return NULL;
    }
    if (kv->v.arr.elem_type != elem_type) {
        terr(err, errlen, "metadata '%s' has element type %u, want %u", key,
             kv->v.arr.elem_type, elem_type);
        return NULL;
    }
    return kv;
}

int tok_load(tokenizer **out, const gguf_file *g, char *err, size_t errlen) {
    *out = NULL;
    init_cp_map();

    const gguf_kv *tokv = need_array(g, "tokenizer.ggml.tokens", GGUF_T_STRING,
                                     err, errlen);
    if (!tokv) return -1;
    const gguf_kv *merv = need_array(g, "tokenizer.ggml.merges", GGUF_T_STRING,
                                     err, errlen);
    if (!merv) return -1;
    const gguf_kv *tyv = need_array(g, "tokenizer.ggml.token_type",
                                    GGUF_T_INT32, err, errlen);
    if (!tyv) return -1;

    if (tokv->v.arr.count == 0 || tokv->v.arr.count > MAX_TOKENS ||
        merv->v.arr.count > MAX_MERGES ||
        tyv->v.arr.count != tokv->v.arr.count) {
        return terr(err, errlen, "invalid tokenizer array sizes");
    }

    tokenizer *t = (tokenizer *)calloc(1, sizeof(*t));
    if (!t) return terr(err, errlen, "out of memory");

    t->n_tokens = (uint32_t)tokv->v.arr.count;
    t->tokens = (tok_entry *)calloc(t->n_tokens, sizeof(*t->tokens));
    if (!t->tokens) goto oom;

    size_t arena_need = 0;
    arr_iter it = {tokv->v.arr.data, tokv->v.arr.end};
    for (uint32_t i = 0; i < t->n_tokens; i++) {
        const uint8_t *s;
        uint32_t n;
        if (!arr_next_str(&it, &s, &n)) {
            terr(err, errlen, "corrupt token array");
            tok_free_inner(t);
            return -1;
        }
        arena_need += n;
    }
    it.p = merv->v.arr.data;
    it.end = merv->v.arr.end;
    for (uint64_t i = 0; i < merv->v.arr.count; i++) {
        const uint8_t *s;
        uint32_t n;
        if (!arr_next_str(&it, &s, &n)) {
            terr(err, errlen, "corrupt merge array");
            tok_free_inner(t);
            return -1;
        }
        arena_need += n;
    }
    arena_need += 16;
    t->arena = (uint8_t *)malloc(arena_need);
    if (!t->arena) goto oom;
    t->arena_size = arena_need;
    uint8_t *cur = t->arena;

    /* token types */
    arr_iter tyit = {tyv->v.arr.data, tyv->v.arr.end};
    int32_t *types = (int32_t *)calloc(t->n_tokens, sizeof(int32_t));
    if (!types) goto oom;
    for (uint32_t i = 0; i < t->n_tokens; i++) {
        if (!arr_next_i32(&tyit, &types[i])) {
            free(types);
            terr(err, errlen, "corrupt token_type array");
            tok_free_inner(t);
            return -1;
        }
    }

    /* tokens */
    it.p = tokv->v.arr.data;
    it.end = tokv->v.arr.end;
    uint32_t n_special = 0;
    for (uint32_t i = 0; i < t->n_tokens; i++) {
        const uint8_t *s;
        uint32_t n;
        if (!arr_next_str(&it, &s, &n)) {
            free(types);
            terr(err, errlen, "corrupt token array");
            tok_free_inner(t);
            return -1;
        }
        t->tokens[i].type = (uint32_t)types[i];
        if (types[i] == 3 || types[i] == 4) {
            memcpy(cur, s, n);
            t->tokens[i].bytes = cur;
            t->tokens[i].len = n;
            cur += n;
            n_special++;
        } else {
            uint32_t dlen;
            if (!decode_mapped(s, n, cur, &dlen)) {
                free(types);
                terr(err, errlen,
                     "token %u is not valid GPT-2 byte-level text", i);
                tok_free_inner(t);
                return -1;
            }
            t->tokens[i].bytes = cur;
            t->tokens[i].len = dlen;
            cur += dlen;
        }
    }
    free(types);

    t->n_special = n_special;
    t->special = n_special ? (special_entry *)calloc(n_special, sizeof(*t->special)) : NULL;
    if (n_special && !t->special) goto oom;
    uint32_t si = 0;
    for (uint32_t i = 0; i < t->n_tokens; i++) {
        if (t->tokens[i].type == 3 || t->tokens[i].type == 4) {
            t->special[si].p = t->tokens[i].bytes;
            t->special[si].len = t->tokens[i].len;
            t->special[si].id = i;
            si++;
        }
    }

    /* vocabulary table (normal + byte tokens only) */
    uint64_t vcap = next_pow2((uint64_t)t->n_tokens * 2 + 1);
    if (vcap > (1u << 30)) {
        terr(err, errlen, "tokenizer vocabulary is too large");
        tok_free_inner(t);
        return -1;
    }
    t->vocab_cap = (uint32_t)vcap;
    t->vocab = (vocab_slot *)malloc((size_t)t->vocab_cap * sizeof(*t->vocab));
    if (!t->vocab) goto oom;
    memset(t->vocab, 0xFF, (size_t)t->vocab_cap * sizeof(*t->vocab));
    for (uint32_t i = 0; i < t->n_tokens; i++) {
        if (t->tokens[i].type == 3 || t->tokens[i].type == 4) continue;
        uint64_t h = seq_hash(t->tokens[i].bytes, t->tokens[i].len);
        vocab_insert(t, t->tokens[i].bytes, t->tokens[i].len, h, i);
    }

    /* merge table */
    uint64_t mcap = next_pow2((uint64_t)merv->v.arr.count * 2 + 1);
    if (mcap > (1u << 30)) {
        terr(err, errlen, "merge table is too large");
        tok_free_inner(t);
        return -1;
    }
    t->merge_cap = (uint32_t)mcap;
    t->merges = (merge_slot *)malloc((size_t)t->merge_cap * sizeof(*t->merges));
    if (!t->merges) goto oom;
    memset(t->merges, 0xFF, (size_t)t->merge_cap * sizeof(*t->merges));

    it.p = merv->v.arr.data;
    it.end = merv->v.arr.end;
    for (uint64_t i = 0; i < merv->v.arr.count; i++) {
        const uint8_t *s;
        uint32_t n;
        if (!arr_next_str(&it, &s, &n)) {
            terr(err, errlen, "corrupt merge array");
            tok_free_inner(t);
            return -1;
        }
        uint32_t sep = UINT32_MAX;
        for (uint32_t k = 0; k < n; k++) {
            if (s[k] == ' ') { sep = k; break; }
        }
        if (sep == UINT32_MAX || sep == 0 || sep + 1 >= n) {
            terr(err, errlen, "malformed merge entry %" PRIu64, i);
            tok_free_inner(t);
            return -1;
        }
        uint32_t ll, rl;
        uint8_t *lp = cur;
        if (!decode_mapped(s, sep, lp, &ll)) {
            terr(err, errlen, "malformed left side in merge %" PRIu64, i);
            tok_free_inner(t);
            return -1;
        }
        cur += ll;
        uint8_t *rp = cur;
        if (!decode_mapped(s + sep + 1, n - sep - 1, rp, &rl)) {
            terr(err, errlen, "malformed right side in merge %" PRIu64, i);
            tok_free_inner(t);
            return -1;
        }
        cur += rl;
        if (cur > t->arena + t->arena_size) {
            terr(err, errlen, "tokenizer arena overflow");
            tok_free_inner(t);
            return -1;
        }
        merge_insert(t, lp, ll, rp, rl, (uint32_t)i);
    }

    if (cur > t->arena + t->arena_size) {
        terr(err, errlen, "tokenizer arena overflow");
        tok_free_inner(t);
        return -1;
    }

    *out = t;
    return 0;

oom:
    terr(err, errlen, "out of memory");
    tok_free_inner(t);
    return -1;
}

/* ------------------------------------------------------------------ */
/* Decode                                                              */
/* ------------------------------------------------------------------ */

uint32_t tok_special_id(const tokenizer *t, const char *name) {
    size_t n = strlen(name);
    for (uint32_t i = 0; i < t->n_special; i++) {
        if (t->special[i].len == n && memcmp(t->special[i].p, name, n) == 0)
            return t->special[i].id;
    }
    return UINT32_MAX;
}

const uint8_t *tok_token_bytes(const tokenizer *t, uint32_t id, size_t *len) {
    if (id >= t->n_tokens) return NULL;
    if (len) *len = t->tokens[id].len;
    return t->tokens[id].bytes;
}

size_t tok_vocab_size(const tokenizer *t) { return t->n_tokens; }

size_t tok_special_count(const tokenizer *t) { return t->n_special; }

int tok_decode(const tokenizer *t, const uint32_t *ids, size_t n,
               uint8_t **bytes, size_t *n_bytes, char *err, size_t errlen) {
    *bytes = NULL;
    *n_bytes = 0;
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        if (ids[i] >= t->n_tokens)
            return terr(err, errlen, "token id %u out of range", ids[i]);
        total += t->tokens[ids[i]].len;
    }
    uint8_t *out = (uint8_t *)malloc(total ? total : 1);
    if (!out) return terr(err, errlen, "out of memory");
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        const tok_entry *e = &t->tokens[ids[i]];
        memcpy(out + o, e->bytes, e->len);
        o += e->len;
    }
    *bytes = out;
    *n_bytes = o;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Pre-tokenizer: Qwen2 regex                                          */
/* ------------------------------------------------------------------ */

static uint32_t ascii_lower(uint32_t c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c;
}

static size_t match_contraction(const uint32_t *cp, size_t n, size_t i) {
    if (i >= n || cp[i] != '\'') return i;
    if (i + 1 < n) {
        uint32_t a = ascii_lower(cp[i + 1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if (i + 2 < n) {
            uint32_t b = ascii_lower(cp[i + 2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                (a == 'l' && b == 'l'))
                return i + 3;
        }
    }
    return i;
}

static size_t match_letters(const uint32_t *cp, size_t n, size_t i) {
    if (i >= n) return i;
    if (uni_letter(cp[i])) {
        size_t j = i + 1;
        while (j < n && uni_letter(cp[j])) j++;
        return j;
    }
    if (i + 1 < n && cp[i] != '\r' && cp[i] != '\n' && !uni_letter(cp[i]) &&
        !uni_number(cp[i]) && uni_letter(cp[i + 1])) {
        size_t j = i + 2;
        while (j < n && uni_letter(cp[j])) j++;
        return j;
    }
    return i;
}

static size_t match_number(const uint32_t *cp, size_t n, size_t i) {
    return (i < n && uni_number(cp[i])) ? i + 1 : i;
}

static size_t match_punct(const uint32_t *cp, size_t n, size_t i) {
    size_t j = i;
    if (j < n && cp[j] == ' ') {
        if (j + 1 < n && !uni_space(cp[j + 1]) && !uni_letter(cp[j + 1]) &&
            !uni_number(cp[j + 1]))
            j++;
        else
            return i;
    }
    if (j < n && !uni_space(cp[j]) && !uni_letter(cp[j]) &&
        !uni_number(cp[j])) {
        j++;
        while (j < n && !uni_space(cp[j]) && !uni_letter(cp[j]) &&
               !uni_number(cp[j]))
            j++;
        while (j < n && (cp[j] == '\r' || cp[j] == '\n')) j++;
        return j;
    }
    return i;
}

static size_t match_newline(const uint32_t *cp, size_t n, size_t i) {
    if (i >= n || !uni_space(cp[i])) return i;
    size_t j = i;
    while (j < n && uni_space(cp[j])) j++;
    size_t last = SIZE_MAX;
    for (size_t k = i; k < j; k++) {
        if (cp[k] == '\r' || cp[k] == '\n') last = k;
    }
    return last == SIZE_MAX ? i : last + 1;
}

static size_t match_ws_alt6(const uint32_t *cp, size_t n, size_t i) {
    if (i >= n || !uni_space(cp[i])) return i;
    size_t j = i;
    while (j < n && uni_space(cp[j])) j++;
    if (j == n) return j;          /* trailing whitespace: consume all */
    if (j - i >= 2) return j - 1;  /* leave the last space for the next chunk */
    return i;
}

static size_t match_ws(const uint32_t *cp, size_t n, size_t i) {
    if (i >= n || !uni_space(cp[i])) return i;
    size_t j = i + 1;
    while (j < n && uni_space(cp[j])) j++;
    return j;
}

/* ------------------------------------------------------------------ */
/* BPE                                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t off;
    uint32_t len;
    uint64_t hash;
    int32_t prev;
    int32_t next;
    uint8_t alive;
} bnode;

typedef struct {
    uint32_t rank;
    uint32_t l;
    uint32_t r;
} hent;

static bool heap_less(const hent *a, const hent *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->l < b->l;
}

static void heap_push(hent *h, uint32_t *n, uint32_t cap, hent e) {
    if (*n >= cap) return; /* cannot happen with 4n capacity */
    uint32_t i = (*n)++;
    h[i] = e;
    while (i > 0) {
        uint32_t p = (i - 1) / 2;
        if (!heap_less(&h[i], &h[p])) break;
        hent tmp = h[i];
        h[i] = h[p];
        h[p] = tmp;
        i = p;
    }
}

static bool heap_pop(hent *h, uint32_t *n, hent *out) {
    if (*n == 0) return false;
    *out = h[0];
    h[0] = h[--(*n)];
    uint32_t i = 0;
    for (;;) {
        uint32_t l = i * 2 + 1, r = l + 1, best = i;
        if (l < *n && heap_less(&h[l], &h[best])) best = l;
        if (r < *n && heap_less(&h[r], &h[best])) best = r;
        if (best == i) break;
        hent tmp = h[i];
        h[i] = h[best];
        h[best] = tmp;
        i = best;
    }
    return true;
}

static int out_push(uint32_t **out, size_t *n, size_t *cap, uint32_t id) {
    if (*n == *cap) {
        size_t nc = *cap ? *cap * 2 : 256;
        uint32_t *p = (uint32_t *)realloc(*out, nc * sizeof(uint32_t));
        if (!p) return -1;
        *out = p;
        *cap = nc;
    }
    (*out)[(*n)++] = id;
    return 0;
}

static int bpe_chunk(const tokenizer *t, const uint8_t *p, size_t n,
                     uint32_t **out, size_t *nout, size_t *cap,
                     char *err, size_t errlen) {
    if (n == 0) return 0;
    if (n > UINT32_MAX) return terr(err, errlen, "chunk is too large");

    bnode *nodes = (bnode *)malloc(n * sizeof(bnode));
    hent *heap = (hent *)malloc((4 * n + 16) * sizeof(hent));
    uint64_t *pw = (uint64_t *)malloc((n + 1) * sizeof(uint64_t));
    if (!nodes || !heap || !pw) {
        free(nodes);
        free(heap);
        free(pw);
        return terr(err, errlen, "out of memory in BPE");
    }

    pw[0] = 1;
    for (size_t i = 1; i <= n; i++) pw[i] = pw[i - 1] * HASH_BASE;

    for (size_t i = 0; i < n; i++) {
        nodes[i].off = (uint32_t)i;
        nodes[i].len = 1;
        nodes[i].hash = p[i];
        nodes[i].prev = i ? (int32_t)(i - 1) : -1;
        nodes[i].next = i + 1 < n ? (int32_t)(i + 1) : -1;
        nodes[i].alive = 1;
    }

    uint32_t hn = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        uint64_t ph = pair_hash(nodes[i].hash, 1, nodes[i + 1].hash, 1);
        int rank = merge_find(t, p + nodes[i].off, 1, nodes[i].hash,
                              p + nodes[i + 1].off, 1, nodes[i + 1].hash);
        if (rank >= 0) {
            hent e = {(uint32_t)rank, (uint32_t)i, (uint32_t)(i + 1)};
            heap_push(heap, &hn, (uint32_t)(4 * n + 16), e);
        }
        (void)ph;
    }

    hent e;
    while (heap_pop(heap, &hn, &e)) {
        if (!nodes[e.l].alive || !nodes[e.r].alive) continue;
        if (nodes[e.l].next != (int32_t)e.r) continue;
        uint32_t ll = nodes[e.l].len, rl = nodes[e.r].len;
        int rank = merge_find(t, p + nodes[e.l].off, ll, nodes[e.l].hash,
                              p + nodes[e.r].off, rl, nodes[e.r].hash);
        if (rank < 0 || (uint32_t)rank != e.rank) continue;

        nodes[e.l].len = ll + rl;
        nodes[e.l].hash = nodes[e.l].hash * pw[rl] + nodes[e.r].hash;
        int32_t nx = nodes[e.r].next;
        nodes[e.l].next = nx;
        if (nx >= 0) nodes[nx].prev = (int32_t)e.l;
        nodes[e.r].alive = 0;

        int32_t pv = nodes[e.l].prev;
        if (pv >= 0) {
            uint32_t llen = nodes[pv].len, rlen = nodes[e.l].len;
            int r2 = merge_find(t, p + nodes[pv].off, llen, nodes[pv].hash,
                                p + nodes[e.l].off, rlen, nodes[e.l].hash);
            if (r2 >= 0) {
                hent ne = {(uint32_t)r2, (uint32_t)pv, e.l};
                heap_push(heap, &hn, (uint32_t)(4 * n + 16), ne);
            }
        }
        if (nx >= 0) {
            uint32_t llen = nodes[e.l].len, rlen = nodes[nx].len;
            int r2 = merge_find(t, p + nodes[e.l].off, llen, nodes[e.l].hash,
                                p + nodes[nx].off, rlen, nodes[nx].hash);
            if (r2 >= 0) {
                hent ne = {(uint32_t)r2, e.l, (uint32_t)nx};
                heap_push(heap, &hn, (uint32_t)(4 * n + 16), ne);
            }
        }
    }

    int rc = 0;
    int32_t i = 0;
    while (i >= 0) {
        if (!nodes[i].alive) {
            rc = terr(err, errlen, "internal BPE list corruption");
            break;
        }
        int id = vocab_find(t, p + nodes[i].off, nodes[i].len, nodes[i].hash);
        if (id < 0) {
            rc = terr(err, errlen, "BPE result is not in vocabulary (len %u)",
                      nodes[i].len);
            break;
        }
        if (out_push(out, nout, cap, (uint32_t)id) != 0) {
            rc = terr(err, errlen, "out of memory");
            break;
        }
        i = nodes[i].next;
    }

    free(nodes);
    free(heap);
    free(pw);
    return rc;
}

static int encode_segment(const tokenizer *t, const uint8_t *text, size_t len,
                          uint32_t **out, size_t *nout, size_t *cap,
                          char *err, size_t errlen) {
    if (len == 0) return 0;

    uint32_t *cps = (uint32_t *)malloc(len * sizeof(uint32_t));
    size_t *offs = (size_t *)malloc((len + 1) * sizeof(size_t));
    if (!cps || !offs) {
        free(cps);
        free(offs);
        return terr(err, errlen, "out of memory in pre-tokenizer");
    }

    size_t ncp = 0, i = 0;
    while (i < len) {
        offs[ncp] = i;
        uint32_t cp;
        if (!utf8_next(text, len, &i, &cp, false)) break;
        cps[ncp++] = cp;
    }
    offs[ncp] = len;

    size_t k = 0;
    while (k < ncp) {
        size_t j = match_contraction(cps, ncp, k);
        if (j == k) j = match_letters(cps, ncp, k);
        if (j == k) j = match_number(cps, ncp, k);
        if (j == k) j = match_punct(cps, ncp, k);
        if (j == k) j = match_newline(cps, ncp, k);
        if (j == k) j = match_ws_alt6(cps, ncp, k);
        if (j == k) j = match_ws(cps, ncp, k);
        if (j == k) j = k + 1;
        if (j > ncp) j = ncp;

        size_t a = offs[k], b = offs[j];
        if (b > a) {
            int rc = bpe_chunk(t, text + a, b - a, out, nout, cap, err, errlen);
            if (rc != 0) {
                free(cps);
                free(offs);
                return rc;
            }
        }
        k = j;
    }

    free(cps);
    free(offs);
    return 0;
}

static bool match_special(const tokenizer *t, const uint8_t *p, size_t n,
                          uint32_t *id, size_t *len) {
    size_t best = 0;
    uint32_t best_id = 0;
    for (uint32_t i = 0; i < t->n_special; i++) {
        const special_entry *s = &t->special[i];
        if (s->len > best && s->len <= n && memcmp(p, s->p, s->len) == 0) {
            best = s->len;
            best_id = s->id;
        }
    }
    if (best) {
        *id = best_id;
        *len = best;
        return true;
    }
    return false;
}

static int tok_encode_raw(const tokenizer *t, const uint8_t *text, size_t len,
                          uint32_t **ids, size_t *n_ids, char *err,
                          size_t errlen) {
    *ids = NULL;
    *n_ids = 0;
    uint32_t *out = NULL;
    size_t n = 0, cap = 0;

    size_t seg = 0, i = 0;
    while (i < len) {
        uint32_t sid;
        size_t sl;
        if (match_special(t, text + i, len - i, &sid, &sl)) {
            if (i > seg) {
                int rc = encode_segment(t, text + seg, i - seg, &out, &n, &cap,
                                        err, errlen);
                if (rc != 0) {
                    free(out);
                    return rc;
                }
            }
            if (out_push(&out, &n, &cap, sid) != 0) {
                free(out);
                return terr(err, errlen, "out of memory");
            }
            i += sl;
            seg = i;
        } else {
            size_t j = i;
            uint32_t cp;
            if (!utf8_next(text, len, &j, &cp, false)) break;
            i = j;
        }
    }
    if (seg < len) {
        int rc = encode_segment(t, text + seg, len - seg, &out, &n, &cap, err,
                                errlen);
        if (rc != 0) {
            free(out);
            return rc;
        }
    }

    *ids = out;
    *n_ids = n;
    return 0;
}

int tok_encode(const tokenizer *t, const uint8_t *text, size_t len,
               uint32_t **ids, size_t *n_ids, char *err, size_t errlen) {
    uint8_t *norm = NULL;
    size_t nlen = 0;
    if (nfc_normalize(text, len, &norm, &nlen, err, errlen) != 0) return -1;
    int rc = tok_encode_raw(t, norm, nlen, ids, n_ids, err, errlen);
    free(norm);
    return rc;
}
