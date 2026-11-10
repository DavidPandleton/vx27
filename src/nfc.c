#include "nfc.h"

#include "nfc_tables.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HANGUL_SBASE 0xAC00u
#define HANGUL_LBASE 0x1100u
#define HANGUL_VBASE 0x1161u
#define HANGUL_TBASE 0x11A7u
#define HANGUL_LCOUNT 19u
#define HANGUL_VCOUNT 21u
#define HANGUL_TCOUNT 28u
#define HANGUL_NCOUNT (HANGUL_VCOUNT * HANGUL_TCOUNT)
#define HANGUL_SCOUNT (HANGUL_LCOUNT * HANGUL_NCOUNT)

static int nerr(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

typedef struct {
    uint32_t *p;
    size_t n, cap;
} cpbuf;

static bool cp_push(cpbuf *b, uint32_t cp) {
    if (b->n == b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 256;
        uint32_t *np = (uint32_t *)realloc(b->p, nc * sizeof(uint32_t));
        if (!np) return false;
        b->p = np;
        b->cap = nc;
    }
    b->p[b->n++] = cp;
    return true;
}

static bool utf8_next(const uint8_t *s, size_t n, size_t *i, uint32_t *cp) {
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
        v = c & 0x1F; len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        v = c & 0x0F; len = 3;
    } else if ((c & 0xF8) == 0xF0) {
        v = c & 0x07; len = 4;
    } else {
        *cp = 0xFFFD; *i = k + 1; return true;
    }
    if (k + (size_t)len > n) {
        *cp = 0xFFFD; *i = k + 1; return true;
    }
    for (int j = 1; j < len; j++) {
        uint8_t cc = s[k + (size_t)j];
        if ((cc & 0xC0) != 0x80) {
            *cp = 0xFFFD; *i = k + 1; return true;
        }
        v = (v << 6) | (cc & 0x3F);
    }
    if ((len == 2 && v < 0x80) || (len == 3 && v < 0x800) ||
        (len == 4 && v < 0x10000) || v > 0x10FFFF ||
        (v >= 0xD800 && v <= 0xDFFF)) {
        *cp = 0xFFFD; *i = k + 1; return true;
    }
    *cp = v;
    *i = k + (size_t)len;
    return true;
}

static size_t utf8_put(uint32_t cp, uint8_t *o) {
    if (cp < 0x80) { o[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800) {
        o[0] = (uint8_t)(0xC0 | (cp >> 6));
        o[1] = (uint8_t)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = (uint8_t)(0xE0 | (cp >> 12));
        o[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (uint8_t)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (uint8_t)(0xF0 | (cp >> 18));
    o[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
    o[3] = (uint8_t)(0x80 | (cp & 0x3F));
    return 4;
}

static const nfc_decomp_entry *find_decomp(uint32_t cp) {
    uint32_t lo = 0, hi = NFC_DECOMP_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (cp < NFC_DECOMP[mid].cp) hi = mid;
        else if (cp > NFC_DECOMP[mid].cp) lo = mid + 1;
        else return &NFC_DECOMP[mid];
    }
    return NULL;
}

static uint8_t get_ccc(uint32_t cp) {
    uint32_t lo = 0, hi = NFC_CCC_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (cp < NFC_CCC[mid].cp) hi = mid;
        else if (cp > NFC_CCC[mid].cp) lo = mid + 1;
        else return NFC_CCC[mid].ccc;
    }
    return 0;
}

static bool compose_pair(uint32_t a, uint32_t b, uint32_t *out) {
    /* Hangul LV */
    if (a >= HANGUL_LBASE && a < HANGUL_LBASE + HANGUL_LCOUNT &&
        b >= HANGUL_VBASE && b < HANGUL_VBASE + HANGUL_VCOUNT) {
        uint32_t l = a - HANGUL_LBASE;
        uint32_t v = b - HANGUL_VBASE;
        *out = HANGUL_SBASE + (l * HANGUL_VCOUNT + v) * HANGUL_TCOUNT;
        return true;
    }
    /* Hangul LVT */
    if (a >= HANGUL_SBASE && a < HANGUL_SBASE + HANGUL_SCOUNT &&
        ((a - HANGUL_SBASE) % HANGUL_TCOUNT) == 0 && b > HANGUL_TBASE &&
        b < HANGUL_TBASE + HANGUL_TCOUNT) {
        *out = a + (b - HANGUL_TBASE);
        return true;
    }

    uint32_t lo = 0, hi = NFC_COMP_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        const nfc_comp_entry *e = &NFC_COMP[mid];
        if (a < e->a || (a == e->a && b < e->b)) hi = mid;
        else if (a > e->a || (a == e->a && b > e->b)) lo = mid + 1;
        else { *out = e->cp; return true; }
    }
    return false;
}

static bool decompose_cp(uint32_t cp, cpbuf *out, int depth) {
    if (depth > 32) return false;
    if (cp >= HANGUL_SBASE && cp < HANGUL_SBASE + HANGUL_SCOUNT) {
        uint32_t si = cp - HANGUL_SBASE;
        uint32_t l = HANGUL_LBASE + si / HANGUL_NCOUNT;
        uint32_t v = HANGUL_VBASE + (si % HANGUL_NCOUNT) / HANGUL_TCOUNT;
        uint32_t t = HANGUL_TBASE + si % HANGUL_TCOUNT;
        if (!cp_push(out, l) || !cp_push(out, v)) return false;
        if (t != HANGUL_TBASE && !cp_push(out, t)) return false;
        return true;
    }
    const nfc_decomp_entry *e = find_decomp(cp);
    if (!e) return cp_push(out, cp);
    if (!decompose_cp(e->a, out, depth + 1)) return false;
    if (e->b && !decompose_cp(e->b, out, depth + 1)) return false;
    return true;
}

int nfc_normalize(const uint8_t *in, size_t len, uint8_t **out, size_t *outlen,
                  char *err, size_t errlen) {
    *out = NULL;
    *outlen = 0;

    bool ascii = true;
    for (size_t i = 0; i < len; i++) {
        if (in[i] >= 0x80) { ascii = false; break; }
    }
    if (ascii) {
        uint8_t *p = (uint8_t *)malloc(len ? len : 1);
        if (!p) return nerr(err, errlen, "out of memory");
        if (len) memcpy(p, in, len);
        *out = p;
        *outlen = len;
        return 0;
    }

    cpbuf src = {0}, dec = {0};
    size_t i = 0;
    while (i < len) {
        uint32_t cp;
        if (!utf8_next(in, len, &i, &cp)) break;
        if (!cp_push(&src, cp)) goto oom;
    }

    dec.cap = src.n * 4 + 16;
    if (dec.cap < src.n) {
        free(src.p);
        return nerr(err, errlen, "input is too large for NFC");
    }
    dec.p = (uint32_t *)malloc(dec.cap * sizeof(uint32_t));
    if (!dec.p) goto oom;

    for (size_t k = 0; k < src.n; k++) {
        if (!decompose_cp(src.p[k], &dec, 0)) {
            free(src.p);
            free(dec.p);
            return nerr(err, errlen, "canonical decomposition overflow");
        }
    }
    free(src.p);
    src.p = NULL;

    /* Canonical ordering: stable insertion sort of non-starters by CCC. */
    for (size_t k = 1; k < dec.n; k++) {
        uint8_t cc = get_ccc(dec.p[k]);
        if (cc == 0) continue;
        size_t j = k;
        while (j > 0) {
            uint8_t prev = get_ccc(dec.p[j - 1]);
            if (prev <= cc) break;
            uint32_t t = dec.p[j];
            dec.p[j] = dec.p[j - 1];
            dec.p[j - 1] = t;
            j--;
        }
    }

    /* Canonical composition. */
    size_t wn = 0;
    size_t starter = SIZE_MAX;
    uint8_t last_cc = 0;
    for (size_t k = 0; k < dec.n; k++) {
        uint32_t cp = dec.p[k];
        uint8_t cc = get_ccc(cp);
        bool composed = false;
        if (starter != SIZE_MAX && (last_cc == 0 || last_cc < cc)) {
            uint32_t c;
            if (compose_pair(dec.p[starter], cp, &c)) {
                dec.p[starter] = c;
                composed = true;
            }
        }
        if (composed) continue;
        if (cc == 0) {
            starter = wn;
            last_cc = 0;
        } else {
            last_cc = cc;
        }
        dec.p[wn++] = cp;
    }

    size_t need = wn * 4 + 1;
    uint8_t *buf = (uint8_t *)malloc(need);
    if (!buf) goto oom;
    size_t o = 0;
    for (size_t k = 0; k < wn; k++) o += utf8_put(dec.p[k], buf + o);
    free(dec.p);

    *out = buf;
    *outlen = o;
    return 0;

oom:
    free(src.p);
    free(dec.p);
    return nerr(err, errlen, "out of memory in NFC");
}
