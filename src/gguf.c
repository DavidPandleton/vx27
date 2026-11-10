#define _WIN32_WINNT 0x0A00
#include "gguf.h"

#include <windows.h>

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* GGML tensor type table: elements per block, bytes per block.        */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t block;
    uint64_t bytes;
    const char *name;
} type_info;

#define TI_MAX 40
static const type_info TYPE_INFO[TI_MAX] = {
    [0]  = {1, 4, "F32"},
    [1]  = {1, 2, "F16"},
    [2]  = {32, 18, "Q4_0"},
    [3]  = {32, 20, "Q4_1"},
    [4]  = {32, 20, "Q4_2"},
    [5]  = {32, 24, "Q4_3"},
    [6]  = {32, 22, "Q5_0"},
    [7]  = {32, 24, "Q5_1"},
    [8]  = {32, 34, "Q8_0"},
    [9]  = {32, 36, "Q8_1"},
    [10] = {256, 84, "Q2_K"},
    [11] = {256, 110, "Q3_K"},
    [12] = {256, 144, "Q4_K"},
    [13] = {256, 176, "Q5_K"},
    [14] = {256, 210, "Q6_K"},
    [15] = {256, 292, "Q8_K"},
    [16] = {256, 66, "IQ2_XXS"},
    [17] = {256, 74, "IQ2_XS"},
    [18] = {256, 98, "IQ3_XXS"},
    [19] = {256, 50, "IQ1_S"},
    [20] = {32, 18, "IQ4_NL"},
    [21] = {256, 110, "IQ3_S"},
    [22] = {256, 82, "IQ2_S"},
    [23] = {256, 136, "IQ4_XS"},
    [24] = {1, 1, "I8"},
    [25] = {1, 2, "I16"},
    [26] = {1, 4, "I32"},
    [27] = {1, 8, "I64"},
    [28] = {1, 8, "F64"},
    [29] = {256, 56, "IQ1_M"},
    [30] = {1, 2, "BF16"},
    [31] = {32, 18, "Q4_0_4_4"},
    [32] = {32, 18, "Q4_0_4_8"},
    [33] = {32, 18, "Q4_0_8_8"},
    [34] = {256, 54, "TQ1_0"},
    [35] = {256, 66, "TQ2_0"},
    [36] = {32, 18, "IQ4_NL_4_4"},
    [37] = {32, 18, "IQ4_NL_4_8"},
    [38] = {32, 18, "IQ4_NL_8_8"},
    [39] = {32, 17, "MXFP4"},
};

uint64_t gguf_type_block(uint32_t type) {
    return type < TI_MAX ? TYPE_INFO[type].block : 0;
}

uint64_t gguf_type_bytes(uint32_t type) {
    return type < TI_MAX ? TYPE_INFO[type].bytes : 0;
}

const char *gguf_type_name(uint32_t type) {
    return type < TI_MAX && TYPE_INFO[type].name ? TYPE_INFO[type].name : "UNKNOWN";
}

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

static gguf_status fail(gguf_status st, char *err, size_t errlen,
                        const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return st;
}

static bool u64_mul_ok(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool u64_add_ok(uint64_t a, uint64_t b, uint64_t *out) {
    if (b > UINT64_MAX - a) return false;
    *out = a + b;
    return true;
}

static bool align_up_u64(uint64_t x, uint64_t a, uint64_t *out) {
    if (a == 0 || (a & (a - 1)) != 0) return false;
    if (x > UINT64_MAX - (a - 1)) return false;
    *out = (x + a - 1) & ~(a - 1);
    return true;
}

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
} cur;

static bool need(const cur *c, uint64_t n) {
    return n <= (uint64_t)(c->size - c->pos);
}

static bool rd_u8(cur *c, uint8_t *v) {
    if (!need(c, 1)) return false;
    *v = c->base[c->pos++];
    return true;
}

static bool rd_u16(cur *c, uint16_t *v) {
    if (!need(c, 2)) return false;
    memcpy(v, c->base + c->pos, 2);
    c->pos += 2;
    return true;
}

static bool rd_u32(cur *c, uint32_t *v) {
    if (!need(c, 4)) return false;
    memcpy(v, c->base + c->pos, 4);
    c->pos += 4;
    return true;
}

static bool rd_u64(cur *c, uint64_t *v) {
    if (!need(c, 8)) return false;
    memcpy(v, c->base + c->pos, 8);
    c->pos += 8;
    return true;
}

static bool rd_str_view(cur *c, const uint8_t **ptr, uint64_t *len) {
    uint64_t n;
    if (!rd_u64(c, &n)) return false;
    if (!need(c, n)) return false;
    *ptr = c->base + c->pos;
    *len = n;
    c->pos += (size_t)n;
    return true;
}

static uint64_t scalar_size(uint32_t type) {
    switch (type) {
    case GGUF_T_UINT8:
    case GGUF_T_INT8:
    case GGUF_T_BOOL:
        return 1;
    case GGUF_T_UINT16:
    case GGUF_T_INT16:
        return 2;
    case GGUF_T_UINT32:
    case GGUF_T_INT32:
    case GGUF_T_FLOAT32:
        return 4;
    case GGUF_T_UINT64:
    case GGUF_T_INT64:
    case GGUF_T_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_array_body(cur *c, uint32_t elem_type, uint64_t count,
                            int depth) {
    if (depth > 8) return false;

    if (elem_type == GGUF_T_STRING) {
        for (uint64_t i = 0; i < count; i++) {
            uint64_t n;
            if (!rd_u64(c, &n) || !need(c, n)) return false;
            c->pos += (size_t)n;
        }
        return true;
    }

    if (elem_type == GGUF_T_ARRAY) {
        for (uint64_t i = 0; i < count; i++) {
            uint32_t sub;
            uint64_t n;
            if (!rd_u32(c, &sub) || !rd_u64(c, &n)) return false;
            if (!skip_array_body(c, sub, n, depth + 1)) return false;
        }
        return true;
    }

    uint64_t es = scalar_size(elem_type);
    if (es == 0) return false;
    uint64_t bytes;
    if (!u64_mul_ok(es, count, &bytes)) return false;
    if (!need(c, bytes)) return false;
    c->pos += (size_t)bytes;
    return true;
}

/* ------------------------------------------------------------------ */
/* File mapping                                                        */
/* ------------------------------------------------------------------ */

static wchar_t *utf8_to_wide(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t *w = (wchar_t *)calloc((size_t)n, sizeof(wchar_t));
    if (!w) return NULL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n) != n) {
        free(w);
        return NULL;
    }
    return w;
}

static gguf_status map_file(const char *path, void **out_map, size_t *out_size,
                            char *err, size_t errlen) {
    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) {
        return fail(GGUF_E_IO, err, errlen, "path is not valid UTF-8");
    }

    HANDLE h = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    free(wpath);
    if (h == INVALID_HANDLE_VALUE) {
        return fail(GGUF_E_IO, err, errlen, "CreateFileW failed (error %lu)",
                    (unsigned long)GetLastError());
    }

    LARGE_INTEGER li;
    if (!GetFileSizeEx(h, &li) || li.QuadPart <= 0) {
        CloseHandle(h);
        return fail(GGUF_E_IO, err, errlen, "cannot get file size");
    }
    if ((uint64_t)li.QuadPart > (uint64_t)SIZE_MAX) {
        CloseHandle(h);
        return fail(GGUF_E_IO, err, errlen, "file is too large");
    }

    HANDLE m = CreateFileMappingW(h, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m) {
        CloseHandle(h);
        return fail(GGUF_E_IO, err, errlen,
                    "CreateFileMappingW failed (error %lu)",
                    (unsigned long)GetLastError());
    }

    void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    CloseHandle(h);
    if (!p) {
        return fail(GGUF_E_IO, err, errlen,
                    "MapViewOfFile failed (error %lu)",
                    (unsigned long)GetLastError());
    }

    *out_map = p;
    *out_size = (size_t)li.QuadPart;
    return GGUF_OK;
}

/* ------------------------------------------------------------------ */
/* Parsing                                                             */
/* ------------------------------------------------------------------ */

static bool kv_read_scalar(cur *c, uint32_t type, gguf_kv *kv) {
    switch (type) {
    case GGUF_T_UINT8:  return rd_u8(c, &kv->v.u8);
    case GGUF_T_INT8: {
        uint8_t t;
        if (!rd_u8(c, &t)) return false;
        kv->v.i8 = (int8_t)t;
        return true;
    }
    case GGUF_T_UINT16: return rd_u16(c, &kv->v.u16);
    case GGUF_T_INT16: {
        uint16_t t;
        if (!rd_u16(c, &t)) return false;
        kv->v.i16 = (int16_t)t;
        return true;
    }
    case GGUF_T_UINT32: return rd_u32(c, &kv->v.u32);
    case GGUF_T_INT32: {
        uint32_t t;
        if (!rd_u32(c, &t)) return false;
        kv->v.i32 = (int32_t)t;
        return true;
    }
    case GGUF_T_FLOAT32: {
        uint32_t t;
        if (!rd_u32(c, &t)) return false;
        memcpy(&kv->v.f32, &t, 4);
        return true;
    }
    case GGUF_T_BOOL: {
        uint8_t t;
        if (!rd_u8(c, &t)) return false;
        kv->v.b = t != 0;
        return true;
    }
    case GGUF_T_UINT64: return rd_u64(c, &kv->v.u64);
    case GGUF_T_INT64: {
        uint64_t t;
        if (!rd_u64(c, &t)) return false;
        kv->v.i64 = (int64_t)t;
        return true;
    }
    case GGUF_T_FLOAT64: {
        uint64_t t;
        if (!rd_u64(c, &t)) return false;
        memcpy(&kv->v.f64, &t, 8);
        return true;
    }
    default:
        return false;
    }
}

static char *dup_bytes(const uint8_t *p, uint64_t len) {
    if (len > 1u << 20) return NULL; /* keys / short names only */
    char *s = (char *)calloc((size_t)len + 1, 1);
    if (!s) return NULL;
    if (len) memcpy(s, p, (size_t)len);
    return s;
}

static gguf_status parse_kv(cur *c, gguf_kv *kv, char *err, size_t errlen) {
    const uint8_t *kp = NULL;
    uint64_t klen = 0;
    if (!rd_str_view(c, &kp, &klen)) {
        return fail(GGUF_E_TRUNC, err, errlen, "truncated metadata key");
    }
    kv->key = dup_bytes(kp, klen);
    if (!kv->key) {
        return fail(GGUF_E_NOMEM, err, errlen, "out of memory for metadata key");
    }

    uint32_t type;
    if (!rd_u32(c, &type)) {
        return fail(GGUF_E_TRUNC, err, errlen, "truncated metadata type for '%s'",
                    kv->key);
    }
    kv->type = type;

    if (type == GGUF_T_STRING) {
        if (!rd_str_view(c, &kv->v.str.ptr, &kv->v.str.len)) {
            return fail(GGUF_E_TRUNC, err, errlen,
                        "truncated string value for '%s'", kv->key);
        }
        return GGUF_OK;
    }

    if (type < GGUF_T_ARRAY) {
        if (!kv_read_scalar(c, type, kv)) {
            return fail(GGUF_E_TRUNC, err, errlen,
                        "truncated metadata value for '%s'", kv->key);
        }
        return GGUF_OK;
    }

    if (type == GGUF_T_ARRAY) {
        uint32_t et;
        uint64_t count;
        if (!rd_u32(c, &et) || !rd_u64(c, &count)) {
            return fail(GGUF_E_TRUNC, err, errlen,
                        "truncated array header for '%s'", kv->key);
        }
        if (et >= GGUF_T_COUNT || count > (uint64_t)1 << 32) {
            return fail(GGUF_E_FORMAT, err, errlen,
                        "unsupported array type/count for '%s'", kv->key);
        }
        kv->v.arr.elem_type = et;
        kv->v.arr.count = count;
        kv->v.arr.data = c->base + c->pos;
        if (!skip_array_body(c, et, count, 0)) {
            return fail(GGUF_E_TRUNC, err, errlen,
                        "truncated array body for '%s'", kv->key);
        }
        kv->v.arr.end = c->base + c->pos;
        return GGUF_OK;
    }

    return fail(GGUF_E_FORMAT, err, errlen, "unknown metadata type %u for '%s'",
                type, kv->key);
}

static gguf_status parse_tensor(cur *c, gguf_tensor *t, char *err,
                                size_t errlen) {
    const uint8_t *np = NULL;
    uint64_t nlen = 0;
    if (!rd_str_view(c, &np, &nlen)) {
        return fail(GGUF_E_TRUNC, err, errlen, "truncated tensor name");
    }
    t->name = dup_bytes(np, nlen);
    if (!t->name) {
        return fail(GGUF_E_NOMEM, err, errlen, "out of memory for tensor name");
    }

    if (!rd_u32(c, &t->n_dims)) {
        return fail(GGUF_E_TRUNC, err, errlen, "truncated dims for '%s'",
                    t->name);
    }
    if (t->n_dims == 0 || t->n_dims > 4) {
        return fail(GGUF_E_FORMAT, err, errlen,
                    "tensor '%s' has unsupported rank %u", t->name, t->n_dims);
    }

    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) {
        if (!rd_u64(c, &t->ne[i])) {
            return fail(GGUF_E_TRUNC, err, errlen, "truncated dims for '%s'",
                        t->name);
        }
        if (t->ne[i] == 0) {
            return fail(GGUF_E_FORMAT, err, errlen,
                        "tensor '%s' has zero dimension", t->name);
        }
        if (!u64_mul_ok(n, t->ne[i], &n)) {
            return fail(GGUF_E_FORMAT, err, errlen,
                        "tensor '%s' dimension overflow", t->name);
        }
    }
    t->n_elements = n;

    if (!rd_u32(c, &t->type) || !rd_u64(c, &t->offset)) {
        return fail(GGUF_E_TRUNC, err, errlen, "truncated tensor info for '%s'",
                    t->name);
    }

    uint64_t block = gguf_type_block(t->type);
    uint64_t bytes = gguf_type_bytes(t->type);
    if (block == 0 || bytes == 0) {
        return fail(GGUF_E_UNSUPPORTED, err, errlen,
                    "tensor '%s' has unknown type %u", t->name, t->type);
    }
    if (t->n_elements % block != 0) {
        return fail(GGUF_E_FORMAT, err, errlen,
                    "tensor '%s' has %" PRIu64 " elements, not a multiple of %"
                    PRIu64, t->name, t->n_elements, block);
    }
    if (!u64_mul_ok(t->n_elements / block, bytes, &t->n_bytes)) {
        return fail(GGUF_E_FORMAT, err, errlen,
                    "tensor '%s' byte size overflow", t->name);
    }
    return GGUF_OK;
}

static int cmp_span(const void *a, const void *b) {
    const gguf_tensor *ta = *(const gguf_tensor *const *)a;
    const gguf_tensor *tb = *(const gguf_tensor *const *)b;
    if (ta->offset < tb->offset) return -1;
    if (ta->offset > tb->offset) return 1;
    return 0;
}

static gguf_status validate_tensors(gguf_file *g, char *err, size_t errlen) {
    if (g->n_tensors > 1000000) {
        return fail(GGUF_E_FORMAT, err, errlen, "too many tensors");
    }

    uint64_t data_avail = (uint64_t)g->size - g->data_offset;
    uint64_t max_end = 0;

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (t->offset % g->alignment != 0) {
            return fail(GGUF_E_FORMAT, err, errlen,
                        "tensor '%s' offset %" PRIu64 " is not aligned to %"
                        PRIu64, t->name, t->offset, g->alignment);
        }
        if (t->offset > data_avail || t->n_bytes > data_avail - t->offset) {
            return fail(GGUF_E_FORMAT, err, errlen,
                        "tensor '%s' exceeds file bounds", t->name);
        }
        uint64_t end;
        if (!u64_add_ok(t->offset, t->n_bytes, &end)) {
            return fail(GGUF_E_FORMAT, err, errlen,
                        "tensor '%s' range overflow", t->name);
        }
        if (end > max_end) max_end = end;
    }
    g->data_size = max_end;

    gguf_tensor **sorted =
        (gguf_tensor **)calloc((size_t)g->n_tensors, sizeof(*sorted));
    if (!sorted) {
        return fail(GGUF_E_NOMEM, err, errlen, "out of memory for validation");
    }
    for (uint64_t i = 0; i < g->n_tensors; i++) sorted[i] = &g->tensors[i];
    qsort(sorted, (size_t)g->n_tensors, sizeof(*sorted), cmp_span);

    for (uint64_t i = 1; i < g->n_tensors; i++) {
        const gguf_tensor *prev = sorted[i - 1];
        const gguf_tensor *cur_t = sorted[i];
        uint64_t prev_end;
        if (!u64_add_ok(prev->offset, prev->n_bytes, &prev_end)) {
            free(sorted);
            return fail(GGUF_E_FORMAT, err, errlen,
                        "tensor '%s' range overflow", prev->name);
        }
        if (cur_t->offset < prev_end) {
            free(sorted);
            return fail(GGUF_E_FORMAT, err, errlen,
                        "tensor '%s' overlaps '%s'", cur_t->name, prev->name);
        }
    }
    free(sorted);
    return GGUF_OK;
}

gguf_status gguf_open(const char *path, gguf_file *g, char *err, size_t errlen) {
    memset(g, 0, sizeof(*g));
    if (!path || !*path) {
        return fail(GGUF_E_IO, err, errlen, "empty path");
    }

    gguf_status st = map_file(path, &g->map, &g->size, err, errlen);
    if (st != GGUF_OK) return st;

    cur c = {(const uint8_t *)g->map, g->size, 0};

    uint32_t magic;
    if (!rd_u32(&c, &magic)) {
        gguf_close(g);
        return fail(GGUF_E_TRUNC, err, errlen, "file is too small");
    }
    if (magic != 0x46554747u) {
        gguf_close(g);
        return fail(GGUF_E_MAGIC, err, errlen, "not a GGUF file (magic %08" PRIx32 ")",
                    magic);
    }
    if (!rd_u32(&c, &g->version)) {
        gguf_close(g);
        return fail(GGUF_E_TRUNC, err, errlen, "truncated version");
    }
    if (g->version < 2 || g->version > 3) {
        gguf_close(g);
        return fail(GGUF_E_VERSION, err, errlen,
                    "unsupported GGUF version %u (need 2 or 3)", g->version);
    }
    if (!rd_u64(&c, &g->n_tensors) || !rd_u64(&c, &g->n_kv)) {
        gguf_close(g);
        return fail(GGUF_E_TRUNC, err, errlen, "truncated header");
    }
    if (g->n_tensors > 1000000 || g->n_kv > 1000000) {
        gguf_close(g);
        return fail(GGUF_E_FORMAT, err, errlen, "unreasonable header counts");
    }

    if (g->n_kv) {
        g->kv = (gguf_kv *)calloc((size_t)g->n_kv, sizeof(*g->kv));
        if (!g->kv) {
            gguf_close(g);
            return fail(GGUF_E_NOMEM, err, errlen, "out of memory for metadata");
        }
    }
    for (uint64_t i = 0; i < g->n_kv; i++) {
        st = parse_kv(&c, &g->kv[i], err, errlen);
        if (st != GGUF_OK) {
            gguf_close(g);
            return st;
        }
    }

    if (g->n_tensors) {
        g->tensors =
            (gguf_tensor *)calloc((size_t)g->n_tensors, sizeof(*g->tensors));
        if (!g->tensors) {
            gguf_close(g);
            return fail(GGUF_E_NOMEM, err, errlen,
                        "out of memory for tensor table");
        }
    }
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        st = parse_tensor(&c, &g->tensors[i], err, errlen);
        if (st != GGUF_OK) {
            gguf_close(g);
            return st;
        }
    }

    g->alignment = 32;
    const gguf_kv *al = gguf_find_kv(g, "general.alignment");
    if (al) {
        uint64_t a = gguf_kv_u64(al, 0);
        if (a == 0 || (a & (a - 1)) != 0 || a > (1u << 20)) {
            gguf_close(g);
            return fail(GGUF_E_FORMAT, err, errlen,
                        "invalid general.alignment %" PRIu64, a);
        }
        g->alignment = a;
    }

    uint64_t data_off;
    if (!align_up_u64((uint64_t)c.pos, g->alignment, &data_off) ||
        data_off > (uint64_t)g->size) {
        gguf_close(g);
        return fail(GGUF_E_TRUNC, err, errlen,
                    "tensor data offset exceeds file size");
    }
    g->data_offset = data_off;

    st = validate_tensors(g, err, errlen);
    if (st != GGUF_OK) {
        gguf_close(g);
        return st;
    }
    return GGUF_OK;
}

void gguf_close(gguf_file *g) {
    if (!g) return;
    if (g->kv) {
        for (uint64_t i = 0; i < g->n_kv; i++) free(g->kv[i].key);
        free(g->kv);
    }
    if (g->tensors) {
        for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
        free(g->tensors);
    }
    if (g->map) UnmapViewOfFile(g->map);
    memset(g, 0, sizeof(*g));
}

const gguf_kv *gguf_find_kv(const gguf_file *g, const char *key) {
    if (!g || !key) return NULL;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    }
    return NULL;
}

const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name) {
    if (!g || !name) return NULL;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

const char *gguf_kv_str(const gguf_kv *kv) {
    if (!kv || kv->type != GGUF_T_STRING) return NULL;
    return (const char *)kv->v.str.ptr;
}

uint64_t gguf_kv_u64(const gguf_kv *kv, uint64_t def) {
    if (!kv) return def;
    switch (kv->type) {
    case GGUF_T_UINT8:  return kv->v.u8;
    case GGUF_T_UINT16: return kv->v.u16;
    case GGUF_T_UINT32: return kv->v.u32;
    case GGUF_T_UINT64: return kv->v.u64;
    case GGUF_T_BOOL:   return kv->v.b ? 1 : 0;
    case GGUF_T_INT8:   return kv->v.i8 >= 0 ? (uint64_t)kv->v.i8 : def;
    case GGUF_T_INT16:  return kv->v.i16 >= 0 ? (uint64_t)kv->v.i16 : def;
    case GGUF_T_INT32:  return kv->v.i32 >= 0 ? (uint64_t)kv->v.i32 : def;
    case GGUF_T_INT64:  return kv->v.i64 >= 0 ? (uint64_t)kv->v.i64 : def;
    default:            return def;
    }
}

double gguf_kv_f64(const gguf_kv *kv, double def) {
    if (!kv) return def;
    switch (kv->type) {
    case GGUF_T_FLOAT32: return (double)kv->v.f32;
    case GGUF_T_FLOAT64: return kv->v.f64;
    default: {
        uint64_t u = gguf_kv_u64(kv, UINT64_MAX);
        return u == UINT64_MAX ? def : (double)u;
    }
    }
}
