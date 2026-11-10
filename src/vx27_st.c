#include "vx27_st.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t vx27_st_u64le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)p[i]) << (8 * i);
    return v;
}

int vx27_dtype_size(const char *dtype) {
    if (!strcmp(dtype, "F32")) return 4;
    if (!strcmp(dtype, "F16") || !strcmp(dtype, "BF16")) return 2;
    if (!strcmp(dtype, "U8") || !strcmp(dtype, "I8") || !strcmp(dtype, "BOOL")) return 1;
    if (!strcmp(dtype, "F64")) return 8;
    if (!strcmp(dtype, "I32") || !strcmp(dtype, "U32")) return 4;
    if (!strcmp(dtype, "I64") || !strcmp(dtype, "U64")) return 8;
    return -1;
}

// Very small scanner: finds "name":{..."dtype":"X","shape":[...],"data_offsets":[a,b]}
int vx27_st_parse(const uint8_t *hdr, size_t hdr_len,
                  vx27_st_entry *out, int cap, int *n_out, char *err, size_t errlen) {
    int n = 0;
    size_t i = 0;
    // copy to NUL-terminated scratch for strstr/sscanf
    char *js = (char *)malloc(hdr_len + 1);
    if (!js) { if (err) snprintf(err, errlen, "oom"); return -1; }
    memcpy(js, hdr, hdr_len); js[hdr_len] = 0;

    char *p = js;
    while (*p && n < cap) {
        // find next "\"...\" : {" candidate with dtype inside
        char *dtype_key = strstr(p, "\"dtype\"");
        if (!dtype_key) break;
        // backtrack to tensor name: nearest preceding "\"name\""
        // Simplification: name = nearest quote-pair before dtype_key on same object.
        // We search backwards for '\"' pairs.
        char *obj_start = dtype_key;
        // find tensor name by scanning from p to dtype_key for last quoted string before '{'
        char *scan = p, *last_q = NULL;
        while (scan < dtype_key) {
            if (*scan == '"') last_q = scan;
            scan++;
        }
        // last_q points at closing quote of... actually need opening; heuristic:
        // walk back from dtype_key to find "\",\" pattern is fragile, so instead
        // find the nearest '"<name>":{' before dtype_key.
        char *name_end = NULL, *name_start = NULL;
        // search backwards for ':' then quotes
        // find opening brace of this tensor object
        char *brace = NULL;
        for (char *q = dtype_key; q > p; q--) { if (*q == '{') { brace = q; break; } }
        if (!brace) { p = dtype_key + 7; continue; }
        // name is "xxx" immediately before brace's key: find quote before brace going back past ':' and spaces
        char *q2 = brace - 1;
        while (q2 > p && (*q2 == ' ' || *q2 == '\t' || *q2 == '\n' || *q2 == '\r' || *q2 == ':')) q2--;
        if (q2 > p && *q2 == '"') {
            name_end = q2;
            char *q3 = q2 - 1;
            while (q3 > p && *q3 != '"') q3--;
            if (*q3 == '"') { name_start = q3 + 1; }
        }
        (void)last_q; (void)obj_start;

        char dtype[16] = {0};
        if (sscanf(dtype_key, "\"dtype\":\"%15[^\"]\"", dtype) != 1) { p = dtype_key + 7; continue; }

        char *shape_key = strstr(dtype_key, "\"shape\"");
        char *off_key = strstr(dtype_key, "\"data_offsets\"");
        if (!shape_key || !off_key || off_key < dtype_key) { p = dtype_key + 7; continue; }
        // only accept keys belonging to same object: next '{' after brace shouldn't come before them
        // (cheap guard: distance < 4096)
        if ((size_t)(shape_key - brace) > 4096 || (size_t)(off_key - brace) > 4096) { p = dtype_key + 7; continue; }

        // parse shape [...] product + first two dims
        char *br = strchr(shape_key, '[');
        char *br_end = br ? strchr(br, ']') : NULL;
        if (!br || !br_end) { p = dtype_key + 7; continue; }
        uint64_t elems = 1;
        uint64_t dims[4] = {1,1,1,1}; int nd = 0;
        int empty = 1;
        char *num = br + 1;
        while (num < br_end) {
            while (num < br_end && (*num == ' ' || *num == ',')) num++;
            if (num >= br_end) break;
            char *e = NULL;
            unsigned long long v = strtoull(num, &e, 10);
            if (e == num) break;
            empty = 0;
            elems *= v;
            if (nd < 4) dims[nd++] = v;
            num = e;
        }
        if (empty) elems = 1; // scalar edge (shouldn't happen for weights)

        unsigned long long o0 = 0, o1 = 0;
        if (sscanf(off_key, "\"data_offsets\":[%llu,%llu]", &o0, &o1) != 2) { p = dtype_key + 7; continue; }

        int ds = vx27_dtype_size(dtype);
        if (ds < 0) { p = dtype_key + 7; continue; }

        vx27_st_entry *e = &out[n];
        if (name_start && name_end && name_end > name_start) {
            size_t nl = (size_t)(name_end - name_start);
            if (nl > sizeof(e->name) - 1) nl = sizeof(e->name) - 1;
            memcpy(e->name, name_start, nl); e->name[nl] = 0;
        } else {
            snprintf(e->name, sizeof(e->name), "tensor_%d", n);
        }
        snprintf(e->dtype, sizeof(e->dtype), "%s", dtype);
        e->elems = elems;
        e->s0 = dims[0]; e->s1 = nd > 1 ? dims[1] : 1;
        e->offset0 = (uint64_t)o0; e->offset1 = (uint64_t)o1;
        e->bytes = elems * (uint64_t)ds;
        n++;
        p = off_key + 14;
    }
    free(js);
    (void)i;
    *n_out = n;
    return 0;
}

int vx27_st_open(const char *path, vx27_st_entry *out, int cap, int *n_out,
                 uint64_t *data_bytes, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) { if (err) snprintf(err, errlen, "cannot open %s", path); return -1; }
    uint8_t lenb[8];
    if (fread(lenb, 1, 8, f) != 8) { fclose(f); if (err) snprintf(err, errlen, "truncated header len"); return -1; }
    uint64_t hl = vx27_st_u64le(lenb);
    if (hl > (1ULL << 30)) { fclose(f); if (err) snprintf(err, errlen, "header too big %llu", (unsigned long long)hl); return -1; }
    // file size for bounds check (64-bit: shard >2GB overflow ftell/long)
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); if (err) snprintf(err, errlen, "seek fail"); return -1; }
    __int64 fsize = _ftelli64(f);
    if (fsize < 0 || (uint64_t)fsize < 8 + hl) { fclose(f); if (err) snprintf(err, errlen, "file smaller than header"); return -1; }
    uint64_t file_data = (uint64_t)fsize - 8 - hl;
    if (_fseeki64(f, 8, SEEK_SET) != 0) { fclose(f); if (err) snprintf(err, errlen, "seek fail"); return -1; }
#else
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); if (err) snprintf(err, errlen, "seek fail"); return -1; }
    long fsize = ftell(f);
    if (fsize < 0 || (uint64_t)fsize < 8 + hl) { fclose(f); if (err) snprintf(err, errlen, "file smaller than header"); return -1; }
    uint64_t file_data = (uint64_t)fsize - 8 - hl;
    if (fseek(f, 8, SEEK_SET) != 0) { fclose(f); if (err) snprintf(err, errlen, "seek fail"); return -1; }
#endif
    uint8_t *hdr = (uint8_t *)malloc((size_t)hl);
    if (!hdr) { fclose(f); if (err) snprintf(err, errlen, "oom"); return -1; }
    if (fread(hdr, 1, (size_t)hl, f) != (size_t)hl) { free(hdr); fclose(f); if (err) snprintf(err, errlen, "truncated header"); return -1; }
    fclose(f);
    int rc = vx27_st_parse(hdr, (size_t)hl, out, cap, n_out, err, errlen);
    free(hdr);
    if (rc != 0) return rc;
    if (vx27_st_validate(out, *n_out, file_data, err, errlen) != 0) return -1;
    uint64_t total = 0;
    for (int i = 0; i < *n_out; i++) total += out[i].bytes;
    if (data_bytes) *data_bytes = total;
    return 0;
}

int vx27_st_validate(const vx27_st_entry *ents, int n, uint64_t file_data,
                     char *err, size_t errlen) {
    if (n <= 0) { if (err) snprintf(err, errlen, "no tensors"); return -1; }
    uint64_t total = 0, max_end = 0;
    for (int i = 0; i < n; i++) {
        const vx27_st_entry *e = &ents[i];
        int ds = vx27_dtype_size(e->dtype);
        if (ds < 0) { if (err) snprintf(err, errlen, "tensor %s bad dtype %s", e->name, e->dtype); return -1; }
        if (e->bytes != e->elems * (uint64_t)ds) { if (err) snprintf(err, errlen, "tensor %s bytes %llu != elems %llu * %d", e->name, (unsigned long long)e->bytes, (unsigned long long)e->elems, ds); return -1; }
        if (e->offset1 <= e->offset0) { if (err) snprintf(err, errlen, "tensor %s empty/inverted offsets", e->name); return -1; }
        if (e->offset1 - e->offset0 != e->bytes) { if (err) snprintf(err, errlen, "tensor %s offsets [%llu,%llu) != %llu bytes", e->name, (unsigned long long)e->offset0, (unsigned long long)e->offset1, (unsigned long long)e->bytes); return -1; }
        if (e->offset1 > file_data) { if (err) snprintf(err, errlen, "tensor %s end %llu beyond data %llu", e->name, (unsigned long long)e->offset1, (unsigned long long)file_data); return -1; }
        for (int j = 0; j < i; j++) {
            if (!strcmp(ents[j].name, e->name)) { if (err) snprintf(err, errlen, "duplicate tensor %s", e->name); return -1; }
            // overlap check (intervals must not intersect)
            uint64_t a0 = ents[j].offset0, a1 = ents[j].offset1;
            if (!(e->offset1 <= a0 || e->offset0 >= a1)) { if (err) snprintf(err, errlen, "overlap %s vs %s", e->name, ents[j].name); return -1; }
        }
        total += e->bytes;
        if (e->offset1 > max_end) max_end = e->offset1;
    }
    if (total != max_end) { if (err) snprintf(err, errlen, "gap: total %llu != max_end %llu", (unsigned long long)total, (unsigned long long)max_end); return -1; }
    if (max_end != file_data) { if (err) snprintf(err, errlen, "trailing data: max_end %llu != file_data %llu", (unsigned long long)max_end, (unsigned long long)file_data); return -1; }
    return 0;
}
