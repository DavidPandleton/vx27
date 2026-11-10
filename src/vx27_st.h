#ifndef VX27_ST_H
#define VX27_ST_H
#include <stddef.h>
#include <stdint.h>

// Minimal safetensors header reader (no JSON lib).
// Format: u64 LE header_len, then UTF-8 JSON dict {tensor: {dtype, shape, data_offsets}}.
// We scan for dtype/shape/data_offsets without a full JSON parser.

typedef struct {
    char name[256];
    char dtype[16];
    uint64_t elems;      // product(shape)
    uint64_t s0, s1;     // shape[0], shape[1] or 1
    uint64_t offset0, offset1; // data_offsets
    uint64_t bytes;      // elems * dtype_size
} vx27_st_entry;

uint64_t vx27_st_u64le(const uint8_t *p);
int vx27_dtype_size(const char *dtype); // F32->4 F16/BF16->2 U8/I8/BOOL->1 F64->8, -1 unknown
int vx27_st_parse(const uint8_t *hdr, size_t hdr_len,
                  vx27_st_entry *out, int cap, int *n_out, char *err, size_t errlen);
// Full validator: checks bytes==elems*ds, bytes==off1-off0, offsets in-bounds,
// sorted non-overlapping, no duplicate names. file_data = file_size-8-header_len.
int vx27_st_validate(const vx27_st_entry *ents, int n, uint64_t file_data,
                     char *err, size_t errlen);
int vx27_st_open(const char *path, vx27_st_entry *out, int cap, int *n_out,
                 uint64_t *data_bytes, char *err, size_t errlen);

#endif
