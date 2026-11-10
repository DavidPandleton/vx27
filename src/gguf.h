#ifndef QL_GGUF_H
#define QL_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* GGUF metadata value types (spec v3). */
enum {
    GGUF_T_UINT8 = 0,
    GGUF_T_INT8 = 1,
    GGUF_T_UINT16 = 2,
    GGUF_T_INT16 = 3,
    GGUF_T_UINT32 = 4,
    GGUF_T_INT32 = 5,
    GGUF_T_FLOAT32 = 6,
    GGUF_T_BOOL = 7,
    GGUF_T_STRING = 8,
    GGUF_T_ARRAY = 9,
    GGUF_T_UINT64 = 10,
    GGUF_T_INT64 = 11,
    GGUF_T_FLOAT64 = 12,
    GGUF_T_COUNT = 13
};

typedef enum {
    GGUF_OK = 0,
    GGUF_E_IO,
    GGUF_E_TRUNC,
    GGUF_E_MAGIC,
    GGUF_E_VERSION,
    GGUF_E_NOMEM,
    GGUF_E_FORMAT,
    GGUF_E_UNSUPPORTED
} gguf_status;

typedef struct {
    char *key;
    uint32_t type;
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        float f32;
        uint32_t u32;
        int32_t i32;
        uint16_t u16;
        int16_t i16;
        uint8_t u8;
        int8_t i8;
        bool b;
        struct {
            const uint8_t *ptr;
            uint64_t len;
        } str;
        struct {
            uint32_t elem_type;
            uint64_t count;
            const uint8_t *data;
            const uint8_t *end;
        } arr;
    } v;
} gguf_kv;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;
    uint64_t n_elements;
    uint64_t n_bytes;
} gguf_tensor;

typedef struct {
    void *map;
    size_t size;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    gguf_kv *kv;
    gguf_tensor *tensors;
    uint64_t alignment;
    uint64_t data_offset;
    uint64_t data_size;
} gguf_file;

gguf_status gguf_open(const char *path, gguf_file *g, char *err, size_t errlen);
void gguf_close(gguf_file *g);

const gguf_kv *gguf_find_kv(const gguf_file *g, const char *key);
const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name);

const char *gguf_kv_str(const gguf_kv *kv);
uint64_t gguf_kv_u64(const gguf_kv *kv, uint64_t def);
double gguf_kv_f64(const gguf_kv *kv, double def);

uint64_t gguf_type_block(uint32_t type);
uint64_t gguf_type_bytes(uint32_t type);
const char *gguf_type_name(uint32_t type);

#endif /* QL_GGUF_H */
