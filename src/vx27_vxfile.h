#ifndef VX27_VXFILE_H
#define VX27_VXFILE_H
#include <stdint.h>
#include <stddef.h>

// .vx27 container v0 (nyata, bukan dry-run):
// magic "VX27" + u32 version(1) + u32 ntensors + entries + blob.
// Entry: name[64], dtype u32 (0=F32,1=F16,2=Q8_K,3=Q4_K), ne0, ne1 (u64), offset (u64), bytes (u64).
// v0 stores Q8_K (nyata via quantize_q8_k) and F32/F16 passthrough.
// Bit-planar 3-bit custom = v1 future (format reserves dtype 4/5).
// Split: core L0-51 -> vx27_core.vx27, head L52-63+lm+mtp -> vx27_head.vx27.

#define VX27_MAGIC "VX27"
#define VX27_VERSION 1u
#define VX27_D_F32 0u
#define VX27_D_F16 1u
#define VX27_D_Q8  2u
#define VX27_D_Q4K 3u
#define VX27_D_BP3 4u  // bit-planar 3-bit v1: grup 128 + FP16 skala (50B/128)

typedef struct {
    char name[64];
    uint32_t dtype;
    uint64_t ne0, ne1;
    uint64_t offset, bytes;
} vx27_entry;

int vx27_write(const char *path,
               const char **names, const uint32_t *dtypes,
               const uint64_t *ne0, const uint64_t *ne1,
               const void **blobs, const uint64_t *blob_bytes,
               int n, char *err, size_t errlen);
int vx27_list(const char *path, vx27_entry *out, int cap, int *n_out,
              char *err, size_t errlen);
int vx27_plan_14gb(int *ok_core, int *ok_head, char *msg, size_t msglen);

#endif
