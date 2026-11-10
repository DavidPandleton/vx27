#ifndef VX27_MMAP_H
#define VX27_MMAP_H
#include <stddef.h>
#include <stdint.h>

typedef struct { void *base; size_t size; void *hFile; void *hMap; } vx27_map;
int vx27_mmap_open(const char *path, vx27_map *m, char *err, size_t errlen);
void vx27_mmap_close(vx27_map *m);
// Blob pointer for entry index (entries follow 12B header, each 96B).
const void *vx27_mmap_blob(const vx27_map *m, int idx, uint64_t *bytes_out);

#endif
