#include "vx27_mmap.h"
#include <windows.h>
#include <stdio.h>
#include <string.h>

int vx27_mmap_open(const char *path, vx27_map *m, char *err, size_t errlen) {
    memset(m, 0, sizeof(*m));
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hf == INVALID_HANDLE_VALUE) { if (err) snprintf(err, errlen, "open %s", path); return -1; }
    LARGE_INTEGER sz; if (!GetFileSizeEx(hf, &sz) || sz.QuadPart < 12) { CloseHandle(hf); return -1; }
    HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hm) { CloseHandle(hf); return -1; }
    void *base = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!base) { CloseHandle(hm); CloseHandle(hf); return -1; }
    m->base = base; m->size = (size_t)sz.QuadPart;
    m->hFile = hf; m->hMap = hm;
    if (memcmp(base, "VX27", 4) != 0) { if (err) snprintf(err, errlen, "bad magic"); vx27_mmap_close(m); return -1; }
    return 0;
}
void vx27_mmap_close(vx27_map *m) {
    if (m->base) UnmapViewOfFile(m->base);
    if (m->hMap) CloseHandle((HANDLE)m->hMap);
    if (m->hFile) CloseHandle((HANDLE)m->hFile);
    memset(m, 0, sizeof(*m));
}
const void *vx27_mmap_blob(const vx27_map *m, int idx, uint64_t *bytes_out) {
    if (!m->base || idx < 0) return NULL;
    uint8_t *b = (uint8_t*)m->base;
    uint32_t n = 0; memcpy(&n, b + 8, 4);
    if ((uint32_t)idx >= n) return NULL;
    // entry 104B: name64 dtype4 ne0(8) ne1(8) off(8) bytes(8) rsv4
    uint8_t *e = b + 12 + (size_t)idx * 104;
    uint64_t off = 0, nb = 0, ne0 = 0, ne1 = 0;
    memcpy(&off, e + 64 + 4 + 8 + 8, 8);
    memcpy(&nb, e + 64 + 4 + 8 + 8 + 8, 8);
    (void)ne0; (void)ne1;
    uint64_t hdr = 12 + (uint64_t)n * 104;
    if (hdr + off + nb > m->size) return NULL;
    if (bytes_out) *bytes_out = nb;
    return b + hdr + off;
}
