#include "vx27_vxfile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int vx27_write(const char *path,
               const char **names, const uint32_t *dtypes,
               const uint64_t *ne0, const uint64_t *ne1,
               const void **blobs, const uint64_t *blob_bytes,
               int n, char *err, size_t errlen) {
    if (n < 0 || n > 4096) { if (err) snprintf(err, errlen, "bad n"); return -1; }
    FILE *f = fopen(path, "wb");
    if (!f) { if (err) snprintf(err, errlen, "cannot create %s", path); return -1; }
    fwrite(VX27_MAGIC, 1, 4, f);
    uint32_t ver = VX27_VERSION; fwrite(&ver, 4, 1, f);
    uint32_t nn = (uint32_t)n; fwrite(&nn, 4, 1, f);
    uint64_t blob_off = 0;
    for (int i = 0; i < n; i++) {
        vx27_entry e; memset(&e, 0, sizeof(e));
        snprintf(e.name, sizeof(e.name), "%s", names[i]);
        e.dtype = dtypes[i]; e.ne0 = ne0[i]; e.ne1 = ne1[i];
        e.offset = blob_off; e.bytes = blob_bytes[i];
        fwrite(e.name, 1, 64, f);
        fwrite(&e.dtype, 4, 1, f);
        fwrite(&e.ne0, 8, 1, f); fwrite(&e.ne1, 8, 1, f);
        fwrite(&e.offset, 8, 1, f); fwrite(&e.bytes, 8, 1, f);
        uint32_t rsv = 0; fwrite(&rsv, 4, 1, f);
        blob_off += blob_bytes[i];
    }
    for (int i = 0; i < n; i++)
        if (blob_bytes[i] && fwrite(blobs[i], 1, (size_t)blob_bytes[i], f) != (size_t)blob_bytes[i]) {
            fclose(f); if (err) snprintf(err, errlen, "write blob %d", i); return -1;
        }
    fclose(f);
    return 0;
}

int vx27_list(const char *path, vx27_entry *out, int cap, int *n_out,
              char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) { if (err) snprintf(err, errlen, "cannot open %s", path); return -1; }
    char magic[4]; uint32_t ver = 0, n = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, VX27_MAGIC, 4) != 0) { fclose(f); if (err) snprintf(err, errlen, "bad magic"); return -1; }
    if (fread(&ver, 4, 1, f) != 1 || ver != VX27_VERSION) { fclose(f); if (err) snprintf(err, errlen, "bad version"); return -1; }
    if (fread(&n, 4, 1, f) != 1 || n > (uint32_t)cap) { fclose(f); if (err) snprintf(err, errlen, "bad n=%u", n); return -1; }
    if (n == 0) { fclose(f); *n_out = 0; return 0; }
    uint64_t expect_off = 0;
    for (uint32_t i = 0; i < n; i++) {
        vx27_entry e;
        if (fread(e.name, 1, 64, f) != 64) { fclose(f); if (err) snprintf(err, errlen, "trunc entry"); return -1; }
        e.name[63] = 0;
        if (fread(&e.dtype, 4, 1, f) != 1) { fclose(f); return -1; }
        if (fread(&e.ne0, 8, 1, f) != 1 || fread(&e.ne1, 8, 1, f) != 1) { fclose(f); return -1; }
        if (fread(&e.offset, 8, 1, f) != 1 || fread(&e.bytes, 8, 1, f) != 1) { fclose(f); return -1; }
        uint32_t rsv = 0; if (fread(&rsv, 4, 1, f) != 1) { fclose(f); return -1; }
        if (e.dtype > VX27_D_BP3) { fclose(f); if (err) snprintf(err, errlen, "tensor %s bad dtype %u", e.name, e.dtype); return -1; }
        if (e.offset != expect_off) { fclose(f); if (err) snprintf(err, errlen, "tensor %s gap/overlap", e.name); return -1; }
        expect_off += e.bytes;
        out[i] = e;
    }
    fclose(f);
    *n_out = (int)n;
    return 0;
}

// Validasi aritmetika 14GB tanpa alokasi: core 18.13B params, head 6.84B.
// core<=11.2GB => <=4.94 bit; head<=2.8GB => <=3.27 bit. Q8(8bit) OVER, Q4 FIT core, head perlu Q3.
int vx27_plan_14gb(int *ok_core, int *ok_head, char *msg, size_t msglen) {
    double core_p = 18.13e9, head_p = 6.842e9;
    double core_q4 = core_p * 4 / 8 / 1e9;   // 9.06GB
    double head_q4 = head_p * 4 / 8 / 1e9;   // 3.42GB OVER 2.8
    double head_q3 = head_p * 3 / 8 / 1e9;   // 2.57GB FIT
    int oc = core_q4 <= 11.2, oh = head_q3 <= 2.8;
    if (ok_core) *ok_core = oc;
    if (ok_head) *ok_head = oh;
    if (msg) snprintf(msg, msglen,
        "core Q4 %.2fGB %s | head Q4 %.2fGB OVER -> Q3 %.2fGB %s",
        core_q4, oc ? "FIT" : "OVER", head_q4, head_q3, oh ? "FIT" : "OVER");
    return (oc && oh) ? 0 : 1;
}
