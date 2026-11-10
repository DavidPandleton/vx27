#include "tokenizer.h"
#include "../src/quant.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Muat tokenizer Qwen3.8 dari 3 file binari (tools/make_tokbin.py):
 *   vocab:  u32 n; n x [u32 len][bytes]  (string GPT-2-mapped, id = urutan)
 *   types:  u32 n; n x int32             (1=normal, 3=special, 2=unknown/padding)
 *   merges: u32 n; n x [u32 len][bytes]  ("kiri kanan")
 * Dibungkus jadi gguf_kv palsu agar tok_load() yg proven dipakai apa adanya.
 */
static uint8_t *read_file(const char *path, size_t *out_n, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "cannot open %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (1LL << 31)) { fclose(f); snprintf(err, errlen, "bad size %s", path); return NULL; }
    uint8_t *b = malloc((size_t)sz);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); snprintf(err, errlen, "cannot read %s", path); return NULL; }
    fclose(f);
    *out_n = (size_t)sz;
    return b;
}

int tok_load_bin(tokenizer **out, const char *vocab_path, const char *types_path,
                 const char *merges_path, char *err, size_t errlen) {
    *out = NULL;
    size_t vn = 0, tn = 0, mn = 0;
    uint8_t *vb = read_file(vocab_path, &vn, err, errlen);
    if (!vb) return -1;
    uint8_t *tb = read_file(types_path, &tn, err, errlen);
    if (!tb) { free(vb); return -1; }
    uint8_t *mb = read_file(merges_path, &mn, err, errlen);
    if (!mb) { free(vb); free(tb); return -1; }
    int rc = -1;
    uint32_t nv = 0, nm = 0;
    if (vn < 4) { snprintf(err, errlen, "trunc vocab"); goto done; }
    memcpy(&nv, vb, 4);
    if (nv == 0 || nv > 4u * 1024u * 1024u) { snprintf(err, errlen, "bad vocab count"); goto done; }
    /* bangun buffer array ala GGUF: tiap string = [u64 len][bytes] */
    uint8_t *tbuf = NULL, *mbuf = NULL;
    int32_t *tyv = NULL;
    {
        /* hitung dulu total ukuran dgn sekali jalan */
        const uint8_t *p = vb + 4; uint64_t need_t = 0; uint32_t cnt = 0;
        while (cnt < nv) {
            if ((size_t)(p - vb) + 4 > vn) break;
            uint32_t L; memcpy(&L, p, 4); p += 4;
            if ((size_t)(p - vb) + L > vn) break;
            p += L; need_t += 8 + L; cnt++;
        }
        if (cnt != nv) { snprintf(err, errlen, "trunc vocab entries"); goto done; }
        if (tn != 4 + (size_t)nv * 4) { snprintf(err, errlen, "types count mismatch"); goto done; }
        p = mb;
        if (mn < 4) { snprintf(err, errlen, "trunc merges"); goto done; }
        memcpy(&nm, mb, 4); p = mb + 4;
        if (nm > 4u * 1024u * 1024u) { snprintf(err, errlen, "bad merges count"); goto done; }
        uint64_t need_m = 0; uint32_t mc = 0;
        while (mc < nm) {
            if ((size_t)(p - mb) + 4 > mn) break;
            uint32_t L; memcpy(&L, p, 4); p += 4;
            if ((size_t)(p - mb) + L > mn) break;
            p += L; need_m += 8 + L; mc++;
        }
        if (mc != nm) { snprintf(err, errlen, "trunc merge entries"); goto done; }
        tbuf = malloc(need_t ? need_t : 1);
        mbuf = malloc(need_m ? need_m : 1);
        tyv = malloc(sizeof(*tyv) * nv);
        if (!tbuf || !mbuf || !tyv) { free(tbuf); free(mbuf); free(tyv); snprintf(err, errlen, "out of memory"); goto done; }
        memcpy(tyv, tb + 4, sizeof(*tyv) * nv);
        /* isi ulang buffer array */
        p = vb + 4;
        uint8_t *q = tbuf;
        for (uint32_t i = 0; i < nv; i++) {
            uint32_t L; memcpy(&L, p, 4); p += 4;
            uint64_t LL = L; memcpy(q, &LL, 8); q += 8;
            memcpy(q, p, L); p += L; q += L;
        }
        p = mb + 4; q = mbuf;
        for (uint32_t i = 0; i < nm; i++) {
            uint32_t L; memcpy(&L, p, 4); p += 4;
            uint64_t LL = L; memcpy(q, &LL, 8); q += 8;
            memcpy(q, p, L); p += L; q += L;
        }
        gguf_kv kv[3];
        memset(kv, 0, sizeof kv);
        kv[0].key = "tokenizer.ggml.tokens"; kv[0].type = 9;
        kv[0].v.arr.elem_type = 8; kv[0].v.arr.count = nv;
        kv[0].v.arr.data = tbuf; kv[0].v.arr.end = tbuf + need_t;
        kv[1].key = "tokenizer.ggml.merges"; kv[1].type = 9;
        kv[1].v.arr.elem_type = 8; kv[1].v.arr.count = nm;
        kv[1].v.arr.data = mbuf; kv[1].v.arr.end = mbuf + need_m;
        kv[2].key = "tokenizer.ggml.token_type"; kv[2].type = 9;
        kv[2].v.arr.elem_type = 5; kv[2].v.arr.count = nv;
        kv[2].v.arr.data = (const uint8_t *)tyv;
        kv[2].v.arr.end = (const uint8_t *)(tyv + nv);
        gguf_file g;
        memset(&g, 0, sizeof g);
        g.kv = kv; g.n_kv = 3;
        rc = tok_load(out, &g, err, errlen);
        free(tbuf); free(mbuf); free(tyv);
    }
done:
    free(vb); free(tb); free(mb);
    return rc;
}
