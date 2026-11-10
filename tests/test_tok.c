#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/tokenizer.h"

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else { printf("ok: %s\n", msg); } } while (0)

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

int main(void) {
    char err[512] = {0};
    tokenizer *t = NULL;
    CHECK(tok_load_bin(&t, "build/tok_vocab.bin", "build/tok_types.bin",
                       "build/tok_merges.bin", err, sizeof(err)) == 0, "load 248320");
    if (!t) { printf("  err: %s\n", err); return 1; }
    CHECK(tok_vocab_size(t) == 248320, "vocab 248320");
    CHECK(tok_special_id(t, "<|im_start|>") == 248045, "im_start=248045");
    CHECK(tok_special_id(t, "<|im_end|>") == 248046, "im_end=248046");
    CHECK(tok_special_id(t, "<|endoftext|>") == 248044, "endoftext=248044");
    CHECK(tok_special_id(t, "<|no_such_token_xyz|>") == 0xFFFFFFFFu, "unknown special -> UINT32_MAX");

    /* golden oracle (lib tokenizers resmi): hex utf8 + daftar id */
    FILE *f = fopen("build/tok_golden.txt", "r");
    CHECK(f != NULL, "buka golden");
    int ngood = 0, ncase = 0;
    char hx[8192], ids[65536];
    while (f && fgets(hx, sizeof hx, f) && fgets(ids, sizeof ids, f)) {
        ncase++;
        size_t hl = strlen(hx);
        while (hl && (hx[hl-1] == '\n' || hx[hl-1] == '\r')) hx[--hl] = 0;
        uint8_t *txt = malloc(hl / 2 + 1);
        size_t tl = 0;
        for (size_t i = 0; i + 1 < hl; i += 2) {
            int a = hexval(hx[i]), b = hexval(hx[i+1]);
            if (a < 0 || b < 0) break;
            txt[tl++] = (uint8_t)(a * 16 + b);
        }
        /* parse ids harapan (bisa kosong) */
        uint32_t want[4096]; int nw = 0;
        for (char *p = ids; *p && nw < 4096; ) {
            while (*p == ' ' || *p == '\n' || *p == '\r') p++;
            if (!*p) break;
            want[nw++] = (uint32_t)strtoul(p, &p, 10);
        }
        uint32_t *got = NULL; size_t ng = 0;
        int rc = tok_encode(t, txt, tl, &got, &ng, err, sizeof(err));
        int same = (rc == 0 && (int)ng == nw);
        if (same) for (int i = 0; i < nw; i++) if (got[i] != want[i]) { same = 0; break; }
        if (!same) {
            printf("  MISMATCH case %d (len %zu): got %zu toks, want %d\n", ncase, tl, ng, nw);
            if (ng > 0) { printf("    got:"); for (size_t i = 0; i < ng && i < 12; i++) printf(" %u", got[i]); printf("\n"); }
            if (nw > 0) { printf("    want:"); for (int i = 0; i < nw && i < 12; i++) printf(" %u", want[i]); printf("\n"); }
            fails++;
        } else {
            /* roundtrip decode == bytes asli */
            uint8_t *dec = NULL; size_t dl = 0;
            if (tok_decode(t, got, ng, &dec, &dl, err, sizeof(err)) != 0 || dl != tl || memcmp(dec, txt, tl) != 0) {
                printf("  ROUNDTRIP FAIL case %d\n", ncase);
                fails++;
            } else ngood++;
            free(dec);
        }
        free(got); free(txt);
    }
    if (f) fclose(f);
    printf("  golden: %d/%d cocok + roundtrip\n", ngood, ncase);
    CHECK(ncase == 14 && ngood == ncase, "14/14 golden + roundtrip");

    /* id padding: decode jadi kosong, encode tak pernah hasilkan (sudah tercakup golden) */
    uint8_t *d0 = NULL; size_t dl0 = 1;
    uint32_t pad = 248100;
    if (tok_decode(t, &pad, 1, &d0, &dl0, err, sizeof(err)) == 0 && dl0 == 0) {
        printf("  ok: padding id decode kosong\n");
    } else {
        printf("  FAIL: padding id\n"); fails++;
    }
    free(d0);
    tok_free(t);
    if (fails) { printf("test_tok: %d FAIL\n", fails); return 1; }
    printf("test_tok: PASS\n");
    return 0;
}
