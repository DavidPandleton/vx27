#ifndef QL_TOKENIZER_H
#define QL_TOKENIZER_H

#include "gguf.h"

#include <stddef.h>
#include <stdint.h>

typedef struct tokenizer tokenizer;

int tok_load(tokenizer **out, const gguf_file *g, char *err, size_t errlen);
/* Muat dari 3 file binari tools/make_tokbin.py (tanpa GGUF). */
int tok_load_bin(tokenizer **out, const char *vocab_path, const char *types_path,
                 const char *merges_path, char *err, size_t errlen);
void tok_free(tokenizer *t);
/* Allocates *ids (caller frees). Returns 0 on success, -1 on error. */
int tok_encode(const tokenizer *t, const uint8_t *text, size_t len,
               uint32_t **ids, size_t *n_ids, char *err, size_t errlen);

/* Allocates *bytes (caller frees). Returns 0 on success, -1 on error. */
int tok_decode(const tokenizer *t, const uint32_t *ids, size_t n,
               uint8_t **bytes, size_t *n_bytes, char *err, size_t errlen);

uint32_t tok_special_id(const tokenizer *t, const char *name);
const uint8_t *tok_token_bytes(const tokenizer *t, uint32_t id, size_t *len);
size_t tok_vocab_size(const tokenizer *t);
size_t tok_special_count(const tokenizer *t);

#endif /* QL_TOKENIZER_H */
