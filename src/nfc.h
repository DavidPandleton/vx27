#ifndef QL_NFC_H
#define QL_NFC_H

#include <stddef.h>
#include <stdint.h>

/* Allocates *out (caller frees). Returns 0 on success, -1 on error. */
int nfc_normalize(const uint8_t *in, size_t len, uint8_t **out, size_t *outlen,
                  char *err, size_t errlen);

#endif /* QL_NFC_H */
