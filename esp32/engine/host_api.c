/* Thin host-side API used by Python (ctypes) and the host benchmark. */
#include "microcoder.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct { mc_model m; mc_state s; mc_tok t; uint8_t *mb, *tb; void *mem; } mc_host;

static uint8_t *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n); if (fread(b, 1, *n, f) != *n) { fclose(f); free(b); return NULL; }
    fclose(f); return b;
}

mc_host *mch_open(const char *model_path, const char *tok_path) {
    mc_host *h = calloc(1, sizeof *h); size_t n;
    h->mb = slurp(model_path, &n);
    if (!h->mb || mc_load(&h->m, h->mb, n)) return NULL;
    if (tok_path) { size_t k; h->tb = slurp(tok_path, &k); if (!h->tb || mc_tok_load(&h->t, h->tb, k)) return NULL; }
    h->mem = malloc(mc_state_bytes(&h->m));
    mc_state_init(&h->m, &h->s, h->mem);
    return h;
}
float *mch_forward(mc_host *h, int tok, int pos) { return mc_forward(&h->m, &h->s, tok, pos); }
int mch_vocab(mc_host *h) { return h->m.vocab; }
long mch_weight_bytes(mc_host *h) { return (long)h->m.weight_bytes; }
long mch_state_bytes(mc_host *h) { return (long)mc_state_bytes(&h->m); }
int mch_encode(mc_host *h, const char *s, int *ids, int max) { return mc_tok_encode(&h->t, s, ids, max); }
