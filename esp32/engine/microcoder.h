/* MicroCoder INT8 inference engine — portable C99, no dynamic allocation
 * inside the decode loop, weights read in place from a const blob (so on an
 * ESP32 they can stay memory-mapped in flash).
 *
 * Weight-only INT8 quantisation (per-output-row symmetric scales), float32
 * activations and float32 KV cache.
 */
#ifndef MICROCODER_H
#define MICROCODER_H
#include <stddef.h>
#include <stdint.h>

#define MC_MAGIC 0x3143434Du /* "MCC1" */

typedef struct {
    const int8_t *w;     /* rows x cols */
    const float *scale;  /* rows */
    int rows, cols;
} mc_qmat;

typedef struct {
    const float *n1, *n2;
    mc_qmat qkv, o, up, down;
} mc_layer;

typedef struct {
    int vocab, d, layers, heads, ff, ctx;
    mc_qmat emb;          /* vocab x d, also the tied output projection */
    const float *nf;
    mc_layer layer[16];
    size_t weight_bytes;
} mc_model;

typedef struct {
    float *x, *xb, *xb2, *hb, *q, *att, *logits; /* activations */
    float *kcache, *vcache;                       /* layers x ctx x d */
    const float *cos_t, *sin_t;                   /* ctx x hd/2 */
    float *rope;                                  /* storage for the tables */
} mc_state;

/* Parse a model blob (pointers point into the blob; nothing is copied). 0 = ok */
int mc_load(mc_model *m, const uint8_t *blob, size_t len);
/* Bytes needed for the runtime state (activations + KV cache + RoPE tables). */
size_t mc_state_bytes(const mc_model *m);
/* Lay out the state inside caller-provided memory of mc_state_bytes() bytes. */
void mc_state_init(const mc_model *m, mc_state *s, void *mem);
/* Run one token at position pos; returns pointer to vocab logits. */
float *mc_forward(const mc_model *m, mc_state *s, int token, int pos);
/* Multiply-accumulate count of one forward step at position pos. */
uint64_t mc_macs_per_token(const mc_model *m, int pos);

/* ---------------- byte-level BPE tokenizer ---------------- */
typedef struct {
    int vocab, n_merges;
    const uint32_t *tok_off;  /* vocab+1 offsets into tok_bytes */
    const uint8_t *tok_bytes;
    const int32_t *merges;    /* n_merges x 3 : left, right, result (in rank order) */
    int byte_tok[256];
    int sp_q, sp_c, sp_b, sp_r, sp_e;
} mc_tok;

int mc_tok_load(mc_tok *t, const uint8_t *blob, size_t len);
/* Encode plain text (no special tokens). Returns number of ids, or -1 if max exceeded. */
int mc_tok_encode(const mc_tok *t, const char *text, int *ids, int max);
/* Pointer/length of the raw bytes of a token. */
const uint8_t *mc_tok_piece(const mc_tok *t, int id, int *len);

/* ---------------- generation ---------------- */
/* Greedy decode after a prompt; writes up to max_new ids, stops at stop_id. Returns count. */
int mc_generate_greedy(const mc_model *m, mc_state *s, const int *prompt, int n_prompt,
                       int *out, int max_new, int stop_id);
#endif
