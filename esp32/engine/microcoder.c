#include "microcoder.h"
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------ blob parsing */
typedef struct { const uint8_t *p, *end; } rd_t;

static const void *take(rd_t *r, size_t n) {
    if (r->p + n > r->end) return NULL;
    const void *q = r->p;
    r->p += (n + 3) & ~(size_t)3; /* every array is padded to 4 bytes */
    if (r->p > r->end) r->p = r->end;
    return q;
}

static int take_q(rd_t *r, mc_qmat *q, int rows, int cols) {
    q->rows = rows; q->cols = cols;
    q->w = (const int8_t *)take(r, (size_t)rows * cols);
    q->scale = (const float *)take(r, (size_t)rows * 4);
    return q->w && q->scale ? 0 : -1;
}

int mc_load(mc_model *m, const uint8_t *blob, size_t len) {
    rd_t r = {blob, blob + len};
    const int32_t *h = (const int32_t *)take(&r, 32);
    if (!h || (uint32_t)h[0] != MC_MAGIC) return -1;
    m->vocab = h[1]; m->d = h[2]; m->layers = h[3]; m->heads = h[4]; m->ff = h[5]; m->ctx = h[6];
    if (m->layers > 16 || m->d % m->heads) return -2;
    int d = m->d, f = m->ff * m->d;
    if (take_q(&r, &m->emb, m->vocab, d)) return -3;
    m->nf = (const float *)take(&r, d * 4);
    for (int l = 0; l < m->layers; l++) {
        mc_layer *L = &m->layer[l];
        L->n1 = (const float *)take(&r, d * 4);
        L->n2 = (const float *)take(&r, d * 4);
        if (take_q(&r, &L->qkv, 3 * d, d) || take_q(&r, &L->o, d, d) ||
            take_q(&r, &L->up, f, d) || take_q(&r, &L->down, d, f)) return -4;
    }
    m->weight_bytes = (size_t)(r.p - blob);
    return 0;
}

/* ------------------------------------------------------------ state */
static size_t state_floats(const mc_model *m) {
    size_t d = m->d, f = (size_t)m->ff * m->d, hd = d / m->heads;
    return 3 * d + f + d + m->ctx + m->vocab + 2 * (size_t)m->layers * m->ctx * d + 2 * (size_t)m->ctx * (hd / 2);
}

size_t mc_state_bytes(const mc_model *m) { return state_floats(m) * sizeof(float); }

void mc_state_init(const mc_model *m, mc_state *s, void *mem) {
    float *p = (float *)mem;
    size_t d = m->d, f = (size_t)m->ff * m->d, hd = d / m->heads;
    s->x = p; p += d; s->xb = p; p += d; s->xb2 = p; p += d; s->hb = p; p += f; s->q = p; p += d;
    s->att = p; p += m->ctx; s->logits = p; p += m->vocab;
    s->kcache = p; p += (size_t)m->layers * m->ctx * d;
    s->vcache = p; p += (size_t)m->layers * m->ctx * d;
    float *c = p, *sn = p + (size_t)m->ctx * (hd / 2);
    for (int t = 0; t < m->ctx; t++)
        for (size_t i = 0; i < hd / 2; i++) {
            double inv = 1.0 / pow(10000.0, (double)(2 * i) / (double)hd);
            c[t * (hd / 2) + i] = (float)cos(t * inv);
            sn[t * (hd / 2) + i] = (float)sin(t * inv);
        }
    s->cos_t = c; s->sin_t = sn; s->rope = c;
}

/* ------------------------------------------------------------ kernels */
static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

static void qmatvec(float *y, const mc_qmat *q, const float *x) {
    for (int r = 0; r < q->rows; r++) {
        const int8_t *w = q->w + (size_t)r * q->cols;
        float acc = 0.f;
        for (int c = 0; c < q->cols; c++) acc += (float)w[c] * x[c];
        y[r] = acc * q->scale[r];
    }
}

static void rope(float *v, const float *c, const float *s, int hd) {
    for (int i = 0; i < hd / 2; i++) {
        float a = v[2 * i], b = v[2 * i + 1];
        v[2 * i] = a * c[i] - b * s[i];
        v[2 * i + 1] = a * s[i] + b * c[i];
    }
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

float *mc_forward(const mc_model *m, mc_state *s, int token, int pos) {
    int d = m->d, H = m->heads, hd = d / H, f = m->ff * d;
    const float *ct = s->cos_t + (size_t)pos * (hd / 2), *st = s->sin_t + (size_t)pos * (hd / 2);
    const int8_t *e = m->emb.w + (size_t)token * d;
    for (int i = 0; i < d; i++) s->x[i] = e[i] * m->emb.scale[token];
    float *qkv = s->hb; /* scratch: ff*d >= 3*d */
    for (int l = 0; l < m->layers; l++) {
        const mc_layer *L = &m->layer[l];
        float *kc = s->kcache + (size_t)l * m->ctx * d, *vc = s->vcache + (size_t)l * m->ctx * d;
        rmsnorm(s->xb, s->x, L->n1, d);
        qmatvec(qkv, &L->qkv, s->xb);
        memcpy(s->q, qkv, d * sizeof(float));
        memcpy(kc + (size_t)pos * d, qkv + d, d * sizeof(float));
        memcpy(vc + (size_t)pos * d, qkv + 2 * d, d * sizeof(float));
        for (int h = 0; h < H; h++) {
            rope(s->q + h * hd, ct, st, hd);
            rope(kc + (size_t)pos * d + h * hd, ct, st, hd);
        }
        float sc = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < H; h++) {
            const float *qh = s->q + h * hd;
            float mx = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const float *k = kc + (size_t)t * d + h * hd;
                float a = 0.f;
                for (int i = 0; i < hd; i++) a += qh[i] * k[i];
                a *= sc;
                s->att[t] = a;
                if (a > mx) mx = a;
            }
            float sum = 0.f;
            for (int t = 0; t <= pos; t++) { s->att[t] = expf(s->att[t] - mx); sum += s->att[t]; }
            float *o = s->xb2 + h * hd;
            for (int i = 0; i < hd; i++) o[i] = 0.f;
            for (int t = 0; t <= pos; t++) {
                const float *v = vc + (size_t)t * d + h * hd;
                float w = s->att[t] / sum;
                for (int i = 0; i < hd; i++) o[i] += w * v[i];
            }
        }
        qmatvec(s->xb, &L->o, s->xb2);
        for (int i = 0; i < d; i++) s->x[i] += s->xb[i];
        rmsnorm(s->xb, s->x, L->n2, d);
        qmatvec(s->hb, &L->up, s->xb);
        for (int i = 0; i < f; i++) s->hb[i] = gelu(s->hb[i]);
        qmatvec(s->xb, &L->down, s->hb);
        for (int i = 0; i < d; i++) s->x[i] += s->xb[i];
    }
    rmsnorm(s->xb, s->x, m->nf, d);
    qmatvec(s->logits, &m->emb, s->xb);
    return s->logits;
}

uint64_t mc_macs_per_token(const mc_model *m, int pos) {
    uint64_t d = m->d, f = (uint64_t)m->ff * m->d;
    uint64_t per_layer = 3 * d * d + d * d + 2 * d * f + 2 * (uint64_t)(pos + 1) * d;
    return m->layers * per_layer + (uint64_t)m->vocab * d;
}

int mc_generate_greedy(const mc_model *m, mc_state *s, const int *prompt, int n_prompt,
                       int *out, int max_new, int stop_id) {
    float *lg = NULL;
    int pos = 0, n = 0;
    for (; pos < n_prompt && pos < m->ctx; pos++) lg = mc_forward(m, s, prompt[pos], pos);
    while (n < max_new && lg) {
        int best = 0;
        for (int i = 1; i < m->vocab; i++) if (lg[i] > lg[best]) best = i;
        if (best == stop_id) break;
        out[n++] = best;
        if (pos >= m->ctx) break;
        lg = mc_forward(m, s, best, pos++);
    }
    return n;
}

/* ------------------------------------------------------------ tokenizer
 * blob: magic "MCT1", vocab, n_merges, sp_q, sp_c, sp_b, sp_r, sp_e,
 *       u32 offsets[vocab+1], bytes[], (pad), i32 merges[n_merges*3]
 */
int mc_tok_load(mc_tok *t, const uint8_t *blob, size_t len) {
    rd_t r = {blob, blob + len};
    const int32_t *h = (const int32_t *)take(&r, 32);
    if (!h || (uint32_t)h[0] != 0x3154434Du) return -1;
    t->vocab = h[1]; t->n_merges = h[2];
    t->sp_q = h[3]; t->sp_c = h[4]; t->sp_b = h[5]; t->sp_r = h[6]; t->sp_e = h[7];
    t->tok_off = (const uint32_t *)take(&r, (size_t)(t->vocab + 1) * 4);
    if (!t->tok_off) return -2;
    t->tok_bytes = (const uint8_t *)take(&r, t->tok_off[t->vocab]);
    t->merges = (const int32_t *)take(&r, (size_t)t->n_merges * 12);
    if (!t->tok_bytes || !t->merges) return -3;
    for (int b = 0; b < 256; b++) t->byte_tok[b] = -1;
    for (int i = 0; i < t->vocab; i++)
        if (t->tok_off[i + 1] - t->tok_off[i] == 1) {
            int b = t->tok_bytes[t->tok_off[i]];
            if (t->byte_tok[b] < 0) t->byte_tok[b] = i;
        }
    return 0;
}

const uint8_t *mc_tok_piece(const mc_tok *t, int id, int *len) {
    *len = (int)(t->tok_off[id + 1] - t->tok_off[id]);
    return t->tok_bytes + t->tok_off[id];
}

static int is_l(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 128; }
static int is_n(int c) { return c >= '0' && c <= '9'; }
static int is_s(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static int is_o(int c) { return !is_l(c) && !is_n(c) && !is_s(c); }

/* GPT-2 pre-tokenisation (ASCII subset; UTF-8 bytes >=128 treated as letters). */
static int next_pretoken(const uint8_t *s, int n, int i) {
    if (s[i] == '\'' && i + 1 < n) {
        static const char *cs[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (int k = 0; k < 7; k++) {
            int L = (int)strlen(cs[k]);
            if (i + 1 + L <= n && memcmp(s + i + 1, cs[k], L) == 0) return i + 1 + L;
        }
    }
    int j = i;
    int sp = (s[i] == ' ' && i + 1 < n) ? 1 : 0;
    if ((sp && is_l(s[i + 1])) || is_l(s[i])) { j = i + (is_l(s[i]) ? 0 : 1); while (j < n && is_l(s[j])) j++; return j; }
    if ((sp && is_n(s[i + 1])) || is_n(s[i])) { j = i + (is_n(s[i]) ? 0 : 1); while (j < n && is_n(s[j])) j++; return j; }
    if ((sp && is_o(s[i + 1])) || is_o(s[i])) { j = i + (is_o(s[i]) ? 0 : 1); while (j < n && is_o(s[j])) j++; return j; }
    j = i;
    while (j < n && is_s(s[j])) j++;
    if (j < n && j - i > 1) return j - 1;  /* \s+(?!\S) */
    return j;                              /* \s+ */
}

static int merge_rank(const mc_tok *t, int a, int b, int *res) {
    /* linear scan in rank order; the merge table is small (< 2k entries) */
    for (int k = 0; k < t->n_merges; k++)
        if (t->merges[3 * k] == a && t->merges[3 * k + 1] == b) { *res = t->merges[3 * k + 2]; return k; }
    return -1;
}

int mc_tok_encode(const mc_tok *t, const char *text, int *ids, int max) {
    const uint8_t *s = (const uint8_t *)text;
    int n = (int)strlen(text), out = 0;
    for (int i = 0; i < n;) {
        int j = next_pretoken(s, n, i);
        if (out + (j - i) > max) return -1;
        int *w = ids + out, m = 0;
        for (int k = i; k < j; k++) w[m++] = t->byte_tok[s[k]];
        for (;;) {
            int best = -1, bi = -1, bres = 0;
            for (int k = 0; k + 1 < m; k++) {
                int res, r = merge_rank(t, w[k], w[k + 1], &res);
                if (r >= 0 && (best < 0 || r < best)) { best = r; bi = k; bres = res; }
            }
            if (best < 0) break;
            w[bi] = bres;
            memmove(w + bi + 1, w + bi + 2, (m - bi - 2) * sizeof(int));
            m--;
        }
        out += m;
        i = j;
    }
    return out;
}
