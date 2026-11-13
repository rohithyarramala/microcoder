/* MicroPython native module exposing the MicroCoder engine.
 *
 *   import microcoder
 *   microcoder.generate("<q>Write a function ...<c>", max_new=120, temperature=0.0) -> str
 *   microcoder.info() -> dict(weight_bytes, state_bytes, vocab, ctx, ...)
 *
 * The REF loop itself is plain MicroPython (see ../../app/ref.py) and uses
 * MicroPython's own built-in compile() as the verification gate.
 */
#include <string.h>
#include "py/runtime.h"
#include "py/objstr.h"
#include "microcoder.h"

extern const uint8_t mc_model_blob[];
extern const unsigned int mc_model_blob_len;
extern const uint8_t mc_tok_blob[];
extern const unsigned int mc_tok_blob_len;

static mc_model g_m;
static mc_tok g_t;
static mc_state g_s;
static void *g_mem = NULL;
static int g_ready = 0;
static uint32_t g_rng = 2463534242u;

#define MAX_IDS 256
static int g_ids[MAX_IDS];
static int g_out[MAX_IDS];

static void ensure_ready(void) {
    if (g_ready) return;
    if (mc_load(&g_m, mc_model_blob, mc_model_blob_len)) mp_raise_ValueError(MP_ERROR_TEXT("bad model blob"));
    if (mc_tok_load(&g_t, mc_tok_blob, mc_tok_blob_len)) mp_raise_ValueError(MP_ERROR_TEXT("bad tokenizer blob"));
    g_mem = m_malloc(mc_state_bytes(&g_m)); /* on ESP32-S3 this lands in PSRAM when enabled */
    mc_state_init(&g_m, &g_s, g_mem);
    g_ready = 1;
}

/* encode a prompt that may contain the special markers <q> <c> <b> <r> <e> */
static int encode_prompt(const char *s, size_t n, int *ids, int max) {
    static const char *names[5] = {"<q>", "<c>", "<b>", "<r>", "<e>"};
    int sp[5] = {g_t.sp_q, g_t.sp_c, g_t.sp_b, g_t.sp_r, g_t.sp_e};
    int out = 0;
    size_t i = 0, start = 0;
    char buf[1024];
    while (i <= n) {
        int hit = -1;
        if (i + 3 <= n)
            for (int k = 0; k < 5; k++) if (memcmp(s + i, names[k], 3) == 0) { hit = k; break; }
        if (hit >= 0 || i == n) {
            size_t L = i - start;
            if (L) {
                if (L >= sizeof buf) return -1;
                memcpy(buf, s + start, L); buf[L] = 0;
                int r = mc_tok_encode(&g_t, buf, ids + out, max - out);
                if (r < 0) return -1;
                out += r;
            }
            if (hit >= 0) { if (out >= max) return -1; ids[out++] = sp[hit]; i += 3; start = i; continue; }
            break;
        }
        i++;
    }
    return out;
}

static int sample(const float *lg, int V, float temp) {
    int best = 0;
    for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
    if (temp <= 0.f) return best;
    float sum = 0.f;
    for (int i = 0; i < V; i++) sum += expf((lg[i] - lg[best]) / temp);
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    float r = (g_rng / 4294967296.0f) * sum;
    for (int i = 0; i < V; i++) { r -= expf((lg[i] - lg[best]) / temp); if (r <= 0.f) return i; }
    return best;
}

static mp_obj_t mc_generate_fn(size_t n_args, const mp_obj_t *pos, mp_map_t *kw) {
    enum { ARG_prompt, ARG_max_new, ARG_temperature };
    static const mp_arg_t allowed[] = {
        {MP_QSTR_prompt, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_max_new, MP_ARG_INT, {.u_int = 120}},
        {MP_QSTR_temperature, MP_ARG_OBJ, {.u_obj = mp_const_none}},
    };
    mp_arg_val_t a[3];
    mp_arg_parse_all(n_args, pos, kw, 3, allowed, a);
    ensure_ready();
    size_t n;
    const char *s = mp_obj_str_get_data(a[ARG_prompt].u_obj, &n);
    float temp = a[ARG_temperature].u_obj == mp_const_none ? 0.f : (float)mp_obj_get_float(a[ARG_temperature].u_obj);
    int np = encode_prompt(s, n, g_ids, MAX_IDS);
    if (np < 0 || np >= g_m.ctx) return mp_const_none; /* prompt does not fit */
    int max_new = a[ARG_max_new].u_int;
    if (max_new > g_m.ctx - np) max_new = g_m.ctx - np;
    float *lg = NULL;
    for (int p = 0; p < np; p++) lg = mc_forward(&g_m, &g_s, g_ids[p], p);
    int pos_i = np, cnt = 0;
    while (cnt < max_new) {
        int nx = sample(lg, g_m.vocab, temp);
        if (nx == g_t.sp_e) break;
        g_out[cnt++] = nx;
        if (pos_i >= g_m.ctx) break;
        lg = mc_forward(&g_m, &g_s, nx, pos_i++);
    }
    vstr_t v;
    vstr_init(&v, 256);
    for (int i = 0; i < cnt; i++) { int L; const uint8_t *b = mc_tok_piece(&g_t, g_out[i], &L); vstr_add_strn(&v, (const char *)b, L); }
    return mp_obj_new_str_from_vstr(&v);
}
static MP_DEFINE_CONST_FUN_OBJ_KW(mc_generate_obj, 1, mc_generate_fn);

static mp_obj_t mc_info_fn(void) {
    ensure_ready();
    mp_obj_t d = mp_obj_new_dict(6);
    mp_obj_dict_store(d, MP_OBJ_NEW_QSTR(MP_QSTR_weight_bytes), mp_obj_new_int(g_m.weight_bytes));
    mp_obj_dict_store(d, MP_OBJ_NEW_QSTR(MP_QSTR_state_bytes), mp_obj_new_int(mc_state_bytes(&g_m)));
    mp_obj_dict_store(d, MP_OBJ_NEW_QSTR(MP_QSTR_vocab), mp_obj_new_int(g_m.vocab));
    mp_obj_dict_store(d, MP_OBJ_NEW_QSTR(MP_QSTR_ctx), mp_obj_new_int(g_m.ctx));
    mp_obj_dict_store(d, MP_OBJ_NEW_QSTR(MP_QSTR_layers), mp_obj_new_int(g_m.layers));
    mp_obj_dict_store(d, MP_OBJ_NEW_QSTR(MP_QSTR_d), mp_obj_new_int(g_m.d));
    return d;
}
static MP_DEFINE_CONST_FUN_OBJ_0(mc_info_obj, mc_info_fn);

static mp_obj_t mc_seed_fn(mp_obj_t s) { g_rng = (uint32_t)mp_obj_get_int(s) | 1u; return mp_const_none; }
static MP_DEFINE_CONST_FUN_OBJ_1(mc_seed_obj, mc_seed_fn);

static const mp_rom_map_elem_t mc_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_microcoder)},
    {MP_ROM_QSTR(MP_QSTR_generate), MP_ROM_PTR(&mc_generate_obj)},
    {MP_ROM_QSTR(MP_QSTR_info), MP_ROM_PTR(&mc_info_obj)},
    {MP_ROM_QSTR(MP_QSTR_seed), MP_ROM_PTR(&mc_seed_obj)},
};
static MP_DEFINE_CONST_DICT(mc_globals, mc_globals_table);

const mp_obj_module_t mc_user_cmodule = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t *)&mc_globals,
};
MP_REGISTER_MODULE(MP_QSTR_microcoder, mc_user_cmodule);
