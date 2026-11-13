/* MicroCoder on-device benchmark for ESP32-S3.
 *
 * For every benchmark prompt: tokenise on device, anchor the answer on the
 * requested signature, run greedy generation with
 * the INT8 engine, print the generated code between markers, and report
 * prefill / decode latency and memory. scripts/score_device_log.py parses the
 * UART log and scores the printed code with the same unit tests as the paper.
 */
#include <stdio.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "microcoder.h"

extern const uint8_t mc_model_blob[];
extern const unsigned int mc_model_blob_len;
extern const uint8_t mc_tok_blob[];
extern const unsigned int mc_tok_blob_len;
extern const char *const mc_prompts[];
extern const int mc_n_prompts;
extern const int mc_prompt_offset;   /* global index of mc_prompts[0] (non-zero only for sharded runs) */

static int ids[256], out[256];

/* Signature anchoring: copy "def name(args):" verbatim from the instruction
 * (same rule as the regex ([A-Za-z_]\w*)\(([^()]*)\) used on the host). */
static int is_w(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }
static int signature_prefix(const char *s, char *dst, int cap) {
    for (int i = 0; s[i]; i++) {
        if (s[i] != '(' || i == 0 || !is_w(s[i - 1])) continue;
        int j = i; while (j > 0 && is_w(s[j - 1])) j--;
        while (j < i && !((s[j] >= 'a' && s[j] <= 'z') || (s[j] >= 'A' && s[j] <= 'Z') || s[j] == '_')) j++;
        if (j == i) continue;
        int k = i + 1; while (s[k] && s[k] != ')' && s[k] != '(') k++;
        if (s[k] != ')') continue;
        int n = snprintf(dst, cap, "def %.*s:", k - j + 1, s + j);
        return n < cap ? n : -1;
    }
    dst[0] = 0;
    return 0;
}

void app_main(void) {
    mc_model m; mc_tok t; mc_state s;
    if (mc_load(&m, mc_model_blob, mc_model_blob_len) || mc_tok_load(&t, mc_tok_blob, mc_tok_blob_len)) {
        printf("MC_ERROR bad blobs\n"); return;
    }
    size_t sb = mc_state_bytes(&m);
    void *mem = heap_caps_malloc(sb, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    const char *where = "PSRAM";
    if (!mem) { mem = heap_caps_malloc(sb, MALLOC_CAP_8BIT); where = "internal"; }
    if (!mem) { printf("MC_ERROR cannot allocate %u bytes\n", (unsigned)sb); return; }
    mc_state_init(&m, &s, mem);
    printf("MC_INFO weights_bytes=%u state_bytes=%u state_in=%s free_internal=%u free_psram=%u\n",
           (unsigned)m.weight_bytes, (unsigned)sb, where,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    for (int i = 0; i < mc_n_prompts; i++) {
        /* prompt = <q> instruction <c> */
        int n = 0;
        ids[n++] = t.sp_q;
        int k = mc_tok_encode(&t, mc_prompts[i], ids + n, 250 - n);
        if (k < 0) { printf("MC_SKIP %d\n", i + mc_prompt_offset); continue; }
        n += k;
        ids[n++] = t.sp_c;
        char pre[256];
        if (signature_prefix(mc_prompts[i], pre, sizeof pre) > 0) {
            int kp = mc_tok_encode(&t, pre, ids + n, 250 - n);
            if (kp < 0) { printf("MC_SKIP %d\n", i + mc_prompt_offset); continue; }
            n += kp;
        }
        int64_t t0 = esp_timer_get_time();
        float *lg = NULL;
        for (int p = 0; p < n; p++) lg = mc_forward(&m, &s, ids[p], p);
        int64_t t1 = esp_timer_get_time();
        int cnt = 0, pos = n;
        while (cnt < 120 && pos < m.ctx) {
            int best = 0;
            for (int v = 1; v < m.vocab; v++) if (lg[v] > lg[best]) best = v;
            if (best == t.sp_e) break;
            out[cnt++] = best;
            lg = mc_forward(&m, &s, best, pos++);
        }
        int64_t t2 = esp_timer_get_time();
        printf("MC_BEGIN %d\n%s", i + mc_prompt_offset, pre);
        for (int j = 0; j < cnt; j++) { int L; const uint8_t *b = mc_tok_piece(&t, out[j], &L); fwrite(b, 1, L, stdout); }
        printf("\nMC_END %d prompt_tokens=%d new_tokens=%d prefill_us=%lld decode_us=%lld\n",
               i + mc_prompt_offset, n, cnt, (long long)(t1 - t0), (long long)(t2 - t1));
    }
    printf("MC_DONE min_free_internal=%u\n", (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
}
