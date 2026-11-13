/* Runs the unmodified firmware app_main() on the host.
 * Budgets: ESP32-S3 has 512 KiB internal SRAM, of which ~300 KiB is typically
 * free for the heap after the IDF starts; PSRAM size is a command-line option
 * (0 = board without PSRAM). Flash-resident weights are const data. */
#include <stdio.h>
#include <stdlib.h>
#include "esp_heap_caps.h"
static size_t internal = 300 * 1024, psram = 8u << 20, min_internal;
size_t sim_internal_free(void) { return internal; }
size_t sim_psram_free(void) { return psram; }
void *heap_caps_malloc(size_t n, unsigned caps) {
    size_t *pool = (caps & MALLOC_CAP_SPIRAM) ? &psram : &internal;
    if (n > *pool) return NULL;
    *pool -= n;
    if (pool == &internal && internal < min_internal) min_internal = internal;
    return malloc(n);
}
size_t heap_caps_get_free_size(unsigned caps) { return (caps & MALLOC_CAP_SPIRAM) ? psram : internal; }
size_t heap_caps_get_minimum_free_size(unsigned caps) { (void)caps; return min_internal; }
void app_main(void);
int main(int argc, char **argv) {
    if (argc > 1) psram = (size_t)atol(argv[1]) * 1024;
    min_internal = internal;
    printf("SIM ESP32-S3 budget: internal_heap=%zu B psram=%zu B\n", internal, psram);
    app_main();
    return 0;
}
