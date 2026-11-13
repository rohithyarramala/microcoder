/* Host stand-in for ESP-IDF heap_caps with an ESP32-S3 memory budget, so the
 * firmware's allocation logic is exercised exactly as on the chip. */
#pragma once
#include <stdlib.h>
#include <stddef.h>
#define MALLOC_CAP_SPIRAM (1 << 10)
#define MALLOC_CAP_8BIT (1 << 2)
#define MALLOC_CAP_INTERNAL (1 << 11)
size_t sim_internal_free(void);
size_t sim_psram_free(void);
void *heap_caps_malloc(size_t n, unsigned caps);
size_t heap_caps_get_free_size(unsigned caps);
size_t heap_caps_get_minimum_free_size(unsigned caps);
