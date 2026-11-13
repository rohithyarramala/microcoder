MC_MOD_DIR := $(USERMOD_DIR)
ENGINE_DIR := $(MC_MOD_DIR)/../../engine
SRC_USERMOD_C += $(MC_MOD_DIR)/modmicrocoder.c $(ENGINE_DIR)/microcoder.c $(MC_MOD_DIR)/model_data.c
CFLAGS_USERMOD += -I$(ENGINE_DIR) -O2
