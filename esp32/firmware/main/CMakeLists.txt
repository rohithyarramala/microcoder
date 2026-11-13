idf_component_register(SRCS "main.c" "../../engine/microcoder.c" "model_data.c" "prompts.c"
                       INCLUDE_DIRS "." "../../engine")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
