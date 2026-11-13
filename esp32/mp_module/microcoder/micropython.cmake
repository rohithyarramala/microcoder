# ESP32 (CMake) build: idf.py -D USER_C_MODULES=<path>/micropython.cmake
add_library(usermod_microcoder INTERFACE)
target_sources(usermod_microcoder INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/modmicrocoder.c
    ${CMAKE_CURRENT_LIST_DIR}/../../engine/microcoder.c
    ${CMAKE_CURRENT_LIST_DIR}/model_data.c)
target_include_directories(usermod_microcoder INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../../engine)
target_link_libraries(usermod INTERFACE usermod_microcoder)
