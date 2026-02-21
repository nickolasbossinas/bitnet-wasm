cmake_minimum_required(VERSION 3.16)
project(bitnet-wasm C)

set(CMAKE_C_STANDARD 11)

# Kernel sources
set(KERNEL_SOURCES
    src/kernels/i2s.c
    src/kernels/tl1.c
    src/kernels/tl2.c
    src/kernels/gemv.c
)

# Inference sources
set(INFERENCE_SOURCES
    src/inference/gguf.c
    src/inference/model.c
    src/inference/sampler.c
    src/inference/weight_loader.c
)

# Benchmark
set(BENCH_SOURCES
    src/bench/benchmark.c
)

# Inference tests
set(TEST_SOURCES
    src/test/test_inference.c
)

if(EMSCRIPTEN)
    # =========================================
    # WASM build (Emscripten)
    # =========================================
    message(STATUS "Building for WebAssembly with SIMD")

    # Enable WASM SIMD128
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128")

    # Optimization
    set(CMAKE_C_FLAGS_RELEASE "-O3 -flto")

    # Benchmark executable -> .js + .wasm
    add_executable(benchmark ${KERNEL_SOURCES} ${BENCH_SOURCES})

    # Emscripten link flags
    target_link_options(benchmark PRIVATE
        # Export run_benchmark for JS calls
        -sEXPORTED_FUNCTIONS=['_main','_run_benchmark']
        -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']

        # Memory settings
        -sINITIAL_MEMORY=256MB
        -sALLOW_MEMORY_GROWTH=1
        -sMAXIMUM_MEMORY=512MB

        # Stack size for deep recursion
        -sSTACK_SIZE=1MB

        # Output format
        -sMODULARIZE=1
        -sEXPORT_NAME='BitNetBench'

        # Environment
        "-sENVIRONMENT=web,worker"
    )

    # Copy HTML to build dir
    add_custom_command(TARGET benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/web/index.html
            ${CMAKE_BINARY_DIR}/index.html
        COMMENT "Copying index.html to build directory"
    )

else()
    # =========================================
    # Native build (for testing without Emscripten)
    # =========================================
    message(STATUS "Building native (no WASM SIMD, scalar only)")

    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

    add_executable(benchmark ${KERNEL_SOURCES} ${BENCH_SOURCES})
    target_link_libraries(benchmark m)  # libm for math functions

    # Inference test executable
    add_executable(test_inference ${KERNEL_SOURCES} ${INFERENCE_SOURCES} ${TEST_SOURCES})
    target_link_libraries(test_inference m)

    # GGUF inspector tool
    add_executable(inspect_gguf src/inference/gguf.c src/test/inspect_gguf.c)
    target_link_libraries(inspect_gguf m)
endif()
