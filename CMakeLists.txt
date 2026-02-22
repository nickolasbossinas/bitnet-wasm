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
    src/inference/tokenizer.c
    src/inference/generate.c
    src/inference/thread_pool.c
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

    # Enable WASM SIMD128 and pthreads (SharedArrayBuffer-backed Web Workers)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128 -pthread")

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

        # Pthreads (shared compile flags require link flags)
        -pthread
        -sPTHREAD_POOL_SIZE=1
    )

    # Copy HTML to build dir
    add_custom_command(TARGET benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/web/index.html
            ${CMAKE_BINARY_DIR}/index.html
        COMMENT "Copying index.html to build directory"
    )

    # =========================================
    # WASM Inference build
    # =========================================
    add_executable(bitnet-inference
        ${KERNEL_SOURCES}
        ${INFERENCE_SOURCES}
        src/wasm_api.c
    )

    target_link_options(bitnet-inference PRIVATE
        # Exported C functions (callable from JS)
        -sEXPORTED_FUNCTIONS=['_bitnet_init','_bitnet_generate','_bitnet_set_threads','_bitnet_abort','_bitnet_get_config','_bitnet_free','_malloc','_free']
        -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','stringToNewUTF8','HEAPU8']

        # Memory: 4GB max for full 30-layer model (~3.2 GB peak during loading)
        -sINITIAL_MEMORY=256MB
        -sALLOW_MEMORY_GROWTH=1
        -sMAXIMUM_MEMORY=4GB

        # Stack
        -sSTACK_SIZE=2MB

        # Output format
        -sMODULARIZE=1
        -sEXPORT_NAME='BitNetInference'

        # Environment (worker = main JS worker, worker subworkers for pthreads)
        "-sENVIRONMENT=web,worker"

        # Pthreads: pre-create pool of Web Workers for instant thread start
        -pthread
        -sPTHREAD_POOL_SIZE=8

        # No main() function
        --no-entry
    )

    # Copy inference web files to build dir
    add_custom_command(TARGET bitnet-inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/web/worker.js
            ${CMAKE_BINARY_DIR}/worker.js
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/web/bitnet.js
            ${CMAKE_BINARY_DIR}/bitnet.js
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/web/inference.html
            ${CMAKE_BINARY_DIR}/inference.html
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/web/serve.py
            ${CMAKE_BINARY_DIR}/serve.py
        COMMENT "Copying inference web files to build directory"
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

    # CLI generation tool
    add_executable(bitnet-cli ${KERNEL_SOURCES} ${INFERENCE_SOURCES} src/main.c)
    target_link_libraries(bitnet-cli m)

    # Weight diagnostic tool
    add_executable(diagnose_weights
        ${KERNEL_SOURCES}
        src/inference/gguf.c
        src/inference/weight_loader.c
        src/test/diagnose_weights.c
    )
    target_link_libraries(diagnose_weights m)

    # Entropy analysis tool
    add_executable(entropy_analysis
        ${KERNEL_SOURCES}
        src/inference/gguf.c
        src/inference/weight_loader.c
        src/test/entropy_analysis.c
    )
    target_link_libraries(entropy_analysis m)

    # Embedding SVD analysis tool
    add_executable(embedding_svd
        ${KERNEL_SOURCES}
        src/inference/gguf.c
        src/inference/weight_loader.c
        src/test/embedding_svd.c
    )
    target_link_libraries(embedding_svd m)

    # Block sparsity analysis tool
    add_executable(block_sparsity
        ${KERNEL_SOURCES}
        src/inference/gguf.c
        src/inference/weight_loader.c
        src/test/block_sparsity.c
    )
    target_link_libraries(block_sparsity m)

    # Codebook compression analysis tool
    add_executable(codebook_analysis
        src/inference/gguf.c
        src/test/codebook_analysis.c
    )
    target_link_libraries(codebook_analysis m)

    # Fourth code analysis tool
    add_executable(fourth_code
        ${KERNEL_SOURCES}
        src/inference/gguf.c
        src/inference/weight_loader.c
        src/test/fourth_code.c
    )
    target_link_libraries(fourth_code m)

    # Embedding compression analysis tool
    add_executable(embedding_compression
        ${KERNEL_SOURCES}
        src/inference/gguf.c
        src/inference/weight_loader.c
        src/test/embedding_compression.c
    )
    target_link_libraries(embedding_compression m)

    # GGUF embedding converter (F16 → INT8)
    add_executable(convert_embedding
        src/tools/convert_embedding.c
    )
    target_link_libraries(convert_embedding m)

    # GGUF weight compressor (I2_S → TRIT5)
    add_executable(compress_weights
        ${KERNEL_SOURCES}
        src/inference/gguf.c
        src/inference/weight_loader.c
        src/tools/compress_weights.c
    )
    target_link_libraries(compress_weights m)
endif()
