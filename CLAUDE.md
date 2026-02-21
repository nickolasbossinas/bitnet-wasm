# BitNet WASM SIMD Kernels

## Project Goal

Build custom WASM SIMD kernels for BitNet b1.58 (1.58-bit ternary LLM) inference in the browser. The target model is Microsoft's BitNet b1.58 2B4T (2 billion parameters, ternary weights {-1, 0, 1}).

This is a standalone project. It will eventually integrate with the S2S (speech-to-speech) project at `C:\Users\nicko\Documents\S2S\git\s2s`, replacing the mock LLM (`services/llm.ts`) with real in-browser inference.

## Architecture Decisions

- **Language**: C compiled to WASM via Emscripten (chosen over Rust/AssemblyScript for closest reference to bitnet.cpp)
- **Approach**: Custom WASM SIMD kernels written from scratch (NOT compiling bitnet.cpp to WASM, which would lose AVX2 optimizations)
- **Key insight**: BitNet's TL1 kernel uses PSHUFB (byte shuffle) as its core op. WASM SIMD's `wasm_i8x16_swizzle` maps directly to PSHUFB on x86 and VTBL on ARM — making it a near-perfect fit for 128-bit WASM SIMD

## Current State

All phases complete. Full end-to-end inference works in both native CLI and browser WASM.

### Completed Phases

- **Phase 1**: GEMV kernel implementations (I2_S scalar/SIMD, TL1 scalar/SIMD, TL2) + benchmark harness
- **Phase 2**: GGUF parser, weight loader (I2_S decode with interleaved blocks + per-tensor scale), F16→F32 conversion, TL1 repacking
- **Phase 3**: BPE tokenizer with GPT-2 byte encoding, encode/decode roundtrip
- **Phase 4**: Forward pass (RMSNorm, RoPE, GQA attention, SqReLU, SubLN), sampler (argmax + top-p), generation loop with streaming callback
- **Phase 5**: WASM build (Emscripten), browser UI with Web Worker, model loading via file picker
- **Phase 6**: Repetition penalty, n-gram loop detection, double-newline stop sequence

### Test Status

21/21 tests pass (native build via Docker).

### File Structure

```
src/
  kernels/
    types.h           - Data structures (weight matrices, activations, output buffers)
    simd_utils.h      - WASM SIMD intrinsic wrappers (swizzle, reduce, nibble extract)
    i2s.h / i2s.c     - I2_S kernel: 2-bit packed weights, multiply-accumulate
    tl1.h / tl1.c     - TL1 kernel: 4-bit LUT index, wasm_i8x16_swizzle lookup
    tl2.h / tl2.c     - TL2 kernel: 3-weight triples, sign+index encoding
    gemv.h / gemv.c   - Unified GEMV wrapper, activation quantization, timing
  inference/
    gguf.h / gguf.c       - GGUF v3 parser (metadata, tensor info, tokenizer data)
    model.h / model.c     - Model config, alloc/free, forward pass
    weight_loader.h / .c  - I2_S interleaved decode, F16 convert, TL1 repack
    tokenizer.h / .c      - BPE tokenizer (GPT-2 byte encoding)
    sampler.h / .c        - Argmax + top-p nucleus sampling
    generate.h / .c       - Generation loop with rep penalty + stop mechanisms
  test/
    test_inference.c      - 21 unit/integration tests
    diagnose_weights.c    - Weight format diagnostic tool
  bench/
    benchmark.c           - Benchmark harness (kernel variants + correctness)
  main.c                  - Native CLI tool (bitnet-cli)
  wasm_api.c              - WASM entry points (bitnet_init/generate/free)
web/
  index.html              - Browser benchmark UI with SIMD detection
  inference.html          - Browser inference UI (model loading, generation, settings)
  worker.js               - Web Worker for off-main-thread inference
docker/
  docker-compose.yml      - Build (emscripten), test (gcc), generate services
CMakeLists.txt            - Emscripten + native build config
```

### I2_S Format (GGML_TYPE_I2_S = 36)

Custom BitNet quantization format. Key details for anyone working on weight loading:
- **Interleaved 128-weight blocks** (32 bytes each), NOT simple LSB-first sequential
- Each byte: bits 6-7=group0 (weights 0-31), bits 4-5=group1 (32-63), bits 2-3=group2 (64-95), bits 0-1=group3 (96-127)
- Value codes: 00=-1, 01=0, 10=+1
- **Per-tensor float32 scale** appended after packed data at byte offset `ceil(n_elements/4)`
- Tail elements (< 128) are packed MSB-first sequentially

### Model Architecture (BitNet b1.58 2B4T)

- LLaMA-like decoder, 30 layers, hidden=2560, intermediate=6912
- GQA: 20 Q heads, 5 KV heads, head_dim=128
- SqReLU activation (relu squared), SubLN (RMSNorm before W_o and W_down)
- RoPE (theta=500000), tied word embeddings, no biases
- Vocab: 128256 tokens (GPT-2 BPE)
- **Base pretrained model** (not instruction-tuned) — generates continuous text, not Q&A

### Generation Controls

- Repetition penalty (default 1.1): penalizes previously seen token logits
- N-gram loop detection: stops on repeated 8-grams
- Double-newline stop: breaks on `\n\n` paragraph boundaries
- EOS token detection (token 128001)
- Top-p nucleus sampling with temperature

## Build Instructions

```bash
# WASM build (via Docker)
docker compose -f docker/docker-compose.yml run --rm build

# Native test (via Docker, runs 21/21 tests)
docker compose -f docker/docker-compose.yml run --rm test

# Native CLI generation
docker compose -f docker/docker-compose.yml run --rm generate

# Serve browser UI
python -m http.server 8080 --directory build
# Open http://localhost:8080/inference.html

# Native build (without Docker)
mkdir build-native && cd build-native
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./bitnet-cli ../models/ggml-model-i2_s.gguf -p "The capital of France is" -n 32
```

## Target Hardware

Primary dev machine: Acer Aspire 15 A15-51M
- CPU: Intel i9-13900H (14 cores, 20 threads, AVX2)
- RAM: 32 GB
- GPU: Intel Iris Xe (integrated, 96 EUs)

## Key Technical References

- BitNet TL1/TL2 kernels paper: https://arxiv.org/abs/2502.11880
- BitNet b1.58 model: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- Microsoft bitnet.cpp: https://github.com/microsoft/BitNet
- WASM SIMD spec: https://github.com/WebAssembly/simd
- Emscripten SIMD docs: https://emscripten.org/docs/porting/simd.html
- wasm_i8x16_swizzle = PSHUFB on x86 = VTBL on ARM

## Possible Future Work

- **WASM SIMD TL1 kernel optimization** — restructure for better SIMD utilization in the inner loop
- **Web Worker threading** — parallelize GEMV across multiple workers via SharedArrayBuffer
- **Instruction-tuned models** — when available, will drop in with no code changes
- **S2S integration** — replace mock LLM service with this WASM inference module

## User Preferences

- Platform: Windows 11
- Shell: bash (Unix syntax, not Windows)
- Commit style: descriptive messages with context
