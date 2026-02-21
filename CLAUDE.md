# BitNet WASM SIMD Kernels

## Project Goal

Build custom WASM SIMD kernels for BitNet b1.58 (1.58-bit ternary LLM) inference in the browser. The target model is Microsoft's BitNet b1.58 2B4T (2 billion parameters, ternary weights {-1, 0, 1}).

## Architecture Decisions

- **Language**: C compiled to WASM via Emscripten (chosen over Rust/AssemblyScript for closest reference to bitnet.cpp)
- **Approach**: Custom WASM SIMD kernels written from scratch (NOT compiling bitnet.cpp to WASM, which would lose AVX2 optimizations)
- **Key insight**: BitNet's TL1 kernel uses PSHUFB (byte shuffle) as its core op. WASM SIMD's `wasm_i8x16_swizzle` maps directly to PSHUFB on x86 and VTBL on ARM — making it a near-perfect fit for 128-bit WASM SIMD

## Current State

**Phase 1 (DONE)**: GEMV kernel implementations + benchmark harness

### File Structure

```
src/
  kernels/
    types.h           - Data structures (weight matrices, activations, output buffers)
    simd_utils.h      - WASM SIMD intrinsic wrappers (swizzle, reduce, nibble extract)
    i2s.h / i2s.c     - I2_S kernel: 2-bit packed weights, multiply-accumulate
    tl1.h / tl1.c     - TL1 kernel: 4-bit LUT index, wasm_i8x16_swizzle lookup
    gemv.h / gemv.c   - Unified GEMV wrapper, activation quantization, timing
  bench/
    benchmark.c       - Benchmark harness (4 kernel variants, correctness check)
web/
  index.html          - Browser benchmark UI with SIMD detection
CMakeLists.txt        - Emscripten + native build config
```

### Four Kernel Variants

1. **I2_S scalar** — baseline: unpack 2-bit weights, multiply, accumulate
2. **I2_S SIMD** — vectorized with wasm_i16x8_mul, int32 accumulation
3. **TL1 scalar** — build 16-byte LUT per activation pair, lookup by index
4. **TL1 SIMD** — same with wasm_i8x16_swizzle for lookups

### How TL1 Works

Each pair of ternary weights (w0, w1) has only 9 possible outcomes. Pre-compute all 9 into a 16-byte LUT (padded), then use swizzle to select the right one. This eliminates ALL multiplications from the inner loop.

```
index = (w0 + 1) * 3 + (w1 + 1)   // maps to 0..8
LUT[index] = w0*a0 + w1*a1         // pre-computed from activations
result = swizzle(LUT, indices)      // single WASM SIMD instruction
```

## Next Steps (TODO)

1. **Install Emscripten** and verify the build compiles
2. **Run benchmark** in browser, measure actual WASM SIMD performance
3. **Optimize TL1 SIMD kernel** — current implementation falls back to scalar inner loop for the LUT lookup since each activation pair has a different 16-byte LUT. Need to restructure for better SIMD utilization (e.g., process multiple output rows in parallel)
4. **Add TL2 kernel** — 3 weights per index (1-bit sign + 4-bit index), more aggressive compression
5. **Full inference pipeline** — model loading (GGUF), tokenization, attention, sampling
6. **Web Worker integration** — parallelize GEMV across multiple threads via SharedArrayBuffer

## Build Instructions

```bash
# Emscripten build
mkdir build && cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
emmake make

# Serve and open in browser
python -m http.server 8080
# Navigate to http://localhost:8080

# Native build (scalar only, for testing)
mkdir build-native && cd build-native
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./benchmark
```

## Target Hardware

Primary dev machine: Acer Aspire 15 A15-51M
- CPU: Intel i9-13900H (14 cores, 20 threads, AVX2)
- RAM: 32 GB
- GPU: Intel Iris Xe (integrated, 96 EUs)
- Expected native bitnet.cpp: ~30-50 tok/s for 2B model
- Expected WASM SIMD in browser: ~12-15 tok/s (estimated)

## Key Technical References

- BitNet TL1/TL2 kernels paper: https://arxiv.org/abs/2502.11880
- BitNet b1.58 model: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- Microsoft bitnet.cpp: https://github.com/microsoft/BitNet
- WASM SIMD spec: https://github.com/WebAssembly/simd
- Emscripten SIMD docs: https://emscripten.org/docs/porting/simd.html
- wasm_i8x16_swizzle = PSHUFB on x86 = VTBL on ARM

## User Preferences

- Platform: Windows 11
- Shell: bash (Unix syntax, not Windows)
- Commit style: descriptive messages with context
