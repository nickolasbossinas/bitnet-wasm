# BitNet WASM SIMD Kernels

Custom WASM SIMD kernels for [BitNet b1.58](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) ternary LLM inference in the browser. Targets the 2-billion parameter model with ternary weights {-1, 0, 1}.

> **12.68x faster** than scalar baseline. The TL1 SIMD kernel achieves **49.34 GOPS** on a 2048x2048 GEMV, competitive with single-threaded native [bitnet.cpp](https://github.com/microsoft/BitNet) running AVX2.

## Benchmark Results

Measured in Chrome on Intel i9-13900H (WASM SIMD, single-threaded):

```
 Matrix: 2048 x 2048
 I2_S (scalar)            2.156 ms    3.89 GOPS   1.00x
 I2_S (WASM SIMD)         1.636 ms    5.13 GOPS   1.32x
 TL1 (scalar)             2.458 ms    3.41 GOPS   0.88x
 TL1 (WASM SIMD)          0.170 ms   49.34 GOPS  12.68x
```

### End-to-End Inference (BitNet 2B4T, 64 tokens)

Measured in Chrome on Intel i9-13900H:

| Threads | Decode tok/s |
|---|---|
| 1 (single-threaded) | ~3.3 |
| 4 (3 workers) | ~8.5 |
| **6 (5 workers)** | **~11.0** |
| 7 (6 workers) | ~10.5 |

### vs Native bitnet.cpp

| Runtime | tok/s (2B model) | Notes |
|---|---|---|
| bitnet.cpp native (multi-threaded AVX2) | ~55-65 | All cores, 256-bit SIMD |
| bitnet.cpp single-thread | ~10-15 | AVX2 only |
| **This project (WASM SIMD, 6 threads)** | **~11** | 128-bit SIMD, SharedArrayBuffer workers |

## Key Insight

BitNet's TL1 kernel uses PSHUFB (byte shuffle) as its core operation. WASM SIMD's `wasm_i8x16_swizzle` maps directly to **the same native instruction** on both platforms:

- **x86**: `PSHUFB` (SSE3/AVX2)
- **ARM**: `VTBL` (NEON)

This means the WASM JIT produces near-native code for the critical path — no emulation overhead.

## How TL1 Works

Each pair of ternary weights (w0, w1) has only **9 possible outcomes**. Pre-compute all 9 into a 16-byte lookup table, then use swizzle to select the right one. This eliminates ALL multiplications from the inner loop.

```
index = (w0 + 1) * 3 + (w1 + 1)   // maps to 0..8
LUT[index] = w0*a0 + w1*a1         // pre-computed from activations
result = swizzle(LUT, indices)      // single WASM SIMD instruction
```

### SIMD Optimizations

The TL1 SIMD kernel applies three key optimizations:

1. **Column-major weight layout** — Weights are transposed so that 16 consecutive rows' indices are contiguous in memory. This replaces 16 scattered byte loads (~31 ops) with a single `wasm_v128_load` (1 op).

2. **Pre-split LUT** — The int16 LUT is deinterleaved into separate lo-byte and hi-byte tables in an O(K) prepass. This eliminates 4 shuffle ops per inner loop iteration.

3. **Int16 accumulation** — Results accumulate in int16 (2 adds per pair) instead of widening to int32 (4 extends + 4 adds per pair), flushing to int32 every 64 iterations to prevent overflow.

## Project Structure

```
bitnet-wasm/
├── src/
│   ├── kernels/
│   │   ├── types.h           # Data structures (weight matrices, activations)
│   │   ├── simd_utils.h      # WASM SIMD intrinsic wrappers
│   │   ├── i2s.h / i2s.c    # I2_S kernel: 2-bit packed, multiply-accumulate
│   │   ├── tl1.h / tl1.c    # TL1 kernel: LUT + wasm_i8x16_swizzle
│   │   └── gemv.h / gemv.c  # Unified GEMV interface, quantization, timing
│   ├── inference/
│   │   ├── gguf.h / gguf.c           # GGUF model format parser
│   │   ├── model.h / model.c         # Transformer forward pass
│   │   ├── tokenizer.h / tokenizer.c # BPE tokenizer
│   │   ├── generate.h / generate.c   # Autoregressive text generation
│   │   ├── weight_loader.h / .c      # GGUF → model weight loading
│   │   └── thread_pool.h / .c        # pthread-based GEMV parallelism
│   ├── main.c               # CLI inference tool
│   ├── wasm_api.c            # Emscripten WASM entry points
│   ├── bench/
│   │   └── benchmark.c       # GEMV benchmark harness
│   └── test/
│       └── test_inference.c   # 21 tests: tokenizer, model, e2e
├── web/
│   ├── bitnet.js             # Integration API (ES module)
│   ├── worker.js             # Web Worker bridge to WASM
│   ├── inference.html        # Demo UI (uses bitnet.js)
│   ├── index.html            # GEMV kernel benchmark UI
│   └── serve.py              # Dev server with COOP/COEP headers
├── docker/
│   └── docker-compose.yml    # Emscripten build + test containers
├── CMakeLists.txt            # Emscripten + native build config
└── CLAUDE.md                 # Architecture decisions & dev notes
```

### Kernel Variants

| Kernel | Packing | Inner Loop | SIMD Strategy |
|---|---|---|---|
| **I2_S scalar** | 2-bit (4 weights/byte) | Unpack → multiply → accumulate | — |
| **I2_S SIMD** | 2-bit (4 weights/byte) | `wasm_i16x8_mul` + int32 accum | Row-major, 16 weights/iter |
| **TL1 scalar** | 4-bit index (2 pairs/byte) | LUT lookup per pair | — |
| **TL1 SIMD** | 4-bit index (column-major) | `wasm_i8x16_swizzle` × 2 | 16 rows/iter, int16 accum |

## Build & Run

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) (for Emscripten builds)
- A browser with [WASM SIMD support](https://caniuse.com/wasm-simd) (Chrome 91+, Firefox 89+, Safari 16.4+)

### Build with Docker

```bash
# Build WASM (uses emscripten/emsdk Docker image)
docker compose -f docker/docker-compose.yml run --rm build

# Serve the build directory
cd build
python -m http.server 8080

# Open http://localhost:8080 in your browser
```

### Build without Docker

If you have [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) installed locally:

```bash
mkdir build && cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
emmake make

python -m http.server 8080
```

### Native Build (scalar only, for testing)

```bash
mkdir build-native && cd build-native
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./benchmark
```

## Architecture

```
                  ┌─────────────────┐
                  │  benchmark.c    │  Harness: warmup, timing, correctness
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │    gemv.c       │  Unified GEMV dispatcher + quantization
                  └────────┬────────┘
                     ┌─────┴─────┐
              ┌──────▼──┐   ┌────▼─────┐
              │  i2s.c  │   │  tl1.c   │  Kernel implementations
              └─────────┘   └──────────┘
                     │           │
              ┌──────▼───────────▼──────┐
              │     simd_utils.h        │  WASM SIMD wrappers
              └─────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  wasm_simd128.h         │  Emscripten SIMD intrinsics
              └─────────────────────────┘
```

### Data Flow (TL1 SIMD GEMV)

```
Raw weights {-1,0,1}
    │
    ▼ tl1_pack_weights()
Nibble-packed indices (row-major)
    │
    ▼ tl1_transpose_weights()
Column-major indices ─────────────────┐
                                      │
Float activations                     │
    │                                 │
    ▼ quantize_activations()          │
Int8 activations                      │
    │                                 │
    ▼ tl1_build_lut()                 │
Int16 LUT (16 entries × K/2 pairs)   │
    │                                 │
    ▼ pre-split lo/hi bytes           │
LUT lo-bytes + LUT hi-bytes          │
    │                                 │
    ▼ tl1_gemv_simd() ◄──────────────┘
    │
    │  For each 16-row block:
    │    v128_load indices (column-major)
    │    extract lo/hi nibbles
    │    swizzle(lut_lo, idx) → 16 lo-byte results
    │    swizzle(lut_hi, idx) → 16 hi-byte results
    │    interleave → int16, accumulate
    │    flush to int32 every 64 iters
    │
    ▼
Float output (scaled)
```

## Optimization History

| Commit | Change | Time | Speedup |
|---|---|---|---|
| Baseline | I2_S scalar reference | 2.156ms | 1.00x |
| `19383c9` | Column-major swizzle + pre-split LUT + int16 accum | 0.330ms | 6.47x |
| `326b583` | Column-major weight transpose (eliminate gather) | 0.170ms | 12.68x |

## Integration API

The `bitnet.js` ES module provides a simple API for embedding BitNet inference in web applications — no manual Worker management or postMessage handling required.

```javascript
import { BitNet } from './bitnet.js';

const bitnet = new BitNet();

// Load model from URL or ArrayBuffer
await bitnet.load('/models/bitnet-2b.gguf', {
    threads: 5,
    onProgress: ({ phase, message, percent }) => {
        console.log(`${phase}: ${message}`);
    }
});

// Generate with streaming tokens
const result = await bitnet.generate('What is the capital of France?', {
    maxTokens: 64,
    temperature: 0.0,
    onToken: (piece) => process.stdout.write(piece)
});

console.log(result.text);              // full generated text
console.log(result.tokenCount);        // number of tokens generated
console.log(result.stats.decodeTokPerSec); // decode throughput

// Abort support via AbortSignal
const controller = new AbortController();
bitnet.generate('Hello', { signal: controller.signal });
controller.abort(); // cancels generation

// Clean up
bitnet.destroy();
```

### API Reference

| Method | Description |
|---|---|
| `new BitNet({ workerUrl })` | Create instance (default worker: `./worker.js`) |
| `bitnet.load(source, options)` | Load model from URL string or ArrayBuffer |
| `bitnet.generate(prompt, options)` | Generate text, returns `{ text, tokenCount, stats }` |
| `bitnet.setThreads(n)` | Change worker thread count (0 = single-threaded) |
| `bitnet.abort()` | Cancel in-flight generation |
| `bitnet.destroy()` | Release model memory and terminate worker |
| `bitnet.loaded` | Whether a model is loaded (read-only) |
| `bitnet.config` | Model config: `{ nLayers, hiddenSize, vocabSize, nHeads, maxSeqLen, nThreads }` |
| `bitnet.generating` | Whether generation is in progress (read-only) |

### Files Required

Copy these files alongside your application:
- `bitnet.js` — Integration API (ES module)
- `worker.js` — Web Worker bridge
- `bitnet-inference.js` — Emscripten glue code (from build)
- `bitnet-inference.wasm` — Compiled WASM binary (from build)

Your server must set [COOP/COEP headers](https://web.dev/cross-origin-isolation-guide/) for SharedArrayBuffer threading support. See `web/serve.py` for an example.

## Roadmap

- [x] Phase 1: GEMV kernels (I2_S + TL1, scalar + SIMD)
- [x] Phase 1.5: TL1 SIMD optimization (12.68x achieved)
- [x] Full inference pipeline: GGUF model loading, BPE tokenizer, transformer forward pass, sampling
- [x] Web Worker parallelism via SharedArrayBuffer (~11 tok/s with 5 workers)
- [x] Integration API for embedding in web applications (`bitnet.js`)
- [x] TL2 kernel: 3 weights per index (1-bit sign + 4-bit index)

## Technical References

- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880) — TL1/TL2 kernel design
- [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) — Target model
- [Microsoft bitnet.cpp](https://github.com/microsoft/BitNet) — Native reference implementation
- [WASM SIMD Specification](https://github.com/WebAssembly/simd)
- [Emscripten SIMD Guide](https://emscripten.org/docs/porting/simd.html)

## License

MIT
