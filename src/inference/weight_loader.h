#ifndef BITNET_WEIGHT_LOADER_H
#define BITNET_WEIGHT_LOADER_H

#include "model.h"
#include "gguf.h"

/*
 * GGUF Weight Loader
 *
 * Loads model weights from a parsed GGUF file into model_t.
 * Handles:
 *   - I2_S ternary weights (2-bit packed) -> decode -> TL1 repack
 *   - F16 token embeddings -> F32 conversion
 *   - F32 norm weights -> direct copy
 */

/*
 * Convert IEEE 754 half-precision float to single-precision.
 * Portable: uses bit manipulation, no hardware FP16.
 */
float f16_to_f32(uint16_t h);

/*
 * Decode I2_S packed ternary weights to int8 {-1, 0, 1}.
 *
 * I2_S encoding (GGML_TYPE_I2_S = 36, used by BitNet b1.58):
 *   Value codes: 00 = -1, 01 = 0, 10 = +1
 *
 * Layout: interleaved 128-weight blocks (32 bytes each).
 * Within each block, each byte packs 4 weights from different groups:
 *   bits 6-7: group 0 (weights 0..31)
 *   bits 4-5: group 1 (weights 32..63)
 *   bits 2-3: group 2 (weights 64..95)
 *   bits 0-1: group 3 (weights 96..127)
 *
 * Tail elements (< 128) are packed MSB-first sequentially.
 *
 * A per-tensor float32 scale factor is appended at byte offset
 * ceil(n_elements/4) after the packed data. The caller (load_tl1_weight)
 * reads this scale and stores it in tl1_weight_t.scale.
 *
 * data:       packed I2_S bytes
 * out:        output int8 array, length n_elements
 * n_elements: total number of weights to decode
 */
void i2s_decode(const uint8_t *data, int8_t *out, int64_t n_elements);

/*
 * Decode TRIT5 packed ternary weights to int8 {-1, 0, 1}.
 *
 * TRIT5 encoding: 5 ternary values per byte using base-3.
 *   Each weight: -1→0, 0→1, +1→2
 *   byte = w0*81 + w1*27 + w2*9 + w3*3 + w4 (range 0..242)
 *
 * Per-tensor float32 scale appended at byte offset ceil(n_elements/5).
 */
void trit5_decode(const uint8_t *data, int8_t *out, int64_t n_elements);

/*
 * Load all model weights from a parsed GGUF file.
 *
 * model:            pre-allocated model (config set, model_alloc called)
 * gguf:             parsed GGUF context with tensor info
 * file_data:        raw GGUF file bytes
 * n_layers_to_load: number of layers to load (-1 = all)
 *
 * Returns 0 on success, -1 on error.
 */
int model_load_weights(model_t *model, const gguf_context_t *gguf,
                       const uint8_t *file_data, int32_t n_layers_to_load);

#endif /* BITNET_WEIGHT_LOADER_H */
