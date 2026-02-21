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
 * I2_S encoding (2 bits per weight, 4 per byte, LSB first):
 *   00 = -1, 01 = 0, 10 = +1, 11 = unused
 *   byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
 *
 * data:       packed I2_S bytes
 * out:        output int8 array, length n_elements
 * n_elements: total number of weights to decode
 */
void i2s_decode(const uint8_t *data, int8_t *out, int64_t n_elements);

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
