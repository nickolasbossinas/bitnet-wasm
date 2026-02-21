#ifndef BITNET_I2S_H
#define BITNET_I2S_H

#include "types.h"

/*
 * I2_S Kernel: Int2 with Scale
 *
 * The simpler BitNet kernel. Each ternary weight {-1, 0, 1} is stored
 * as 2 bits. At runtime, weights are unpacked and multiplied with
 * int8 activations using standard multiply-accumulate.
 *
 * Weight encoding (2 bits):
 *   00 = -1
 *   01 =  0
 *   10 = +1
 *   11 = unused
 *
 * Packing: 4 weights per byte, LSB first
 *   byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
 */

/* Pack 4 ternary weights into a single byte */
static inline uint8_t i2s_pack4(int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    return (uint8_t)(
        ((w0 + 1) & 0x03)       |
        (((w1 + 1) & 0x03) << 2) |
        (((w2 + 1) & 0x03) << 4) |
        (((w3 + 1) & 0x03) << 6)
    );
}

/* Unpack 4 ternary weights from a byte */
static inline void i2s_unpack4(uint8_t packed, int8_t out[4]) {
    out[0] = (int8_t)((packed & 0x03))       - 1;
    out[1] = (int8_t)((packed >> 2) & 0x03)  - 1;
    out[2] = (int8_t)((packed >> 4) & 0x03)  - 1;
    out[3] = (int8_t)((packed >> 6) & 0x03)  - 1;
}

/*
 * Pack a full weight matrix into I2_S format.
 *
 * weights: M x K matrix of ternary values {-1, 0, 1} (row-major)
 * out:     packed buffer, size = M * ceil(K/4) bytes
 */
void i2s_pack_weights(const int8_t *weights, uint8_t *out,
                      int32_t M, int32_t K);

/*
 * I2_S GEMV: y = W * x (scalar implementation)
 *
 * Computes matrix-vector product with 2-bit packed ternary weights
 * and int8 quantized activations.
 *
 * output[i] = (sum_j(W[i][j] * x[j]) * w_scale * a_scale)
 */
void i2s_gemv_scalar(const i2s_weight_t *W,
                     const activation_t *x,
                     output_t *y);

/*
 * I2_S GEMV: WASM SIMD implementation
 *
 * Unpacks 2-bit weights in groups of 16, sign-extends to int16,
 * multiplies with int16 activations, accumulates in int32.
 */
void i2s_gemv_simd(const i2s_weight_t *W,
                   const activation_t *x,
                   output_t *y);

#endif /* BITNET_I2S_H */
