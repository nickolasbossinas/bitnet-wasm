#ifndef BITNET_TL2_H
#define BITNET_TL2_H

#include "types.h"

/*
 * TL2 Kernel: Ternary Lookup Table (3-weight, sign + 4-bit index)
 *
 * Like TL1, but processes weight TRIPLES instead of pairs.
 * Each triple of ternary weights (w0, w1, w2) is encoded as:
 *   combined = w0*9 + w1*3 + w2    (range [-13, +13])
 *   sign     = (combined < 0) ? 1 : 0
 *   index    = abs(combined)        (range [0, 13], fits in 4 bits)
 *
 * The 14 unsigned entries (0-13) fit in a 16-byte LUT with 2 padding zeros,
 * enabling the same wasm_i8x16_swizzle lookup as TL1. Sign is applied
 * after lookup via branchless negation: (val XOR mask) - mask.
 *
 * Benefits vs TL1:
 *   - 33% fewer inner loop iterations (K/3 vs K/2 lookups)
 *   - Trades sign handling overhead for fewer total ops
 *
 * K must be divisible by 6 (3 weights per triple, 2 triples per byte).
 */

/*
 * Pack weight triples into 4-bit unsigned indices + sign bits.
 *
 * weights:  M x K ternary matrix {-1, 0, 1} (row-major)
 * indices:  nibble-packed unsigned indices, 2 per byte
 * signs:    2 sign bits per byte (bit 0=even triple, bit 1=odd triple)
 * K must be divisible by 6.
 */
void tl2_pack_weights(const int8_t *weights, uint8_t *indices,
                      uint8_t *signs, int32_t M, int32_t K);

/*
 * Transpose packed indices and signs from row-major to column-major.
 * Allocates W->indices_col and W->signs_col.
 * Call once after tl2_pack_weights, before any SIMD calls.
 */
void tl2_transpose_weights(tl2_weight_t *W);

/*
 * Build the TL2 lookup table from int8 activations.
 *
 * For each triple of consecutive activations (a[3i], a[3i+1], a[3i+2]),
 * compute the 14 unsigned partial sums + 2 padding zeros.
 *
 * lut:  output table, size = 16 * (K/3) int16 entries
 * x:    int8 activation vector, length K
 * K:    must be divisible by 3
 */
void tl2_build_lut(int16_t *lut, const int8_t *x, int32_t K);

/*
 * TL2 GEMV: y = W * x (scalar implementation)
 */
void tl2_gemv_scalar(const tl2_weight_t *W,
                     const int16_t *lut,
                     const activation_t *x,
                     output_t *y);

/*
 * TL2 GEMV: y = W * x (WASM SIMD implementation)
 *
 * Same architecture as TL1 SIMD (column-major, pre-split LUT, int16 accum)
 * with added sign bit handling via branchless negation.
 */
void tl2_gemv_simd(const tl2_weight_t *W,
                   const int16_t *lut,
                   const activation_t *x,
                   output_t *y);

#endif /* BITNET_TL2_H */
