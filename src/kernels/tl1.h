#ifndef BITNET_TL1_H
#define BITNET_TL1_H

#include "types.h"

/*
 * TL1 Kernel: Ternary Lookup Table (2-weight, 4-bit index)
 *
 * Core idea: Replace multiply-accumulate with table lookups.
 *
 * For each pair of ternary weights (w0, w1), there are only
 * 3x3 = 9 possible combinations. We pre-compute all possible
 * partial sums of activations for these 9 combos into a LUT,
 * then use wasm_i8x16_swizzle (PSHUFB equivalent) to do
 * the "multiplication" as a single table lookup.
 *
 * This eliminates ALL multiplications from the GEMV inner loop.
 *
 * Weight pair index encoding:
 *   index = (w0 + 1) * 3 + (w1 + 1)
 *
 *   w0  w1  | index | operation
 *   -1  -1  |   0   | -(a0 + a1)
 *   -1   0  |   1   | -a0
 *   -1  +1  |   2   | -a0 + a1
 *    0  -1  |   3   | -a1
 *    0   0  |   4   | 0
 *    0  +1  |   5   | a1
 *   +1  -1  |   6   | a0 - a1
 *   +1   0  |   7   | a0
 *   +1  +1  |   8   | a0 + a1
 */

/*
 * Pack weight pairs into 4-bit indices (nibble-packed).
 *
 * weights: M x K ternary matrix {-1, 0, 1} (row-major)
 * out:     nibble-packed indices, size = M * ceil(K/4) bytes
 *          (2 weight-pair indices per byte: low nibble, high nibble)
 * K must be even.
 */
void tl1_pack_weights(const int8_t *weights, uint8_t *out,
                      int32_t M, int32_t K);

/*
 * Build the TL1 lookup table from int8 activations.
 *
 * For each pair of consecutive activations (a[2i], a[2i+1]),
 * compute the 9 possible partial sums and store in the LUT.
 *
 * lut:  output table, size = 16 * (K/2) bytes
 *       (16 entries per activation pair, padded from 9)
 *       Each entry is int16, stored as 2 bytes.
 * x:    int8 activation vector, length K
 * K:    must be even
 *
 * LUT layout per activation pair i:
 *   lut[i*16 + idx] = partial_sum(a[2i], a[2i+1], idx)
 *   where idx = (w0+1)*3 + (w1+1), entries 9-15 are zero-padded
 */
void tl1_build_lut(int8_t *lut, const int8_t *x, int32_t K);

/*
 * TL1 GEMV: y = W * x (scalar implementation)
 */
void tl1_gemv_scalar(const tl1_weight_t *W,
                     const int8_t *lut,
                     const activation_t *x,
                     output_t *y);

/*
 * TL1 GEMV: y = W * x (WASM SIMD implementation)
 *
 * Uses wasm_i8x16_swizzle for table lookup — this is the
 * critical path that maps directly to PSHUFB on x86 and
 * VTBL on ARM, giving near-native performance.
 *
 * Inner loop (per output row):
 *   1. Load 16 weight-pair indices
 *   2. Load corresponding LUT entries (16 bytes)
 *   3. swizzle(lut, indices) -> 16 partial sums
 *   4. Accumulate into int16/int32
 */
void tl1_gemv_simd(const tl1_weight_t *W,
                   const int8_t *lut,
                   const activation_t *x,
                   output_t *y);

#endif /* BITNET_TL1_H */
