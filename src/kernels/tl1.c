#include "tl1.h"
#include "simd_utils.h"
#include <stdlib.h>
#include <string.h>

/*
 * TL1 Kernel Implementation
 *
 * The key insight: for ternary weights {-1, 0, 1}, multiplication
 * reduces to addition/subtraction/zero. A pair of weights has only
 * 9 possible outcomes. We pre-compute all 9 outcomes for each
 * activation pair into a 16-byte lookup table (padded from 9),
 * then use PSHUFB/swizzle to select the right one.
 *
 * This turns the entire GEMV inner loop into:
 *   load indices -> load LUT -> swizzle -> accumulate
 *
 * No multiplications at all.
 */

/* --- Weight Packing --- */

static inline uint8_t tl1_pair_index(int8_t w0, int8_t w1) {
    return (uint8_t)((w0 + 1) * 3 + (w1 + 1));
}

void tl1_pack_weights(const int8_t *weights, uint8_t *out,
                      int32_t M, int32_t K) {
    /*
     * Pack pairs of weights into nibbles.
     * Each byte stores 2 pair-indices (= 4 weights total):
     *   low nibble  = index for (w[j], w[j+1])
     *   high nibble = index for (w[j+2], w[j+3])
     */
    int32_t pairs_per_row = K / 2;
    int32_t bytes_per_row = (pairs_per_row + 1) / 2;

    for (int32_t i = 0; i < M; i++) {
        const int8_t *row = &weights[i * K];
        uint8_t *out_row = &out[i * bytes_per_row];

        for (int32_t p = 0; p < pairs_per_row; p += 2) {
            uint8_t lo = tl1_pair_index(row[p * 2], row[p * 2 + 1]);
            uint8_t hi = 0;
            if (p + 1 < pairs_per_row) {
                hi = tl1_pair_index(row[(p + 1) * 2], row[(p + 1) * 2 + 1]);
            }
            out_row[p / 2] = lo | (hi << 4);
        }
    }
}

/* --- LUT Construction --- */

void tl1_build_lut(int16_t *lut, const int8_t *x, int32_t K) {
    /*
     * For each pair of activations (a0, a1), build a 16-entry LUT.
     *
     * The 9 valid entries:
     *   [0] (-1,-1) = -(a0 + a1)
     *   [1] (-1, 0) = -a0
     *   [2] (-1,+1) = -a0 + a1
     *   [3] ( 0,-1) = -a1
     *   [4] ( 0, 0) = 0
     *   [5] ( 0,+1) = a1
     *   [6] (+1,-1) = a0 - a1
     *   [7] (+1, 0) = a0
     *   [8] (+1,+1) = a0 + a1
     *   [9..15]     = 0 (padding)
     *
     * Stored as int16 to avoid clamping (a0+a1 can be up to 254).
     * Accumulation uses int32 to avoid overflow.
     */
    int32_t num_pairs = K / 2;

    for (int32_t p = 0; p < num_pairs; p++) {
        int16_t a0 = (int16_t)x[p * 2];
        int16_t a1 = (int16_t)x[p * 2 + 1];
        int16_t *entry = &lut[p * 16];

        entry[0] = -(a0 + a1);
        entry[1] = -a0;
        entry[2] = -a0 + a1;
        entry[3] = -a1;
        entry[4] = 0;
        entry[5] = a1;
        entry[6] = a0 - a1;
        entry[7] = a0;
        entry[8] = a0 + a1;

        /* Zero-pad entries 9-15 */
        memset(&entry[9], 0, 7 * sizeof(int16_t));
    }
}

/* --- Scalar GEMV --- */

void tl1_gemv_scalar(const tl1_weight_t *W,
                     const int16_t *lut,
                     const activation_t *x,
                     output_t *y) {
    int32_t M = W->M;
    int32_t K = W->K;
    int32_t num_pairs = K / 2;
    int32_t bytes_per_row = (num_pairs + 1) / 2;
    float scale = W->scale * x->scale;

    for (int32_t i = 0; i < M; i++) {
        int32_t acc = 0;
        const uint8_t *row_indices = &W->indices[i * bytes_per_row];

        for (int32_t p = 0; p < num_pairs; p++) {
            /* Extract 4-bit index for this weight pair */
            uint8_t packed = row_indices[p / 2];
            uint8_t idx = (p & 1) ? (packed >> 4) : (packed & 0x0F);

            /* Look up the pre-computed partial sum */
            acc += (int32_t)lut[p * 16 + idx];
        }

        y->data[i] = (float)acc * scale;
    }
}

/* --- WASM SIMD GEMV --- */

#ifdef __wasm_simd128__

void tl1_gemv_simd(const tl1_weight_t *W,
                   const int16_t *lut,
                   const activation_t *x,
                   output_t *y) {
    int32_t M = W->M;
    int32_t K = W->K;
    int32_t num_pairs = K / 2;
    int32_t bytes_per_row = (num_pairs + 1) / 2;
    int32_t full_bytes = num_pairs / 2;
    float scale = W->scale * x->scale;

    /*
     * Pre-split LUT: deinterleave int16 LUT into separate lo-byte and
     * hi-byte tables. This moves the 4 deinterleave shuffles per byte
     * iteration out of the hot inner loop into a one-time O(K) prepass.
     *
     * Each activation pair's 16 int16 entries (32 bytes) become:
     *   lut_lo[pair*16 .. pair*16+15] = low bytes of each int16
     *   lut_hi[pair*16 .. pair*16+15] = high bytes of each int16
     */
    uint8_t *lut_lo_bytes = (uint8_t *)malloc(num_pairs * 16);
    uint8_t *lut_hi_bytes = (uint8_t *)malloc(num_pairs * 16);

    for (int32_t p = 0; p < num_pairs; p++) {
        v128_t raw0 = wasm_v128_load(&lut[p * 16]);
        v128_t raw1 = wasm_v128_load(&lut[p * 16 + 8]);
        wasm_v128_store(&lut_lo_bytes[p * 16], wasm_i8x16_shuffle(raw0, raw1,
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
        wasm_v128_store(&lut_hi_bytes[p * 16], wasm_i8x16_shuffle(raw0, raw1,
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31));
    }

    /*
     * Column-major swizzle with pre-split LUT and int16 accumulation.
     *
     * Inner loop per byte iteration (2 pairs, 16 rows):
     *   1. Gather 16 packed index bytes (one per row)
     *   2. Extract lo/hi nibbles for even/odd pair indices
     *   3. Load pre-split lo/hi byte tables (2 loads, no deinterleave)
     *   4. swizzle × 2 for 16 lookups each
     *   5. Interleave lo/hi → int16, accumulate with i16x8_add
     *
     * Int16 accumulators flush to int32 every 64 byte iterations.
     * Overflow budget: 64 iters × 2 pairs × 254 max = 32,512 < 32,767.
     */
    int32_t i;
    for (i = 0; i + 16 <= M; i += 16) {
        v128_t acc0 = simd_zero();  /* int32: rows i+0  .. i+3  */
        v128_t acc1 = simd_zero();  /* int32: rows i+4  .. i+7  */
        v128_t acc2 = simd_zero();  /* int32: rows i+8  .. i+11 */
        v128_t acc3 = simd_zero();  /* int32: rows i+12 .. i+15 */

        const uint8_t *rows[16];
        for (int32_t r = 0; r < 16; r++)
            rows[r] = &W->indices[(i + r) * bytes_per_row];

        /* Two-level loop: outer chunks of 64, inner accumulates int16 */
        for (int32_t b_outer = 0; b_outer < full_bytes; b_outer += 64) {
            int32_t b_end = b_outer + 64;
            if (b_end > full_bytes) b_end = full_bytes;

            v128_t acc16_lo = simd_zero();  /* int16: rows 0..7  */
            v128_t acc16_hi = simd_zero();  /* int16: rows 8..15 */

            for (int32_t b = b_outer; b < b_end; b++) {
                /* Gather 16 packed bytes (one per row) */
                v128_t packed = wasm_i8x16_make(
                    rows[0][b],  rows[1][b],  rows[2][b],  rows[3][b],
                    rows[4][b],  rows[5][b],  rows[6][b],  rows[7][b],
                    rows[8][b],  rows[9][b],  rows[10][b], rows[11][b],
                    rows[12][b], rows[13][b], rows[14][b], rows[15][b]
                );

                /* --- Even pair (low nibble) --- */
                v128_t idx_even = simd_extract_low_nibbles(packed);
                int32_t ep = b * 2;

                v128_t lut_lo = wasm_v128_load(&lut_lo_bytes[ep * 16]);
                v128_t lut_hi = wasm_v128_load(&lut_hi_bytes[ep * 16]);

                v128_t val_lo = wasm_i8x16_swizzle(lut_lo, idx_even);
                v128_t val_hi = wasm_i8x16_swizzle(lut_hi, idx_even);

                acc16_lo = wasm_i16x8_add(acc16_lo, wasm_i8x16_shuffle(
                    val_lo, val_hi,
                    0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23));
                acc16_hi = wasm_i16x8_add(acc16_hi, wasm_i8x16_shuffle(
                    val_lo, val_hi,
                    8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31));

                /* --- Odd pair (high nibble) --- */
                v128_t idx_odd = simd_extract_high_nibbles(packed);
                int32_t op = b * 2 + 1;

                v128_t lut_lo2 = wasm_v128_load(&lut_lo_bytes[op * 16]);
                v128_t lut_hi2 = wasm_v128_load(&lut_hi_bytes[op * 16]);

                v128_t val_lo2 = wasm_i8x16_swizzle(lut_lo2, idx_odd);
                v128_t val_hi2 = wasm_i8x16_swizzle(lut_hi2, idx_odd);

                acc16_lo = wasm_i16x8_add(acc16_lo, wasm_i8x16_shuffle(
                    val_lo2, val_hi2,
                    0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23));
                acc16_hi = wasm_i16x8_add(acc16_hi, wasm_i8x16_shuffle(
                    val_lo2, val_hi2,
                    8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31));
            }

            /* Flush int16 accumulators to int32 */
            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_extend_low_i16x8(acc16_lo));
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_extend_high_i16x8(acc16_lo));
            acc2 = wasm_i32x4_add(acc2, wasm_i32x4_extend_low_i16x8(acc16_hi));
            acc3 = wasm_i32x4_add(acc3, wasm_i32x4_extend_high_i16x8(acc16_hi));
        }

        /* Handle last byte if num_pairs is odd */
        if (num_pairs & 1) {
            int32_t b = full_bytes;
            v128_t packed = wasm_i8x16_make(
                rows[0][b],  rows[1][b],  rows[2][b],  rows[3][b],
                rows[4][b],  rows[5][b],  rows[6][b],  rows[7][b],
                rows[8][b],  rows[9][b],  rows[10][b], rows[11][b],
                rows[12][b], rows[13][b], rows[14][b], rows[15][b]
            );
            v128_t idx_even = simd_extract_low_nibbles(packed);
            int32_t ep = b * 2;

            v128_t lut_lo = wasm_v128_load(&lut_lo_bytes[ep * 16]);
            v128_t lut_hi = wasm_v128_load(&lut_hi_bytes[ep * 16]);

            v128_t val_lo = wasm_i8x16_swizzle(lut_lo, idx_even);
            v128_t val_hi = wasm_i8x16_swizzle(lut_hi, idx_even);

            v128_t pairs_0_7 = wasm_i8x16_shuffle(val_lo, val_hi,
                0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            v128_t pairs_8_15 = wasm_i8x16_shuffle(val_lo, val_hi,
                8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_extend_low_i16x8(pairs_0_7));
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_extend_high_i16x8(pairs_0_7));
            acc2 = wasm_i32x4_add(acc2, wasm_i32x4_extend_low_i16x8(pairs_8_15));
            acc3 = wasm_i32x4_add(acc3, wasm_i32x4_extend_high_i16x8(pairs_8_15));
        }

        /* Scale and store 16 results */
        v128_t sv = wasm_f32x4_splat(scale);
        wasm_v128_store(&y->data[i],
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(acc0), sv));
        wasm_v128_store(&y->data[i + 4],
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(acc1), sv));
        wasm_v128_store(&y->data[i + 8],
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(acc2), sv));
        wasm_v128_store(&y->data[i + 12],
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(acc3), sv));
    }

    /* Handle remaining rows (< 16) with scalar fallback */
    for (; i < M; i++) {
        int32_t acc = 0;
        const uint8_t *row_indices = &W->indices[i * bytes_per_row];

        for (int32_t b = 0; b < full_bytes; b++) {
            uint8_t packed = row_indices[b];
            acc += (int32_t)lut[b * 32 + (packed & 0x0F)]
                 + (int32_t)lut[b * 32 + 16 + (packed >> 4)];
        }
        if (num_pairs & 1) {
            acc += (int32_t)lut[full_bytes * 32 + (row_indices[full_bytes] & 0x0F)];
        }

        y->data[i] = (float)acc * scale;
    }

    free(lut_lo_bytes);
    free(lut_hi_bytes);
}

#else

void tl1_gemv_simd(const tl1_weight_t *W,
                   const int16_t *lut,
                   const activation_t *x,
                   output_t *y) {
    tl1_gemv_scalar(W, lut, x, y);
}

#endif /* __wasm_simd128__ */
