#include "tl1.h"
#include "simd_utils.h"
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

void tl1_build_lut(int8_t *lut, const int8_t *x, int32_t K) {
    /*
     * For each pair of activations (a0, a1), build a 16-byte LUT.
     *
     * The 9 valid entries:
     *   [0] (-1,-1) = -(a0 + a1)  clamp to int8
     *   [1] (-1, 0) = -a0
     *   [2] (-1,+1) = -a0 + a1
     *   [3] ( 0,-1) = -a1
     *   [4] ( 0, 0) = 0
     *   [5] ( 0,+1) = a1
     *   [6] (+1,-1) = a0 - a1
     *   [7] (+1, 0) = a0
     *   [8] (+1,+1) = a0 + a1
     *   [9..15]     = 0 (padding for 16-byte alignment)
     *
     * We store as int8 to fit in a single v128 register.
     * For large activations, results are clamped to [-128, 127].
     * Accumulation uses int32 to avoid overflow.
     */
    int32_t num_pairs = K / 2;

    for (int32_t p = 0; p < num_pairs; p++) {
        int16_t a0 = (int16_t)x[p * 2];
        int16_t a1 = (int16_t)x[p * 2 + 1];
        int8_t *entry = &lut[p * 16];

        /* Compute and clamp to int8 range */
        entry[0] = (int8_t)(-(a0 + a1) < -128 ? -128 :
                            (-(a0 + a1) > 127 ? 127 : -(a0 + a1)));
        entry[1] = (int8_t)(-a0);
        entry[2] = (int8_t)(-a0 + a1 < -128 ? -128 :
                            (-a0 + a1 > 127 ? 127 : -a0 + a1));
        entry[3] = (int8_t)(-a1);
        entry[4] = 0;
        entry[5] = (int8_t)(a1);
        entry[6] = (int8_t)(a0 - a1 < -128 ? -128 :
                            (a0 - a1 > 127 ? 127 : a0 - a1));
        entry[7] = (int8_t)(a0);
        entry[8] = (int8_t)(a0 + a1 < -128 ? -128 :
                            (a0 + a1 > 127 ? 127 : a0 + a1));

        /* Zero-pad entries 9-15 */
        memset(&entry[9], 0, 7);
    }
}

/* --- Scalar GEMV --- */

void tl1_gemv_scalar(const tl1_weight_t *W,
                     const int8_t *lut,
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
                   const int8_t *lut,
                   const activation_t *x,
                   output_t *y) {
    int32_t M = W->M;
    int32_t K = W->K;
    int32_t num_pairs = K / 2;
    int32_t bytes_per_row = (num_pairs + 1) / 2;
    float scale = W->scale * x->scale;

    for (int32_t i = 0; i < M; i++) {
        v128_t acc = simd_zero();
        const uint8_t *row_indices = &W->indices[i * bytes_per_row];

        /*
         * Process 16 weight pairs per iteration.
         * 16 pairs = 8 packed bytes of indices.
         *
         * For each pair p, we need:
         *   1. Load the 16-byte LUT for that activation pair
         *   2. Use the weight index to select the right entry
         *
         * But swizzle operates on a SINGLE 16-byte table with
         * 16 index bytes. So we process pairs that share the
         * same LUT entry together.
         *
         * Strategy: for each activation pair, load its 16-byte
         * LUT, gather the indices for that pair from all M rows,
         * and swizzle. But that's column-major.
         *
         * Row-major strategy (what we use):
         * Process 16 consecutive pairs. Each has its OWN 16-byte
         * LUT. We do 16 individual swizzle lookups and sum results.
         *
         * Optimization: batch the LUT lookups and accumulate
         * the int8 results into int32 to avoid overflow.
         */
        int32_t p;
        for (p = 0; p + 16 <= num_pairs; p += 16) {
            /* Load 8 bytes = 16 nibble-packed indices */
            /* We need to unpack nibbles into 16 separate bytes */
            const uint8_t *idx_ptr = &row_indices[p / 2];

            /*
             * Unpack 8 packed bytes -> 16 indices (one per pair).
             * Each byte has low nibble (even pair) and high nibble (odd pair).
             */
            v128_t packed_indices = wasm_v128_load64_zero(idx_ptr);

            /* Interleave: create vector where each byte is one 4-bit index */
            /* Low nibbles = even pairs, high nibbles = odd pairs */
            v128_t lo_nibs = simd_extract_low_nibbles(packed_indices);
            v128_t hi_nibs = simd_extract_high_nibbles(packed_indices);

            /*
             * Now do 16 individual lookups.
             * Each pair p+k has its LUT at lut[(p+k)*16].
             *
             * For each, we do:
             *   lut_val = lut_table_k[index_k]
             * and accumulate.
             *
             * Since each pair has a DIFFERENT LUT, we can't batch
             * the swizzle across pairs. But we CAN use swizzle
             * to do a single lookup per pair efficiently.
             *
             * We broadcast the index to all lanes, load the LUT,
             * swizzle, and extract lane 0. But that's wasteful.
             *
             * Better: accumulate scalar results from the LUT.
             * The SIMD win comes from processing multiple OUTPUT
             * ROWS in parallel (not multiple pairs).
             *
             * For now: scalar inner loop, SIMD accumulation.
             */
            int32_t batch_acc = 0;
            for (int32_t k = 0; k < 16; k++) {
                uint8_t idx_byte = idx_ptr[k / 2];
                uint8_t idx = (k & 1) ? (idx_byte >> 4) : (idx_byte & 0x0F);
                batch_acc += (int32_t)lut[(p + k) * 16 + idx];
            }

            /* Accumulate batch into SIMD register lane 0 */
            v128_t batch_v = wasm_i32x4_make(batch_acc, 0, 0, 0);
            acc = simd_add_i32(acc, batch_v);
        }

        /* Remaining pairs */
        int32_t scalar_acc = 0;
        for (; p < num_pairs; p++) {
            uint8_t packed = row_indices[p / 2];
            uint8_t idx = (p & 1) ? (packed >> 4) : (packed & 0x0F);
            scalar_acc += (int32_t)lut[p * 16 + idx];
        }

        int32_t total = wasm_i32x4_extract_lane(acc, 0) + scalar_acc;
        y->data[i] = (float)total * scale;
    }
}

#else

void tl1_gemv_simd(const tl1_weight_t *W,
                   const int8_t *lut,
                   const activation_t *x,
                   output_t *y) {
    tl1_gemv_scalar(W, lut, x, y);
}

#endif /* __wasm_simd128__ */
