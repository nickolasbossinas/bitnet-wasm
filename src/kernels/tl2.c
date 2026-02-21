#include "tl2.h"
#include "simd_utils.h"
#include <stdlib.h>
#include <string.h>

/*
 * TL2 Kernel Implementation
 *
 * Extends TL1's approach from weight pairs to triples.
 * combined = w0*9 + w1*3 + w2 gives 27 unique values in [-13, +13].
 * Store as sign bit + 4-bit unsigned index (abs value).
 * LUT has 14 valid entries per activation triple, padded to 16.
 *
 * 33% fewer inner loop iterations than TL1 (K/3 vs K/2 lookups),
 * at the cost of sign bit handling overhead.
 */

/* --- Weight Packing --- */

void tl2_pack_weights(const int8_t *weights, uint8_t *indices,
                      uint8_t *signs, int32_t M, int32_t K) {
    /*
     * Pack triples of weights into nibbles + sign bits.
     * Each byte stores 2 triple-indices:
     *   low nibble  = abs(combined) for triple i
     *   high nibble = abs(combined) for triple i+1
     * Sign byte stores:
     *   bit 0 = sign of even triple
     *   bit 1 = sign of odd triple
     */
    int32_t triples_per_row = K / 3;
    int32_t bytes_per_row = (triples_per_row + 1) / 2;

    for (int32_t i = 0; i < M; i++) {
        const int8_t *row = &weights[i * K];
        uint8_t *idx_row = &indices[i * bytes_per_row];
        uint8_t *sgn_row = &signs[i * bytes_per_row];

        for (int32_t t = 0; t < triples_per_row; t += 2) {
            /* Even triple */
            int32_t j0 = t * 3;
            int32_t combined0 = row[j0] * 9 + row[j0 + 1] * 3 + row[j0 + 2];
            uint8_t sign0 = (combined0 < 0) ? 1 : 0;
            uint8_t idx0 = (uint8_t)(combined0 < 0 ? -combined0 : combined0);

            /* Odd triple */
            uint8_t idx1 = 0;
            uint8_t sign1 = 0;
            if (t + 1 < triples_per_row) {
                int32_t j1 = (t + 1) * 3;
                int32_t combined1 = row[j1] * 9 + row[j1 + 1] * 3 + row[j1 + 2];
                sign1 = (combined1 < 0) ? 1 : 0;
                idx1 = (uint8_t)(combined1 < 0 ? -combined1 : combined1);
            }

            idx_row[t / 2] = idx0 | (idx1 << 4);
            sgn_row[t / 2] = sign0 | (sign1 << 1);
        }
    }
}

/* --- Weight Transpose (row-major -> column-major) --- */

void tl2_transpose_weights(tl2_weight_t *W) {
    int32_t M = W->M;
    int32_t triples = W->K / 3;
    int32_t bytes_per_row = (triples + 1) / 2;

    W->indices_col = (uint8_t *)malloc(M * bytes_per_row);
    W->signs_col = (uint8_t *)malloc(M * bytes_per_row);

    for (int32_t row = 0; row < M; row++) {
        for (int32_t b = 0; b < bytes_per_row; b++) {
            W->indices_col[b * M + row] = W->indices[row * bytes_per_row + b];
            W->signs_col[b * M + row] = W->signs[row * bytes_per_row + b];
        }
    }
}

/* --- LUT Construction --- */

void tl2_build_lut(int16_t *lut, const int8_t *x, int32_t K) {
    /*
     * For each triple of activations (a0, a1, a2), build a 16-entry LUT.
     *
     * Index = abs(w0*9 + w1*3 + w2), sign handled separately.
     * Each entry is the unsigned dot product for that index.
     *
     * LUT[0]  = 0                  (w=(0,0,0))
     * LUT[1]  = a2                 (w=(0,0,1))
     * LUT[2]  = a1 - a2            (w=(0,1,-1))
     * LUT[3]  = a1                 (w=(0,1,0))
     * LUT[4]  = a1 + a2            (w=(0,1,1))
     * LUT[5]  = a0 - a1 - a2       (w=(1,-1,-1))
     * LUT[6]  = a0 - a1            (w=(1,-1,0))
     * LUT[7]  = a0 - a1 + a2       (w=(1,-1,1))
     * LUT[8]  = a0 - a2            (w=(1,0,-1))
     * LUT[9]  = a0                 (w=(1,0,0))
     * LUT[10] = a0 + a2            (w=(1,0,1))
     * LUT[11] = a0 + a1 - a2       (w=(1,1,-1))
     * LUT[12] = a0 + a1            (w=(1,1,0))
     * LUT[13] = a0 + a1 + a2       (w=(1,1,1))
     * LUT[14..15] = 0 (padding)
     */
    int32_t num_triples = K / 3;

    for (int32_t t = 0; t < num_triples; t++) {
        int16_t a0 = (int16_t)x[t * 3];
        int16_t a1 = (int16_t)x[t * 3 + 1];
        int16_t a2 = (int16_t)x[t * 3 + 2];
        int16_t *entry = &lut[t * 16];

        entry[0]  = 0;
        entry[1]  = a2;
        entry[2]  = a1 - a2;
        entry[3]  = a1;
        entry[4]  = a1 + a2;
        entry[5]  = a0 - a1 - a2;
        entry[6]  = a0 - a1;
        entry[7]  = a0 - a1 + a2;
        entry[8]  = a0 - a2;
        entry[9]  = a0;
        entry[10] = a0 + a2;
        entry[11] = a0 + a1 - a2;
        entry[12] = a0 + a1;
        entry[13] = a0 + a1 + a2;

        /* Zero-pad entries 14-15 */
        entry[14] = 0;
        entry[15] = 0;
    }
}

/* --- Scalar GEMV --- */

void tl2_gemv_scalar(const tl2_weight_t *W,
                     const int16_t *lut,
                     const activation_t *x,
                     output_t *y) {
    int32_t M = W->M;
    int32_t K = W->K;
    int32_t num_triples = K / 3;
    int32_t bytes_per_row = (num_triples + 1) / 2;
    float scale = W->scale * x->scale;

    for (int32_t i = 0; i < M; i++) {
        int32_t acc = 0;
        const uint8_t *row_idx = &W->indices[i * bytes_per_row];
        const uint8_t *row_sgn = &W->signs[i * bytes_per_row];

        for (int32_t t = 0; t < num_triples; t++) {
            /* Extract 4-bit unsigned index */
            uint8_t packed_idx = row_idx[t / 2];
            uint8_t idx = (t & 1) ? (packed_idx >> 4) : (packed_idx & 0x0F);

            /* Extract sign bit */
            uint8_t packed_sgn = row_sgn[t / 2];
            uint8_t sign = (t & 1) ? ((packed_sgn >> 1) & 1) : (packed_sgn & 1);

            /* Lookup unsigned value and apply sign */
            int32_t val = (int32_t)lut[t * 16 + idx];
            if (sign) val = -val;
            acc += val;
        }

        y->data[i] = (float)acc * scale;
    }
}

/* --- WASM SIMD GEMV --- */

#ifdef __wasm_simd128__

void tl2_gemv_simd(const tl2_weight_t *W,
                   const int16_t *lut,
                   const activation_t *x,
                   output_t *y) {
    int32_t M = W->M;
    int32_t K = W->K;
    int32_t num_triples = K / 3;
    int32_t bytes_per_row = (num_triples + 1) / 2;
    int32_t full_bytes = num_triples / 2;
    float scale = W->scale * x->scale;

    /*
     * Pre-split LUT: deinterleave int16 LUT into lo/hi byte tables.
     * Same technique as TL1.
     */
    uint8_t *lut_lo_bytes = (uint8_t *)malloc(num_triples * 16);
    uint8_t *lut_hi_bytes = (uint8_t *)malloc(num_triples * 16);

    for (int32_t t = 0; t < num_triples; t++) {
        v128_t raw0 = wasm_v128_load(&lut[t * 16]);
        v128_t raw1 = wasm_v128_load(&lut[t * 16 + 8]);
        wasm_v128_store(&lut_lo_bytes[t * 16], wasm_i8x16_shuffle(raw0, raw1,
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
        wasm_v128_store(&lut_hi_bytes[t * 16], wasm_i8x16_shuffle(raw0, raw1,
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31));
    }

    /*
     * Column-major swizzle with sign handling.
     *
     * Same architecture as TL1 SIMD, but after lookup we apply
     * branchless sign negation: (val XOR mask) - mask
     *   mask=0x0000 -> val unchanged
     *   mask=0xFFFF -> ~val + 1 = -val
     *
     * Sign extraction per byte (2 triples, 16 rows):
     *   sign_byte AND 0x01 -> even triple signs (16 bytes, each 0 or 1)
     *   (sign_byte >> 1) AND 0x01 -> odd triple signs
     *   negate(sign) -> 0x00 or 0xFF
     *   extend to int16 -> 0x0000 or 0xFFFF
     *
     * Flush int16 to int32 every 32 byte iterations.
     * Overflow budget: 32 × 2 × 381 = 24,384 < 32,767. ✓
     */
    const uint8_t *col_idx = W->indices_col;
    const uint8_t *col_sgn = W->signs_col;
    v128_t mask_one = wasm_i8x16_splat(1);
    v128_t zero = simd_zero();

    int32_t i;
    for (i = 0; i + 16 <= M; i += 16) {
        v128_t acc0 = simd_zero();
        v128_t acc1 = simd_zero();
        v128_t acc2 = simd_zero();
        v128_t acc3 = simd_zero();

        for (int32_t b_outer = 0; b_outer < full_bytes; b_outer += 32) {
            int32_t b_end = b_outer + 32;
            if (b_end > full_bytes) b_end = full_bytes;

            v128_t acc16_lo = simd_zero();
            v128_t acc16_hi = simd_zero();

            for (int32_t b = b_outer; b < b_end; b++) {
                /* Load 16 packed index+sign bytes from column-major layout */
                v128_t packed_idx = wasm_v128_load(&col_idx[b * M + i]);
                v128_t packed_sgn = wasm_v128_load(&col_sgn[b * M + i]);

                /* --- Even triple (low nibble, bit 0 sign) --- */
                v128_t idx_even = simd_extract_low_nibbles(packed_idx);
                int32_t et = b * 2;  /* even triple index */

                v128_t lut_lo = wasm_v128_load(&lut_lo_bytes[et * 16]);
                v128_t lut_hi = wasm_v128_load(&lut_hi_bytes[et * 16]);

                v128_t val_lo = wasm_i8x16_swizzle(lut_lo, idx_even);
                v128_t val_hi = wasm_i8x16_swizzle(lut_hi, idx_even);

                /* Interleave lo/hi -> int16 */
                v128_t val16_lo = wasm_i8x16_shuffle(val_lo, val_hi,
                    0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                v128_t val16_hi = wasm_i8x16_shuffle(val_lo, val_hi,
                    8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

                /* Extract even sign bits -> negate mask */
                v128_t sgn_even = wasm_v128_and(packed_sgn, mask_one);
                v128_t neg_bytes = wasm_i8x16_sub(zero, sgn_even); /* 0x00 or 0xFF */
                v128_t neg_lo = wasm_i16x8_extend_low_i8x16(neg_bytes);
                v128_t neg_hi = wasm_i16x8_extend_high_i8x16(neg_bytes);

                /* Branchless negate: (val XOR mask) - mask */
                val16_lo = wasm_i16x8_sub(wasm_v128_xor(val16_lo, neg_lo), neg_lo);
                val16_hi = wasm_i16x8_sub(wasm_v128_xor(val16_hi, neg_hi), neg_hi);

                acc16_lo = wasm_i16x8_add(acc16_lo, val16_lo);
                acc16_hi = wasm_i16x8_add(acc16_hi, val16_hi);

                /* --- Odd triple (high nibble, bit 1 sign) --- */
                v128_t idx_odd = simd_extract_high_nibbles(packed_idx);
                int32_t ot = b * 2 + 1;  /* odd triple index */

                v128_t lut_lo2 = wasm_v128_load(&lut_lo_bytes[ot * 16]);
                v128_t lut_hi2 = wasm_v128_load(&lut_hi_bytes[ot * 16]);

                v128_t val_lo2 = wasm_i8x16_swizzle(lut_lo2, idx_odd);
                v128_t val_hi2 = wasm_i8x16_swizzle(lut_hi2, idx_odd);

                v128_t val16_lo2 = wasm_i8x16_shuffle(val_lo2, val_hi2,
                    0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                v128_t val16_hi2 = wasm_i8x16_shuffle(val_lo2, val_hi2,
                    8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

                /* Extract odd sign bits -> negate mask */
                v128_t sgn_odd = wasm_v128_and(wasm_u8x16_shr(packed_sgn, 1), mask_one);
                v128_t neg_bytes2 = wasm_i8x16_sub(zero, sgn_odd);
                v128_t neg_lo2 = wasm_i16x8_extend_low_i8x16(neg_bytes2);
                v128_t neg_hi2 = wasm_i16x8_extend_high_i8x16(neg_bytes2);

                val16_lo2 = wasm_i16x8_sub(wasm_v128_xor(val16_lo2, neg_lo2), neg_lo2);
                val16_hi2 = wasm_i16x8_sub(wasm_v128_xor(val16_hi2, neg_hi2), neg_hi2);

                acc16_lo = wasm_i16x8_add(acc16_lo, val16_lo2);
                acc16_hi = wasm_i16x8_add(acc16_hi, val16_hi2);
            }

            /* Flush int16 accumulators to int32 */
            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_extend_low_i16x8(acc16_lo));
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_extend_high_i16x8(acc16_lo));
            acc2 = wasm_i32x4_add(acc2, wasm_i32x4_extend_low_i16x8(acc16_hi));
            acc3 = wasm_i32x4_add(acc3, wasm_i32x4_extend_high_i16x8(acc16_hi));
        }

        /* Handle last byte if num_triples is odd */
        if (num_triples & 1) {
            int32_t b = full_bytes;
            v128_t packed_idx = wasm_v128_load(&col_idx[b * M + i]);
            v128_t packed_sgn = wasm_v128_load(&col_sgn[b * M + i]);

            v128_t idx_even = simd_extract_low_nibbles(packed_idx);
            int32_t et = b * 2;

            v128_t lut_lo = wasm_v128_load(&lut_lo_bytes[et * 16]);
            v128_t lut_hi = wasm_v128_load(&lut_hi_bytes[et * 16]);

            v128_t val_lo = wasm_i8x16_swizzle(lut_lo, idx_even);
            v128_t val_hi = wasm_i8x16_swizzle(lut_hi, idx_even);

            v128_t val16_lo = wasm_i8x16_shuffle(val_lo, val_hi,
                0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            v128_t val16_hi = wasm_i8x16_shuffle(val_lo, val_hi,
                8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

            v128_t sgn_even = wasm_v128_and(packed_sgn, mask_one);
            v128_t neg_bytes = wasm_i8x16_sub(zero, sgn_even);
            v128_t neg_lo = wasm_i16x8_extend_low_i8x16(neg_bytes);
            v128_t neg_hi = wasm_i16x8_extend_high_i8x16(neg_bytes);

            val16_lo = wasm_i16x8_sub(wasm_v128_xor(val16_lo, neg_lo), neg_lo);
            val16_hi = wasm_i16x8_sub(wasm_v128_xor(val16_hi, neg_hi), neg_hi);

            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_extend_low_i16x8(val16_lo));
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_extend_high_i16x8(val16_lo));
            acc2 = wasm_i32x4_add(acc2, wasm_i32x4_extend_low_i16x8(val16_hi));
            acc3 = wasm_i32x4_add(acc3, wasm_i32x4_extend_high_i16x8(val16_hi));
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
        const uint8_t *row_idx = &W->indices[i * bytes_per_row];
        const uint8_t *row_sgn = &W->signs[i * bytes_per_row];

        for (int32_t b = 0; b < full_bytes; b++) {
            uint8_t pi = row_idx[b];
            uint8_t ps = row_sgn[b];

            int32_t val0 = (int32_t)lut[b * 32 + (pi & 0x0F)];
            if (ps & 1) val0 = -val0;
            acc += val0;

            int32_t val1 = (int32_t)lut[b * 32 + 16 + (pi >> 4)];
            if (ps & 2) val1 = -val1;
            acc += val1;
        }
        if (num_triples & 1) {
            uint8_t pi = row_idx[full_bytes];
            uint8_t ps = row_sgn[full_bytes];
            int32_t val = (int32_t)lut[full_bytes * 32 + (pi & 0x0F)];
            if (ps & 1) val = -val;
            acc += val;
        }

        y->data[i] = (float)acc * scale;
    }

    free(lut_lo_bytes);
    free(lut_hi_bytes);
}

#else

void tl2_gemv_simd(const tl2_weight_t *W,
                   const int16_t *lut,
                   const activation_t *x,
                   output_t *y) {
    tl2_gemv_scalar(W, lut, x, y);
}

#endif /* __wasm_simd128__ */
