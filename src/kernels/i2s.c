#include "i2s.h"
#include "simd_utils.h"
#include <string.h>

/*
 * I2_S Kernel Implementation
 *
 * This is the simpler BitNet kernel used as a baseline.
 * Each ternary weight is stored as 2 bits, unpacked at runtime,
 * and multiplied with int8 activations.
 *
 * Performance: ~2-3x speedup over fp16 baseline (native).
 * Expected WASM: ~1.5-2x (due to 128-bit SIMD).
 */

void i2s_pack_weights(const int8_t *weights, uint8_t *out,
                      int32_t M, int32_t K) {
    int32_t packed_K = (K + 3) / 4; /* ceil(K/4) bytes per row */

    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < K; j += 4) {
            int8_t w[4] = {0, 0, 0, 0};
            for (int32_t k = 0; k < 4 && (j + k) < K; k++) {
                w[k] = weights[i * K + j + k];
            }
            out[i * packed_K + j / 4] = i2s_pack4(w[0], w[1], w[2], w[3]);
        }
    }
}

/* --- Scalar implementation --- */

void i2s_gemv_scalar(const i2s_weight_t *W,
                     const activation_t *x,
                     output_t *y) {
    int32_t M = W->M;
    int32_t K = W->K;
    int32_t packed_K = (K + 3) / 4;
    float scale = W->scale * x->scale;

    for (int32_t i = 0; i < M; i++) {
        int32_t acc = 0;

        for (int32_t j = 0; j < packed_K; j++) {
            uint8_t packed = W->data[i * packed_K + j];
            int8_t w[4];
            i2s_unpack4(packed, w);

            int32_t base = j * 4;
            for (int32_t k = 0; k < 4 && (base + k) < K; k++) {
                acc += (int32_t)w[k] * (int32_t)x->data[base + k];
            }
        }

        y->data[i] = (float)acc * scale;
    }
}

/* --- WASM SIMD implementation --- */

#ifdef __wasm_simd128__

void i2s_gemv_simd(const i2s_weight_t *W,
                   const activation_t *x,
                   output_t *y) {
    int32_t M = W->M;
    int32_t K = W->K;
    int32_t packed_K = (K + 3) / 4;
    float scale = W->scale * x->scale;

    /* Masks for 2-bit extraction */
    v128_t mask_2bit = simd_splat_i8(0x03);
    v128_t one = simd_splat_i8(1);

    for (int32_t i = 0; i < M; i++) {
        v128_t acc_lo = simd_zero();
        v128_t acc_hi = simd_zero();

        /*
         * Process 16 weights (4 packed bytes) per SIMD iteration.
         * Each byte has 4 weights, so 4 bytes = 16 weights.
         *
         * Unpack: shift/mask to get 2-bit values, subtract 1 for ternary.
         * Multiply: widen to int16, multiply with int16 activations.
         * Accumulate: add to int32 accumulator.
         */
        int32_t j;
        for (j = 0; j + 16 <= K; j += 16) {
            /* Load 4 packed bytes (16 weights) */
            /* We need to unpack 4 bytes into 16 int8 ternary values */
            const uint8_t *wp = &W->data[i * packed_K + j / 4];

            /* Unpack all 16 weights from 4 bytes */
            int8_t unpacked[16];
            for (int k = 0; k < 4; k++) {
                i2s_unpack4(wp[k], &unpacked[k * 4]);
            }
            v128_t weights_v = simd_load(unpacked);

            /* Load 16 int8 activations */
            v128_t act_v = simd_load(&x->data[j]);

            /* Widen both to int16 and multiply */
            /* Low 8 elements */
            v128_t w_lo = simd_extend_low_i8_to_i16(weights_v);
            v128_t a_lo = simd_extend_low_i8_to_i16(act_v);
            v128_t prod_lo = wasm_i16x8_mul(w_lo, a_lo);

            /* High 8 elements */
            v128_t w_hi = simd_extend_high_i8_to_i16(weights_v);
            v128_t a_hi = simd_extend_high_i8_to_i16(act_v);
            v128_t prod_hi = wasm_i16x8_mul(w_hi, a_hi);

            /* Widen products to int32 and accumulate */
            acc_lo = simd_add_i32(acc_lo,
                wasm_i32x4_extend_low_i16x8(prod_lo));
            acc_lo = simd_add_i32(acc_lo,
                wasm_i32x4_extend_high_i16x8(prod_lo));
            acc_hi = simd_add_i32(acc_hi,
                wasm_i32x4_extend_low_i16x8(prod_hi));
            acc_hi = simd_add_i32(acc_hi,
                wasm_i32x4_extend_high_i16x8(prod_hi));
        }

        /* Reduce SIMD accumulators to scalar */
        int32_t acc = simd_reduce_add_i32(acc_lo)
                    + simd_reduce_add_i32(acc_hi);

        /* Handle remaining elements (< 16) */
        for (; j < K; j++) {
            int32_t idx = j / 4;
            int32_t bit = j % 4;
            int8_t w = (int8_t)(((W->data[i * packed_K + idx]) >> (bit * 2)) & 0x03) - 1;
            acc += (int32_t)w * (int32_t)x->data[j];
        }

        y->data[i] = (float)acc * scale;
    }
}

#else

/* Non-WASM fallback: just use scalar */
void i2s_gemv_simd(const i2s_weight_t *W,
                   const activation_t *x,
                   output_t *y) {
    i2s_gemv_scalar(W, x, y);
}

#endif /* __wasm_simd128__ */
