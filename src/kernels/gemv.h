#ifndef BITNET_GEMV_H
#define BITNET_GEMV_H

#include "types.h"

/*
 * GEMV wrapper: unified interface for all kernel variants.
 *
 * Provides a single entry point that dispatches to the selected
 * kernel implementation (I2_S or TL1, scalar or SIMD).
 */

typedef enum {
    KERNEL_I2S_SCALAR,
    KERNEL_I2S_SIMD,
    KERNEL_TL1_SCALAR,
    KERNEL_TL1_SIMD,
    KERNEL_TL2_SCALAR,
    KERNEL_TL2_SIMD,
    KERNEL_COUNT
} kernel_type_t;

const char *kernel_name(kernel_type_t type);

/*
 * Quantize float activations to int8.
 *
 * Uses per-tensor absmax quantization:
 *   scale = max(|x|) / 127
 *   quantized = round(x / scale)
 */
void quantize_activations(const float *input, int32_t len,
                          int8_t *output, float *scale);

/*
 * Run GEMV with the specified kernel.
 *
 * For TL1 kernels, the LUT is built internally.
 * Caller must provide appropriately packed weights.
 *
 * Returns elapsed time in milliseconds.
 */
double gemv_run(kernel_type_t kernel,
                const void *weights,       /* i2s_weight_t* or tl1_weight_t* */
                const float *activations,  /* raw float input */
                float *output,             /* float output */
                int32_t M, int32_t K);

#endif /* BITNET_GEMV_H */
