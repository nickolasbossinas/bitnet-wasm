#include "gemv.h"
#include "i2s.h"
#include "tl1.h"
#include <stdlib.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
/* Use emscripten_get_now() for high-res timing */
static double get_time_ms(void) {
    return emscripten_get_now();
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

const char *kernel_name(kernel_type_t type) {
    switch (type) {
        case KERNEL_I2S_SCALAR: return "I2_S (scalar)";
        case KERNEL_I2S_SIMD:   return "I2_S (WASM SIMD)";
        case KERNEL_TL1_SCALAR: return "TL1 (scalar)";
        case KERNEL_TL1_SIMD:   return "TL1 (WASM SIMD)";
        default:                return "unknown";
    }
}

void quantize_activations(const float *input, int32_t len,
                          int8_t *output, float *scale) {
    /* Find absmax */
    float absmax = 0.0f;
    for (int32_t i = 0; i < len; i++) {
        float a = fabsf(input[i]);
        if (a > absmax) absmax = a;
    }

    *scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    float inv_scale = 1.0f / *scale;

    for (int32_t i = 0; i < len; i++) {
        float v = input[i] * inv_scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        output[i] = (int8_t)roundf(v);
    }
}

double gemv_run(kernel_type_t kernel,
                const void *weights,
                const float *activations,
                float *output,
                int32_t M, int32_t K) {
    /* Quantize activations */
    int8_t *quant_act = (int8_t *)malloc(K * sizeof(int8_t));
    float a_scale;
    quantize_activations(activations, K, quant_act, &a_scale);

    activation_t act = { .data = quant_act, .len = K, .scale = a_scale };
    output_t out = { .data = output, .len = M };

    double t0, t1;

    switch (kernel) {
        case KERNEL_I2S_SCALAR:
            t0 = get_time_ms();
            i2s_gemv_scalar((const i2s_weight_t *)weights, &act, &out);
            t1 = get_time_ms();
            break;

        case KERNEL_I2S_SIMD:
            t0 = get_time_ms();
            i2s_gemv_simd((const i2s_weight_t *)weights, &act, &out);
            t1 = get_time_ms();
            break;

        case KERNEL_TL1_SCALAR:
        case KERNEL_TL1_SIMD: {
            /* Build LUT (included in timing — it's part of the kernel) */
            int32_t lut_size = (K / 2) * 16;
            int8_t *lut = (int8_t *)calloc(lut_size, sizeof(int8_t));
            tl1_build_lut(lut, quant_act, K);

            if (kernel == KERNEL_TL1_SCALAR) {
                t0 = get_time_ms();
                tl1_gemv_scalar((const tl1_weight_t *)weights, lut, &act, &out);
                t1 = get_time_ms();
            } else {
                t0 = get_time_ms();
                tl1_gemv_simd((const tl1_weight_t *)weights, lut, &act, &out);
                t1 = get_time_ms();
            }

            free(lut);
            break;
        }

        default:
            t0 = t1 = 0.0;
            break;
    }

    free(quant_act);
    return t1 - t0;
}
