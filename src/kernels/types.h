#ifndef BITNET_TYPES_H
#define BITNET_TYPES_H

#include <stdint.h>
#include <stddef.h>

/*
 * BitNet b1.58 WASM SIMD Kernels
 *
 * Ternary weight values: {-1, 0, 1}
 * Activations: int8 per-tensor quantization
 * Accumulation: int16/int32
 *
 * Weight packing formats:
 *   I2_S: 2 bits per weight, 4 weights per byte
 *   TL1:  4-bit index per 2 weights (9 valid combos)
 *   TL2:  1-bit sign + 4-bit index per 3 weights
 */

/* Ternary weight encoding for I2_S (2-bit packed) */
/* 00 = -1, 01 = 0, 10 = 1 (11 unused) */
#define TERNARY_NEG  0x00
#define TERNARY_ZERO 0x01
#define TERNARY_POS  0x02

/* TL1 lookup table: 9 valid entries for pairs of ternary weights */
/* Index = (w0 + 1) * 3 + (w1 + 1), maps to 0..8 */
#define TL1_NUM_ENTRIES 9
#define TL1_WEIGHTS_PER_INDEX 2

/* TL2: 27 combos for 3 weights, compressed to 4-bit + sign */
#define TL2_NUM_ENTRIES 16
#define TL2_WEIGHTS_PER_INDEX 3

/* GEMV parameters */
typedef struct {
    int32_t M;           /* output dimension (rows) */
    int32_t K;           /* input dimension (cols)  */
    float   w_scale;     /* weight quantization scale */
    float   a_scale;     /* activation quantization scale */
} gemv_params_t;

/* Packed weight matrix for I2_S kernel */
typedef struct {
    uint8_t *data;       /* 2-bit packed weights, 4 per byte */
    int32_t  M;          /* rows */
    int32_t  K;          /* cols */
    float    scale;      /* per-tensor scale factor */
} i2s_weight_t;

/* Packed weight matrix for TL1 kernel */
typedef struct {
    uint8_t *indices;    /* 4-bit indices, 2 per byte (nibble packed) */
    int32_t  M;          /* rows */
    int32_t  K;          /* cols (must be even) */
    float    scale;
} tl1_weight_t;

/* Activation vector (int8 quantized) */
typedef struct {
    int8_t  *data;
    int32_t  len;
    float    scale;      /* scale = max(|x|) / 127 */
} activation_t;

/* Output buffer */
typedef struct {
    float   *data;
    int32_t  len;
} output_t;

#endif /* BITNET_TYPES_H */
