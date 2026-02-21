#ifndef BITNET_SIMD_UTILS_H
#define BITNET_SIMD_UTILS_H

/*
 * WASM SIMD utility wrappers for BitNet kernels.
 *
 * Key mapping from x86 AVX2 to WASM SIMD:
 *   _mm_shuffle_epi8 (PSHUFB)  ->  wasm_i8x16_swizzle
 *   _mm_add_epi16              ->  wasm_i16x8_add
 *   _mm_set1_epi8              ->  wasm_i8x16_splat
 *   _mm_loadu_si128            ->  wasm_v128_load
 *   _mm_storeu_si128           ->  wasm_v128_store
 *
 * WASM SIMD is 128-bit (vs 256-bit AVX2), so each AVX2 op
 * becomes two WASM ops for the same throughput.
 */

#ifdef __wasm_simd128__
#include <wasm_simd128.h>

/* --- Load / Store --- */

static inline v128_t simd_load(const void *ptr) {
    return wasm_v128_load(ptr);
}

static inline void simd_store(void *ptr, v128_t v) {
    wasm_v128_store(ptr, v);
}

/* --- Ternary Lookup (core TL1 operation) --- */

/*
 * This is THE critical function for BitNet TL1 performance.
 *
 * wasm_i8x16_swizzle does exactly what PSHUFB does:
 *   For each byte in 'indices', use its lower 4 bits as
 *   an index into 'table' (16 bytes). Out-of-range -> 0.
 *
 * This maps directly to PSHUFB on x86 and VTBL on ARM,
 * so it's a single native instruction on both platforms.
 */
static inline v128_t simd_lut_lookup(v128_t table, v128_t indices) {
    return wasm_i8x16_swizzle(table, indices);
}

/* --- Integer Arithmetic --- */

static inline v128_t simd_add_i16(v128_t a, v128_t b) {
    return wasm_i16x8_add(a, b);
}

static inline v128_t simd_add_i32(v128_t a, v128_t b) {
    return wasm_i32x4_add(a, b);
}

static inline v128_t simd_sub_i16(v128_t a, v128_t b) {
    return wasm_i16x8_sub(a, b);
}

/* --- Splat (broadcast scalar to all lanes) --- */

static inline v128_t simd_splat_i8(int8_t val) {
    return wasm_i8x16_splat(val);
}

static inline v128_t simd_splat_i16(int16_t val) {
    return wasm_i16x8_splat(val);
}

static inline v128_t simd_zero(void) {
    return wasm_i32x4_const(0, 0, 0, 0);
}

/* --- Bitwise --- */

static inline v128_t simd_and(v128_t a, v128_t b) {
    return wasm_v128_and(a, b);
}

static inline v128_t simd_or(v128_t a, v128_t b) {
    return wasm_v128_or(a, b);
}

static inline v128_t simd_shr_i16(v128_t a, int32_t bits) {
    return wasm_i16x8_shr(a, bits);
}

static inline v128_t simd_shl_i16(v128_t a, int32_t bits) {
    return wasm_i16x8_shl(a, bits);
}

/* --- Widen (int8 -> int16) --- */

/* Extend lower 8 bytes from int8 to int16 (sign-extend) */
static inline v128_t simd_extend_low_i8_to_i16(v128_t a) {
    return wasm_i16x8_extend_low_i8x16(a);
}

/* Extend upper 8 bytes from int8 to int16 (sign-extend) */
static inline v128_t simd_extend_high_i8_to_i16(v128_t a) {
    return wasm_i16x8_extend_high_i8x16(a);
}

/* --- Horizontal reduction --- */

/* Sum all 8 int16 lanes into a single int32 */
static inline int32_t simd_reduce_add_i16(v128_t a) {
    /* Widen to int32: low 4 lanes + high 4 lanes */
    v128_t lo = wasm_i32x4_extend_low_i16x8(a);
    v128_t hi = wasm_i32x4_extend_high_i16x8(a);
    v128_t sum = wasm_i32x4_add(lo, hi);

    /* Horizontal sum of 4 int32 lanes */
    /* Shuffle high pair to low and add */
    v128_t hi2 = wasm_i64x2_shuffle(sum, sum, 1, 0);
    sum = wasm_i32x4_add(sum, hi2);
    /* Final pair */
    v128_t hi3 = wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2);
    sum = wasm_i32x4_add(sum, hi3);

    return wasm_i32x4_extract_lane(sum, 0);
}

/* Sum all 4 int32 lanes */
static inline int32_t simd_reduce_add_i32(v128_t a) {
    v128_t hi = wasm_i64x2_shuffle(a, a, 1, 0);
    v128_t sum = wasm_i32x4_add(a, hi);
    v128_t hi2 = wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2);
    sum = wasm_i32x4_add(sum, hi2);
    return wasm_i32x4_extract_lane(sum, 0);
}

/* --- Nibble extraction (for TL1 4-bit indices) --- */

static inline v128_t simd_mask_low_nibble(void) {
    return wasm_i8x16_splat(0x0F);
}

/* Extract low nibbles (bits 0-3) from packed byte */
static inline v128_t simd_extract_low_nibbles(v128_t packed) {
    return wasm_v128_and(packed, simd_mask_low_nibble());
}

/* Extract high nibbles (bits 4-7) from packed byte, shifted to low position */
static inline v128_t simd_extract_high_nibbles(v128_t packed) {
    return wasm_u8x16_shr(packed, 4);
}

#else
/* Scalar fallback declarations for non-WASM builds (testing) */
#warning "Building without WASM SIMD - using scalar fallbacks"
#endif /* __wasm_simd128__ */

#endif /* BITNET_SIMD_UTILS_H */
