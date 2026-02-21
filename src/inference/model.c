#include "model.h"
#include "thread_pool.h"
#include "../kernels/gemv.h"
#include "../kernels/tl1.h"
#include "../inference/weight_loader.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* F32 -> F16 conversion (portable, for KV cache storage) */
static inline uint16_t f32_to_f16_scalar(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;

    if (exp <= 0) return sign;         /* underflow to ±0 */
    if (exp >= 31) return sign | 0x7C00; /* overflow to ±inf */
    return sign | ((uint16_t)exp << 10) | (uint16_t)mant;
}

#ifdef __wasm_simd128__
#include <wasm_simd128.h>

/*
 * SIMD F16 -> F32 conversion (4 values at a time).
 *
 * Uses the "magic number" trick: for normal F16 values,
 *   f32_bits = ((h & 0x7FFF) << 13) + 0x38000000
 * handles exponent bias adjustment (F16 bias=15 -> F32 bias=127, diff=112).
 * Zero and sign bits handled separately.
 */
static inline v128_t f16x4_to_f32x4(const uint16_t *p) {
    /* Load 4 u16, zero-extend to u32x4 */
    v128_t h = wasm_u32x4_load16x4(p);

    /* Extract and position sign bit */
    v128_t sign = wasm_i32x4_shl(
        wasm_v128_and(h, wasm_i32x4_const(0x8000, 0x8000, 0x8000, 0x8000)), 16);

    /* Magnitude: shift mantissa+exponent, add exponent bias */
    v128_t mag = wasm_v128_and(h, wasm_i32x4_const(0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF));
    mag = wasm_i32x4_add(wasm_i32x4_shl(mag, 13),
                          wasm_i32x4_const(0x38000000, 0x38000000,
                                           0x38000000, 0x38000000));

    /* Handle zeros: if h & 0x7FFF == 0, result should be ±0 (not 2^-112) */
    v128_t is_zero = wasm_i32x4_eq(
        wasm_v128_and(h, wasm_i32x4_const(0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF)),
        wasm_i32x4_const(0, 0, 0, 0));
    mag = wasm_v128_andnot(mag, is_zero);  /* zero out mag where input was zero */

    return wasm_v128_or(sign, mag);
}
#endif

/* --- Config --- */

void model_config_from_gguf(model_config_t *config, const gguf_context_t *gguf) {
    config->n_layers          = gguf->n_layers;
    config->hidden_size       = gguf->hidden_size;
    config->intermediate_size = gguf->intermediate_size;
    config->n_heads           = gguf->n_heads;
    config->n_kv_heads        = gguf->n_kv_heads;
    config->head_dim          = gguf->head_dim;
    config->kv_dim            = gguf->n_kv_heads * gguf->head_dim;
    config->vocab_size        = gguf->vocab_size;
    config->max_seq_len       = gguf->max_seq_len;
    config->rope_theta        = gguf->rope_theta;
    config->rms_norm_eps      = gguf->rms_norm_eps;
}

/* --- Memory allocation --- */

int model_alloc(model_t *model) {
    model_config_t *c = &model->config;
    int32_t dim = c->hidden_size;
    int32_t kv_dim = c->kv_dim;
    int32_t inter = c->intermediate_size;
    int32_t vocab = c->vocab_size;
    int32_t seq = c->max_seq_len;
    int32_t nl = c->n_layers;

    /* Scratch buffers */
    model->x      = (float *)calloc(dim, sizeof(float));
    model->xb     = (float *)calloc(dim, sizeof(float));
    model->xb2    = (float *)calloc(dim, sizeof(float));
    model->q      = (float *)calloc(dim, sizeof(float));
    model->k      = (float *)calloc(kv_dim, sizeof(float));
    model->v      = (float *)calloc(kv_dim, sizeof(float));
    model->att    = (float *)calloc(c->n_heads * seq, sizeof(float));
    model->hb     = (float *)calloc(inter, sizeof(float));
    model->hb2    = (float *)calloc(inter, sizeof(float));
    model->logits = (float *)calloc(vocab, sizeof(float));

    if (!model->x || !model->xb || !model->xb2 || !model->q ||
        !model->k || !model->v || !model->att || !model->hb ||
        !model->hb2 || !model->logits) {
        fprintf(stderr, "model: failed to allocate scratch buffers\n");
        return -1;
    }

    /* KV cache (F16 — halves memory: 300 MB vs 600 MB for 30 layers) */
    size_t kv_size = (size_t)nl * seq * kv_dim * sizeof(uint16_t);
    model->key_cache   = (uint16_t *)calloc(1, kv_size);
    model->value_cache = (uint16_t *)calloc(1, kv_size);
    if (!model->key_cache || !model->value_cache) {
        fprintf(stderr, "model: failed to allocate KV cache (%.1f MB)\n",
                2.0f * kv_size / (1024.0f * 1024.0f));
        return -1;
    }

    /* Layer weights array (weights themselves are loaded separately) */
    model->layers = (layer_weights_t *)calloc(nl, sizeof(layer_weights_t));
    if (!model->layers) return -1;

    /* Precompute RoPE sin/cos tables — eliminates trig from forward pass */
    {
        int32_t half_dim = c->head_dim / 2;
        model->rope_cos = (float *)malloc(seq * half_dim * sizeof(float));
        model->rope_sin = (float *)malloc(seq * half_dim * sizeof(float));
        if (!model->rope_cos || !model->rope_sin) {
            fprintf(stderr, "model: failed to allocate RoPE tables\n");
            return -1;
        }
        for (int32_t pos = 0; pos < seq; pos++) {
            float *cos_row = &model->rope_cos[pos * half_dim];
            float *sin_row = &model->rope_sin[pos * half_dim];
            for (int32_t i = 0; i < half_dim; i++) {
                float freq = 1.0f / powf(c->rope_theta,
                                          (float)(i * 2) / (float)c->head_dim);
                float angle = (float)pos * freq;
                cos_row[i] = cosf(angle);
                sin_row[i] = sinf(angle);
            }
        }
    }

    /* GEMV scratch buffers — eliminates all malloc/free from forward pass */
    int32_t max_K = (dim > inter) ? dim : inter;
    int32_t max_pairs = max_K / 2;
    model->scratch.max_K = max_K;
    model->scratch.prepared_K = 0;
    model->scratch.quant_buf  = (int8_t  *)malloc(max_K * sizeof(int8_t));
    model->scratch.lut_buf    = (int16_t *)calloc(max_pairs * 16, sizeof(int16_t));
    model->scratch.lut_lo_buf = (uint8_t *)malloc(max_pairs * 16);
    model->scratch.lut_hi_buf = (uint8_t *)malloc(max_pairs * 16);
    if (!model->scratch.quant_buf || !model->scratch.lut_buf ||
        !model->scratch.lut_lo_buf || !model->scratch.lut_hi_buf) {
        fprintf(stderr, "model: failed to allocate GEMV scratch buffers\n");
        return -1;
    }

    return 0;
}

void model_free(model_t *model) {
    free(model->x);
    free(model->xb);
    free(model->xb2);
    free(model->q);
    free(model->k);
    free(model->v);
    free(model->att);
    free(model->hb);
    free(model->hb2);
    free(model->logits);
    free(model->key_cache);
    free(model->value_cache);

    if (model->layers) {
        for (int32_t l = 0; l < model->config.n_layers; l++) {
            layer_weights_t *lw = &model->layers[l];
            free(lw->attn_norm);
            free(lw->ffn_norm);
            free(lw->attn_sub_norm);
            free(lw->ffn_sub_norm);
            /* TL1 weight data is freed by whoever loaded it */
        }
        free(model->layers);
    }

    free(model->token_embedding);
    free(model->output_norm);
    free(model->emb_quantized);
    free(model->emb_row_scales);
    free(model->rope_cos);
    free(model->rope_sin);

    /* GEMV scratch buffers */
    free(model->scratch.quant_buf);
    free(model->scratch.lut_buf);
    free(model->scratch.lut_lo_buf);
    free(model->scratch.lut_hi_buf);

    memset(model, 0, sizeof(*model));
}

/* --- Math primitives --- */

void rms_norm(float *out, const float *x, const float *weight,
              int32_t dim, float eps) {
    float ss = 0.0f;
    for (int32_t i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / dim + eps);
    for (int32_t i = 0; i < dim; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

void softmax(float *x, int32_t size) {
    float max_val = x[0];
    for (int32_t i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int32_t i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int32_t i = 0; i < size; i++) {
        x[i] *= inv;
    }
}

void matmul_f32_range(float *out, const float *x, const float *W,
                      int32_t K, int32_t row_start, int32_t row_end) {
    /* out[i] = sum_j(W[i*K + j] * x[j]) for rows [row_start, row_end) */
#ifdef __wasm_simd128__
    for (int32_t i = row_start; i < row_end; i++) {
        const float *row = &W[i * K];
        v128_t sum0 = wasm_f32x4_const(0, 0, 0, 0);
        v128_t sum1 = wasm_f32x4_const(0, 0, 0, 0);
        int32_t j;
        for (j = 0; j + 8 <= K; j += 8) {
            sum0 = wasm_f32x4_add(sum0, wasm_f32x4_mul(
                wasm_v128_load(&row[j]),     wasm_v128_load(&x[j])));
            sum1 = wasm_f32x4_add(sum1, wasm_f32x4_mul(
                wasm_v128_load(&row[j + 4]), wasm_v128_load(&x[j + 4])));
        }
        sum0 = wasm_f32x4_add(sum0, sum1);
        v128_t hi = wasm_i64x2_shuffle(sum0, sum0, 1, 0);
        sum0 = wasm_f32x4_add(sum0, hi);
        v128_t hi2 = wasm_i32x4_shuffle(sum0, sum0, 1, 0, 3, 2);
        sum0 = wasm_f32x4_add(sum0, hi2);
        float sum = wasm_f32x4_extract_lane(sum0, 0);
        for (; j < K; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
#else
    for (int32_t i = row_start; i < row_end; i++) {
        float sum = 0.0f;
        const float *row = &W[i * K];
        for (int32_t j = 0; j < K; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
#endif
}

void matmul_f32(float *out, const float *x, const float *W,
                int32_t M, int32_t K) {
    matmul_f32_range(out, x, W, K, 0, M);
}

/*
 * F16 x F32 matmul: out[i] = W_f16[i,:] · x for rows [row_start, row_end).
 * W is stored as uint16_t (F16). Converts to F32 on-the-fly during multiply.
 * Halves memory bandwidth vs F32 matmul — critical for logits (128K x 2560).
 */
void matmul_f16f32_range(float *out, const float *x, const uint16_t *W,
                          int32_t K, int32_t row_start, int32_t row_end) {
#ifdef __wasm_simd128__
    for (int32_t i = row_start; i < row_end; i++) {
        const uint16_t *row = &W[i * K];
        v128_t sum0 = wasm_f32x4_const(0, 0, 0, 0);
        v128_t sum1 = wasm_f32x4_const(0, 0, 0, 0);
        int32_t j;
        for (j = 0; j + 8 <= K; j += 8) {
            /* Convert 8 F16 weights to 2 x f32x4, multiply by x, accumulate */
            v128_t w0 = f16x4_to_f32x4(&row[j]);
            v128_t w1 = f16x4_to_f32x4(&row[j + 4]);
            sum0 = wasm_f32x4_add(sum0, wasm_f32x4_mul(w0, wasm_v128_load(&x[j])));
            sum1 = wasm_f32x4_add(sum1, wasm_f32x4_mul(w1, wasm_v128_load(&x[j + 4])));
        }
        sum0 = wasm_f32x4_add(sum0, sum1);
        /* Horizontal sum */
        v128_t hi = wasm_i64x2_shuffle(sum0, sum0, 1, 0);
        sum0 = wasm_f32x4_add(sum0, hi);
        v128_t hi2 = wasm_i32x4_shuffle(sum0, sum0, 1, 0, 3, 2);
        sum0 = wasm_f32x4_add(sum0, hi2);
        float sum = wasm_f32x4_extract_lane(sum0, 0);
        /* Scalar tail */
        for (; j < K; j++) {
            sum += f16_to_f32(row[j]) * x[j];
        }
        out[i] = sum;
    }
#else
    for (int32_t i = row_start; i < row_end; i++) {
        float sum = 0.0f;
        const uint16_t *row = &W[i * K];
        for (int32_t j = 0; j < K; j++) {
            sum += f16_to_f32(row[j]) * x[j];
        }
        out[i] = sum;
    }
#endif
}

void matmul_f16f32(float *out, const float *x, const uint16_t *W,
                    int32_t M, int32_t K) {
    matmul_f16f32_range(out, x, W, K, 0, M);
}

/*
 * INT8 x INT8 matmul: out[i] = dot(W[i,:], x) * row_scales[i] * x_scale
 * Both W and x are symmetric int8 quantized.
 * Uses i32x4.dot_i16x8_s for efficient pairwise multiply-add.
 * 2x less memory bandwidth than F16 path — critical for logits (128K rows).
 */
void matmul_i8_range(float *out, const int8_t *x_quant, float x_scale,
                     const int8_t *W, const float *row_scales,
                     int32_t K, int32_t row_start, int32_t row_end) {
#ifdef __wasm_simd128__
    for (int32_t i = row_start; i < row_end; i++) {
        const int8_t *row = &W[(int64_t)i * K];
        v128_t acc0 = wasm_i32x4_const(0, 0, 0, 0);
        v128_t acc1 = wasm_i32x4_const(0, 0, 0, 0);
        int32_t j;
        for (j = 0; j + 32 <= K; j += 32) {
            /* First 16 bytes */
            v128_t w0 = wasm_v128_load(&row[j]);
            v128_t x0 = wasm_v128_load(&x_quant[j]);
            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_low_i8x16(w0), wasm_i16x8_extend_low_i8x16(x0)));
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_high_i8x16(w0), wasm_i16x8_extend_high_i8x16(x0)));
            /* Second 16 bytes */
            v128_t w1 = wasm_v128_load(&row[j + 16]);
            v128_t x1 = wasm_v128_load(&x_quant[j + 16]);
            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_low_i8x16(w1), wasm_i16x8_extend_low_i8x16(x1)));
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_high_i8x16(w1), wasm_i16x8_extend_high_i8x16(x1)));
        }
        /* Handle remaining 16-byte chunk */
        for (; j + 16 <= K; j += 16) {
            v128_t w0 = wasm_v128_load(&row[j]);
            v128_t x0 = wasm_v128_load(&x_quant[j]);
            acc0 = wasm_i32x4_add(acc0, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_low_i8x16(w0), wasm_i16x8_extend_low_i8x16(x0)));
            acc1 = wasm_i32x4_add(acc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_high_i8x16(w0), wasm_i16x8_extend_high_i8x16(x0)));
        }
        acc0 = wasm_i32x4_add(acc0, acc1);
        /* Horizontal sum */
        v128_t hi = wasm_i64x2_shuffle(acc0, acc0, 1, 0);
        acc0 = wasm_i32x4_add(acc0, hi);
        v128_t hi2 = wasm_i32x4_shuffle(acc0, acc0, 1, 0, 3, 2);
        acc0 = wasm_i32x4_add(acc0, hi2);
        int32_t sum = wasm_i32x4_extract_lane(acc0, 0);
        /* Scalar tail */
        for (; j < K; j++) {
            sum += (int32_t)row[j] * (int32_t)x_quant[j];
        }
        out[i] = (float)sum * row_scales[i] * x_scale;
    }
#else
    for (int32_t i = row_start; i < row_end; i++) {
        const int8_t *row = &W[(int64_t)i * K];
        int32_t sum = 0;
        for (int32_t j = 0; j < K; j++) {
            sum += (int32_t)row[j] * (int32_t)x_quant[j];
        }
        out[i] = (float)sum * row_scales[i] * x_scale;
    }
#endif
}

void matmul_i8(float *out, const int8_t *x_quant, float x_scale,
               const int8_t *W, const float *row_scales,
               int32_t M, int32_t K) {
    matmul_i8_range(out, x_quant, x_scale, W, row_scales, K, 0, M);
}

void rope_apply(float *q, float *k, int32_t n_heads, int32_t n_kv_heads,
                int32_t head_dim, int32_t pos, float theta) {
    int32_t half_dim = head_dim / 2;
    float cos_cache[256], sin_cache[256];

    for (int32_t i = 0; i < half_dim; i++) {
        float freq = 1.0f / powf(theta, (float)(i * 2) / (float)head_dim);
        float angle = (float)pos * freq;
        cos_cache[i] = cosf(angle);
        sin_cache[i] = sinf(angle);
    }

    for (int32_t h = 0; h < n_heads; h++) {
        float *qh = &q[h * head_dim];
        for (int32_t i = 0; i < half_dim; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_cache[i] - q1 * sin_cache[i];
            qh[i * 2 + 1] = q0 * sin_cache[i] + q1 * cos_cache[i];
        }
    }

    for (int32_t h = 0; h < n_kv_heads; h++) {
        float *kh = &k[h * head_dim];
        for (int32_t i = 0; i < half_dim; i++) {
            float k0 = kh[i * 2];
            float k1 = kh[i * 2 + 1];
            kh[i * 2]     = k0 * cos_cache[i] - k1 * sin_cache[i];
            kh[i * 2 + 1] = k0 * sin_cache[i] + k1 * cos_cache[i];
        }
    }
}

/* Fast RoPE using precomputed cos/sin tables (zero trig calls) */
static void rope_apply_cached(float *q, float *k, int32_t n_heads, int32_t n_kv_heads,
                               int32_t head_dim, const float *cos_tab, const float *sin_tab) {
    int32_t half_dim = head_dim / 2;

    for (int32_t h = 0; h < n_heads; h++) {
        float *qh = &q[h * head_dim];
        for (int32_t i = 0; i < half_dim; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_tab[i] - q1 * sin_tab[i];
            qh[i * 2 + 1] = q0 * sin_tab[i] + q1 * cos_tab[i];
        }
    }

    for (int32_t h = 0; h < n_kv_heads; h++) {
        float *kh = &k[h * head_dim];
        for (int32_t i = 0; i < half_dim; i++) {
            float k0 = kh[i * 2];
            float k1 = kh[i * 2 + 1];
            kh[i * 2]     = k0 * cos_tab[i] - k1 * sin_tab[i];
            kh[i * 2 + 1] = k0 * sin_tab[i] + k1 * cos_tab[i];
        }
    }
}

void bitlinear(float *out, const float *x, const tl1_weight_t *W) {
    /*
     * BitLinear: ternary GEMV with activation quantization.
     * Legacy interface — allocates/frees per call. Used by benchmark.
     */
    gemv_run(KERNEL_TL1_SIMD, W, x, out, W->M, W->K);
}

void bitlinear_prepare(model_t *model, const float *x, int32_t K) {
    /*
     * Phase 1: quantize activations + build LUT + pre-split for SIMD.
     * Results stored in model->scratch for subsequent bitlinear_run calls.
     * Call once per unique input vector.
     */
    gemv_scratch_t *s = &model->scratch;

    /* Quantize float activations to int8 */
    quantize_activations(x, K, s->quant_buf, &s->act_scale);

    /* Build TL1 lookup table */
    int32_t num_pairs = K / 2;
    tl1_build_lut(s->lut_buf, s->quant_buf, K);

    /* Pre-split LUT into lo/hi byte tables for SIMD */
    tl1_presplit_lut(s->lut_buf, s->lut_lo_buf, s->lut_hi_buf, num_pairs);

    s->prepared_K = K;
}

void bitlinear_run(float *out, const tl1_weight_t *W, model_t *model) {
    /*
     * Phase 2: execute GEMV using pre-built LUT from bitlinear_prepare.
     * Zero allocation. Can be called multiple times with different weight
     * matrices that share the same input vector.
     * Uses thread pool for parallel row processing when available.
     */
    gemv_scratch_t *s = &model->scratch;
    float scale = W->scale * s->act_scale;

    if (model->pool && model->pool->n_threads > 0) {
        thread_pool_gemv(model->pool, W, s->lut_buf, s->lut_lo_buf,
                          s->lut_hi_buf, scale, out);
    } else {
        tl1_gemv_simd_fast(W, s->lut_buf, s->lut_lo_buf, s->lut_hi_buf,
                            scale, out);
    }
}

/*
 * Batched bitlinear: dispatch multiple GEMVs sharing the same LUT in one
 * thread pool barrier pair. Reduces Q+K+V from 3 dispatches to 1.
 * Each op carries its own scale (W->scale * act_scale).
 */
static void bitlinear_run_batch(model_t *model,
                                 gemv_batch_op_t *ops, int32_t n_ops) {
    gemv_scratch_t *s = &model->scratch;

    /* Fill per-op combined scales */
    for (int32_t i = 0; i < n_ops; i++) {
        ops[i].scale = ops[i].W->scale * s->act_scale;
    }

    if (model->pool && model->pool->n_threads > 0) {
        thread_pool_gemv_batch(model->pool, ops, n_ops,
                                s->lut_buf, s->lut_lo_buf, s->lut_hi_buf);
    } else {
        for (int32_t i = 0; i < n_ops; i++) {
            tl1_gemv_simd_fast(ops[i].W, s->lut_buf, s->lut_lo_buf, s->lut_hi_buf,
                                ops[i].scale, ops[i].out);
        }
    }
}

/* --- Forward pass --- */

/*
 * KV cache layout:
 *   key_cache[layer * max_seq * kv_dim + pos * kv_dim + head * head_dim + d]
 *   value_cache[same layout]
 */

float *forward(model_t *model, int32_t token, int32_t pos) {
    model_config_t *c = &model->config;
    int32_t dim = c->hidden_size;
    int32_t kv_dim = c->kv_dim;
    int32_t inter = c->intermediate_size;
    int32_t n_heads = c->n_heads;
    int32_t n_kv_heads = c->n_kv_heads;
    int32_t head_dim = c->head_dim;
    int32_t kv_group = n_heads / n_kv_heads;  /* 4 for GQA */
    int32_t seq = c->max_seq_len;

    float *x = model->x;

    /* Token embedding lookup (INT8 dequantize or F16 fallback) */
    if (model->emb_quantized) {
        const int8_t *emb_row = &model->emb_quantized[(int64_t)token * dim];
        float scale = model->emb_row_scales[token];
        for (int32_t i = 0; i < dim; i++) {
            x[i] = (float)emb_row[i] * scale;
        }
    } else {
        const uint16_t *emb_row = &model->token_embedding[(int64_t)token * dim];
        for (int32_t i = 0; i < dim; i++) {
            x[i] = f16_to_f32(emb_row[i]);
        }
    }

    /* Process each layer */
    for (int32_t l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &model->layers[l];

        /* ---- Attention block ---- */

        /* Pre-norm */
        rms_norm(model->xb, x, lw->attn_norm, dim, c->rms_norm_eps);

        /* Q, K, V projections — shared input, prepare LUT once, batch dispatch */
        bitlinear_prepare(model, model->xb, dim);
        {
            gemv_batch_op_t qkv_ops[3] = {
                { .W = &lw->attn_q, .out = model->q },
                { .W = &lw->attn_k, .out = model->k },
                { .W = &lw->attn_v, .out = model->v },
            };
            bitlinear_run_batch(model, qkv_ops, 3);
        }

        /* Apply RoPE (precomputed tables or fallback) */
        if (model->rope_cos) {
            int32_t half_dim = head_dim / 2;
            rope_apply_cached(model->q, model->k, n_heads, n_kv_heads,
                              head_dim,
                              &model->rope_cos[pos * half_dim],
                              &model->rope_sin[pos * half_dim]);
        } else {
            rope_apply(model->q, model->k, n_heads, n_kv_heads,
                       head_dim, pos, c->rope_theta);
        }

        /* Store K, V in cache (F32 → F16 conversion) */
        {
            size_t kv_offset = (size_t)l * seq * kv_dim + (size_t)pos * kv_dim;
            for (int32_t i = 0; i < kv_dim; i++) {
                model->key_cache[kv_offset + i] = f32_to_f16_scalar(model->k[i]);
                model->value_cache[kv_offset + i] = f32_to_f16_scalar(model->v[i]);
            }
        }

        /* Grouped Query Attention */
        for (int32_t h = 0; h < n_heads; h++) {
            float *qh = &model->q[h * head_dim];
            float *att = &model->att[h * seq];
            int32_t kv_h = h / kv_group;  /* which KV head */

            /* Compute attention scores: Q @ K^T / sqrt(head_dim) */
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int32_t t = 0; t <= pos; t++) {
                const uint16_t *kh = &model->key_cache[(size_t)l * seq * kv_dim +
                                                        (size_t)t * kv_dim +
                                                        kv_h * head_dim];
#ifdef __wasm_simd128__
                v128_t s0 = wasm_f32x4_const(0,0,0,0);
                v128_t s1 = wasm_f32x4_const(0,0,0,0);
                int32_t d;
                for (d = 0; d + 8 <= head_dim; d += 8) {
                    s0 = wasm_f32x4_add(s0, wasm_f32x4_mul(
                        wasm_v128_load(&qh[d]), f16x4_to_f32x4(&kh[d])));
                    s1 = wasm_f32x4_add(s1, wasm_f32x4_mul(
                        wasm_v128_load(&qh[d+4]), f16x4_to_f32x4(&kh[d+4])));
                }
                s0 = wasm_f32x4_add(s0, s1);
                v128_t hi = wasm_i64x2_shuffle(s0, s0, 1, 0);
                s0 = wasm_f32x4_add(s0, hi);
                v128_t hi2 = wasm_i32x4_shuffle(s0, s0, 1, 0, 3, 2);
                s0 = wasm_f32x4_add(s0, hi2);
                float score = wasm_f32x4_extract_lane(s0, 0);
                for (; d < head_dim; d++) score += qh[d] * f16_to_f32(kh[d]);
#else
                float score = 0.0f;
                for (int32_t d = 0; d < head_dim; d++) {
                    score += qh[d] * f16_to_f32(kh[d]);
                }
#endif
                att[t] = score * scale;
            }

            /* Softmax over [0..pos] */
            softmax(att, pos + 1);

            /* Weighted sum of values */
            float *out_h = &model->xb[h * head_dim];
            memset(out_h, 0, head_dim * sizeof(float));
            for (int32_t t = 0; t <= pos; t++) {
                const uint16_t *vh = &model->value_cache[(size_t)l * seq * kv_dim +
                                                          (size_t)t * kv_dim +
                                                          kv_h * head_dim];
                float a = att[t];
#ifdef __wasm_simd128__
                v128_t va = wasm_f32x4_splat(a);
                int32_t d;
                for (d = 0; d + 8 <= head_dim; d += 8) {
                    wasm_v128_store(&out_h[d], wasm_f32x4_add(
                        wasm_v128_load(&out_h[d]),
                        wasm_f32x4_mul(va, f16x4_to_f32x4(&vh[d]))));
                    wasm_v128_store(&out_h[d+4], wasm_f32x4_add(
                        wasm_v128_load(&out_h[d+4]),
                        wasm_f32x4_mul(va, f16x4_to_f32x4(&vh[d+4]))));
                }
                for (; d < head_dim; d++) out_h[d] += a * f16_to_f32(vh[d]);
#else
                for (int32_t d = 0; d < head_dim; d++) {
                    out_h[d] += a * f16_to_f32(vh[d]);
                }
#endif
            }
        }

        /* SubLN: RMSNorm before output projection */
        /* attn_sub_norm is [hidden_size], applied over full concatenated output */
        rms_norm(model->xb, model->xb, lw->attn_sub_norm, dim, c->rms_norm_eps);

        /* Output projection + residual (different input after attention) */
        bitlinear_prepare(model, model->xb, dim);
        bitlinear_run(model->xb2, &lw->attn_o, model);
        for (int32_t i = 0; i < dim; i++) {
            x[i] += model->xb2[i];
        }

        /* ---- FFN block ---- */

        /* Pre-norm */
        rms_norm(model->xb, x, lw->ffn_norm, dim, c->rms_norm_eps);

        /* Gate and up projections — shared input, prepare LUT once, batch dispatch */
        bitlinear_prepare(model, model->xb, dim);
        {
            gemv_batch_op_t gate_up_ops[2] = {
                { .W = &lw->ffn_gate, .out = model->hb },
                { .W = &lw->ffn_up,   .out = model->hb2 },
            };
            bitlinear_run_batch(model, gate_up_ops, 2);
        }

        /* Squared ReLU: max(0, gate)^2 * up */
        for (int32_t i = 0; i < inter; i++) {
            float g = model->hb[i];
            g = (g > 0.0f) ? g * g : 0.0f;
            model->hb[i] = g * model->hb2[i];
        }

        /* SubLN: RMSNorm before down projection */
        rms_norm(model->hb2, model->hb, lw->ffn_sub_norm, inter, c->rms_norm_eps);

        /* Down projection + residual (different input, K=intermediate) */
        bitlinear_prepare(model, model->hb2, inter);
        bitlinear_run(model->xb, &lw->ffn_down, model);
        for (int32_t i = 0; i < dim; i++) {
            x[i] += model->xb[i];
        }
    }

    /* Final RMSNorm */
    rms_norm(x, x, model->output_norm, dim, c->rms_norm_eps);

    /* Output logits: x @ embedding^T */
    if (model->emb_quantized) {
        /* INT8 path: quantize hidden state, then int8×int8 matmul (2x less BW) */
        int8_t *x_quant = model->scratch.quant_buf;
        float max_abs = 0.0f;
        for (int32_t i = 0; i < dim; i++) {
            float av = x[i] > 0 ? x[i] : -x[i];
            if (av > max_abs) max_abs = av;
        }
        float logits_x_scale = max_abs / 127.0f;
        if (max_abs > 0.0f) {
            float inv_scale = 127.0f / max_abs;
            for (int32_t i = 0; i < dim; i++) {
                int32_t q = (int32_t)(x[i] * inv_scale + (x[i] > 0 ? 0.5f : -0.5f));
                x_quant[i] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
            }
        } else {
            memset(x_quant, 0, dim);
        }

        if (model->pool && model->pool->n_threads > 0) {
            thread_pool_matmul_i8(model->pool, model->logits, x_quant, logits_x_scale,
                                   model->emb_quantized, model->emb_row_scales,
                                   c->vocab_size, dim);
        } else {
            matmul_i8(model->logits, x_quant, logits_x_scale,
                       model->emb_quantized, model->emb_row_scales,
                       c->vocab_size, dim);
        }
    } else {
        /* F16 fallback */
        if (model->pool && model->pool->n_threads > 0) {
            thread_pool_matmul_f16(model->pool, model->logits, x,
                                    model->token_embedding, c->vocab_size, dim);
        } else {
            matmul_f16f32(model->logits, x, model->token_embedding, c->vocab_size, dim);
        }
    }

    return model->logits;
}
