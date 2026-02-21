#include "model.h"
#include "../kernels/gemv.h"
#include "../kernels/tl1.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
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

    /* KV cache */
    size_t kv_size = (size_t)nl * seq * kv_dim * sizeof(float);
    model->key_cache   = (float *)calloc(1, kv_size);
    model->value_cache = (float *)calloc(1, kv_size);
    if (!model->key_cache || !model->value_cache) {
        fprintf(stderr, "model: failed to allocate KV cache (%.1f MB)\n",
                2.0f * kv_size / (1024.0f * 1024.0f));
        return -1;
    }

    /* Layer weights array (weights themselves are loaded separately) */
    model->layers = (layer_weights_t *)calloc(nl, sizeof(layer_weights_t));
    if (!model->layers) return -1;

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
    /* Compute sum of squares */
    float ss = 0.0f;
    for (int32_t i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / dim + eps);
    /* Normalize and scale */
    for (int32_t i = 0; i < dim; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

void softmax(float *x, int32_t size) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int32_t i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    /* Exp and sum */
    float sum = 0.0f;
    for (int32_t i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    /* Normalize */
    float inv = 1.0f / sum;
    for (int32_t i = 0; i < size; i++) {
        x[i] *= inv;
    }
}

void matmul_f32(float *out, const float *x, const float *W,
                int32_t M, int32_t K) {
    /* out[i] = sum_j(W[i*K + j] * x[j]) */
#ifdef __wasm_simd128__
    /* SIMD: 8-wide f32 accumulation (2 × f32x4) → ~3-4x speedup.
     * Critical for logits: 128256 rows × 2560 cols = 328M FMAs. */
    for (int32_t i = 0; i < M; i++) {
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
        /* Horizontal sum of 4 f32 lanes */
        v128_t hi = wasm_i64x2_shuffle(sum0, sum0, 1, 0);
        sum0 = wasm_f32x4_add(sum0, hi);
        v128_t hi2 = wasm_i32x4_shuffle(sum0, sum0, 1, 0, 3, 2);
        sum0 = wasm_f32x4_add(sum0, hi2);
        float sum = wasm_f32x4_extract_lane(sum0, 0);
        /* Scalar tail */
        for (; j < K; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
#else
    for (int32_t i = 0; i < M; i++) {
        float sum = 0.0f;
        const float *row = &W[i * K];
        for (int32_t j = 0; j < K; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
#endif
}

void rope_apply(float *q, float *k, int32_t n_heads, int32_t n_kv_heads,
                int32_t head_dim, int32_t pos, float theta) {
    /*
     * RoPE: rotate pairs of dimensions using position-dependent frequencies.
     *
     * Optimized: compute cos/sin once per dimension pair (head_dim/2 values),
     * then apply to all heads. Reduces trig calls from
     * head_dim/2 * (n_heads + n_kv_heads) to just head_dim/2.
     * For BitNet 2B4T: 1600 powf+cosf+sinf → 64 each (25x reduction).
     */
    int32_t half_dim = head_dim / 2;

    /* Pre-compute cos/sin for this position (stack-allocated) */
    float cos_cache[256], sin_cache[256];  /* head_dim/2 max (128/2=64) */

    for (int32_t i = 0; i < half_dim; i++) {
        float freq = 1.0f / powf(theta, (float)(i * 2) / (float)head_dim);
        float angle = (float)pos * freq;
        cos_cache[i] = cosf(angle);
        sin_cache[i] = sinf(angle);
    }

    /* Apply to all Q heads */
    for (int32_t h = 0; h < n_heads; h++) {
        float *qh = &q[h * head_dim];
        for (int32_t i = 0; i < half_dim; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_cache[i] - q1 * sin_cache[i];
            qh[i * 2 + 1] = q0 * sin_cache[i] + q1 * cos_cache[i];
        }
    }

    /* Apply to all K heads */
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
     */
    gemv_scratch_t *s = &model->scratch;
    float scale = W->scale * s->act_scale;

    tl1_gemv_simd_fast(W, s->lut_buf, s->lut_lo_buf, s->lut_hi_buf,
                        scale, out);
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

    /* Token embedding lookup */
    memcpy(x, &model->token_embedding[token * dim], dim * sizeof(float));

    /* Process each layer */
    for (int32_t l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &model->layers[l];

        /* ---- Attention block ---- */

        /* Pre-norm */
        rms_norm(model->xb, x, lw->attn_norm, dim, c->rms_norm_eps);

        /* Q, K, V projections — shared input, prepare LUT once */
        bitlinear_prepare(model, model->xb, dim);
        bitlinear_run(model->q, &lw->attn_q, model);
        bitlinear_run(model->k, &lw->attn_k, model);
        bitlinear_run(model->v, &lw->attn_v, model);

        /* Apply RoPE */
        rope_apply(model->q, model->k, n_heads, n_kv_heads,
                   head_dim, pos, c->rope_theta);

        /* Store K, V in cache */
        size_t kv_offset = (size_t)l * seq * kv_dim + (size_t)pos * kv_dim;
        memcpy(&model->key_cache[kv_offset], model->k, kv_dim * sizeof(float));
        memcpy(&model->value_cache[kv_offset], model->v, kv_dim * sizeof(float));

        /* Grouped Query Attention */
        for (int32_t h = 0; h < n_heads; h++) {
            float *qh = &model->q[h * head_dim];
            float *att = &model->att[h * seq];
            int32_t kv_h = h / kv_group;  /* which KV head */

            /* Compute attention scores: Q @ K^T / sqrt(head_dim) */
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int32_t t = 0; t <= pos; t++) {
                float *kh = &model->key_cache[(size_t)l * seq * kv_dim +
                                              (size_t)t * kv_dim +
                                              kv_h * head_dim];
                float score = 0.0f;
                for (int32_t d = 0; d < head_dim; d++) {
                    score += qh[d] * kh[d];
                }
                att[t] = score * scale;
            }

            /* Softmax over [0..pos] */
            softmax(att, pos + 1);

            /* Weighted sum of values */
            float *out_h = &model->xb[h * head_dim];
            memset(out_h, 0, head_dim * sizeof(float));
            for (int32_t t = 0; t <= pos; t++) {
                float *vh = &model->value_cache[(size_t)l * seq * kv_dim +
                                                (size_t)t * kv_dim +
                                                kv_h * head_dim];
                float a = att[t];
                for (int32_t d = 0; d < head_dim; d++) {
                    out_h[d] += a * vh[d];
                }
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

        /* Gate and up projections — shared input, prepare LUT once */
        bitlinear_prepare(model, model->xb, dim);
        bitlinear_run(model->hb, &lw->ffn_gate, model);
        bitlinear_run(model->hb2, &lw->ffn_up, model);

        /* Squared ReLU: max(0, gate)^2 * up */
        for (int32_t i = 0; i < inter; i++) {
            float g = model->hb[i];
            g = (g > 0.0f) ? g * g : 0.0f;  /* relu2 */
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

    /* Output logits: x @ embedding^T (tied weights) */
    matmul_f32(model->logits, x, model->token_embedding, c->vocab_size, dim);

    return model->logits;
}
