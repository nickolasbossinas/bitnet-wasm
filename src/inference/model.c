#include "model.h"
#include "../kernels/gemv.h"
#include "../kernels/tl1.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

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
    for (int32_t i = 0; i < M; i++) {
        float sum = 0.0f;
        const float *row = &W[i * K];
        for (int32_t j = 0; j < K; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

void rope_apply(float *q, float *k, int32_t n_heads, int32_t n_kv_heads,
                int32_t head_dim, int32_t pos, float theta) {
    /*
     * RoPE: rotate pairs of dimensions using position-dependent frequencies.
     * freq_i = 1 / (theta ^ (2i / head_dim))
     * For each pair (x[2i], x[2i+1]):
     *   x[2i]   = x[2i] * cos(freq) - x[2i+1] * sin(freq)
     *   x[2i+1] = x[2i] * sin(freq) + x[2i+1] * cos(freq)
     */
    for (int32_t h = 0; h < n_heads; h++) {
        float *qh = &q[h * head_dim];
        for (int32_t i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float q0 = qh[i];
            float q1 = qh[i + 1];
            qh[i]     = q0 * cos_a - q1 * sin_a;
            qh[i + 1] = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int32_t h = 0; h < n_kv_heads; h++) {
        float *kh = &k[h * head_dim];
        for (int32_t i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float k0 = kh[i];
            float k1 = kh[i + 1];
            kh[i]     = k0 * cos_a - k1 * sin_a;
            kh[i + 1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

void bitlinear(float *out, const float *x, const tl1_weight_t *W) {
    /*
     * BitLinear: ternary GEMV with activation quantization.
     * Uses the existing optimized gemv_run() which handles:
     *   1. Quantize float activations -> int8
     *   2. Build TL1 LUT
     *   3. TL1 SIMD GEMV
     *   4. Scale output
     */
    gemv_run(KERNEL_TL1_SIMD, W, x, out, W->M, W->K);
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

        /* Q, K, V projections (ternary GEMV) */
        bitlinear(model->q, model->xb, &lw->attn_q);
        bitlinear(model->k, model->xb, &lw->attn_k);
        bitlinear(model->v, model->xb, &lw->attn_v);

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

        /* Output projection + residual */
        bitlinear(model->xb2, model->xb, &lw->attn_o);
        for (int32_t i = 0; i < dim; i++) {
            x[i] += model->xb2[i];
        }

        /* ---- FFN block ---- */

        /* Pre-norm */
        rms_norm(model->xb, x, lw->ffn_norm, dim, c->rms_norm_eps);

        /* Gate and up projections */
        bitlinear(model->hb, model->xb, &lw->ffn_gate);
        bitlinear(model->hb2, model->xb, &lw->ffn_up);

        /* Squared ReLU: max(0, gate)^2 * up */
        for (int32_t i = 0; i < inter; i++) {
            float g = model->hb[i];
            g = (g > 0.0f) ? g * g : 0.0f;  /* relu2 */
            model->hb[i] = g * model->hb2[i];
        }

        /* SubLN: RMSNorm before down projection */
        rms_norm(model->hb2, model->hb, lw->ffn_sub_norm, inter, c->rms_norm_eps);

        /* Down projection + residual */
        bitlinear(model->xb, model->hb2, &lw->ffn_down);
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
