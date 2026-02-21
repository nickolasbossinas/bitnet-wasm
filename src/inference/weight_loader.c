#include "weight_loader.h"
#include "../kernels/tl1.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* --- F16 -> F32 conversion --- */

float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        /* Subnormal: normalize */
        exp = 1;
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
        exp = exp + (127 - 15);
    } else if (exp == 31) {
        /* Inf or NaN */
        uint32_t bits = sign | 0x7F800000 | ((uint32_t)mant << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    } else {
        exp = exp + (127 - 15);
    }

    uint32_t bits = sign | ((uint32_t)exp << 23) | ((uint32_t)mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* --- I2_S decoder --- */

void i2s_decode(const uint8_t *data, int8_t *out, int64_t n_elements) {
    int64_t full_bytes = n_elements / 4;
    int64_t remainder = n_elements % 4;

    for (int64_t i = 0; i < full_bytes; i++) {
        uint8_t byte = data[i];
        out[i * 4 + 0] = (int8_t)((byte >> 0) & 0x03) - 1;
        out[i * 4 + 1] = (int8_t)((byte >> 2) & 0x03) - 1;
        out[i * 4 + 2] = (int8_t)((byte >> 4) & 0x03) - 1;
        out[i * 4 + 3] = (int8_t)((byte >> 6) & 0x03) - 1;
    }

    if (remainder > 0) {
        uint8_t byte = data[full_bytes];
        for (int64_t j = 0; j < remainder; j++) {
            out[full_bytes * 4 + j] = (int8_t)((byte >> (j * 2)) & 0x03) - 1;
        }
    }
}

/* --- Helper: load F32 tensor --- */

static float *load_f32_tensor(const gguf_context_t *gguf,
                               const uint8_t *file_data,
                               const char *name,
                               int64_t expected_elements) {
    gguf_tensor_info_t *t = gguf_find_tensor(gguf, name);
    if (!t) {
        fprintf(stderr, "weight_loader: tensor '%s' not found\n", name);
        return NULL;
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "weight_loader: tensor '%s' is type %s, expected F32\n",
                name, ggml_type_name(t->type));
        return NULL;
    }

    int64_t n_elements = 1;
    for (int d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];

    if (expected_elements > 0 && n_elements != expected_elements) {
        fprintf(stderr, "weight_loader: tensor '%s' has %lld elements, expected %lld\n",
                name, (long long)n_elements, (long long)expected_elements);
        return NULL;
    }

    const float *src = (const float *)gguf_tensor_data(gguf, t, file_data);
    float *dst = (float *)malloc(n_elements * sizeof(float));
    if (!dst) return NULL;
    memcpy(dst, src, n_elements * sizeof(float));
    return dst;
}

/* --- Helper: load F16 tensor as F32 --- */

static float *load_f16_as_f32(const gguf_context_t *gguf,
                               const uint8_t *file_data,
                               const char *name,
                               int64_t expected_elements) {
    gguf_tensor_info_t *t = gguf_find_tensor(gguf, name);
    if (!t) {
        fprintf(stderr, "weight_loader: tensor '%s' not found\n", name);
        return NULL;
    }
    if (t->type != GGML_TYPE_F16) {
        fprintf(stderr, "weight_loader: tensor '%s' is type %s, expected F16\n",
                name, ggml_type_name(t->type));
        return NULL;
    }

    int64_t n_elements = 1;
    for (int d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];

    if (expected_elements > 0 && n_elements != expected_elements) {
        fprintf(stderr, "weight_loader: tensor '%s' has %lld elements, expected %lld\n",
                name, (long long)n_elements, (long long)expected_elements);
        return NULL;
    }

    const uint16_t *src = (const uint16_t *)gguf_tensor_data(gguf, t, file_data);
    float *dst = (float *)malloc(n_elements * sizeof(float));
    if (!dst) return NULL;

    for (int64_t i = 0; i < n_elements; i++) {
        dst[i] = f16_to_f32(src[i]);
    }
    return dst;
}

/* --- Helper: load I2_S ternary tensor into TL1 format --- */

static int load_tl1_weight(tl1_weight_t *w,
                            const gguf_context_t *gguf,
                            const uint8_t *file_data,
                            const char *name,
                            int32_t M, int32_t K) {
    gguf_tensor_info_t *t = gguf_find_tensor(gguf, name);
    if (!t) {
        fprintf(stderr, "weight_loader: tensor '%s' not found\n", name);
        return -1;
    }
    if (t->type != GGML_TYPE_I2_S) {
        fprintf(stderr, "weight_loader: tensor '%s' is type %s, expected I2_S\n",
                name, ggml_type_name(t->type));
        return -1;
    }

    int64_t n_elements = (int64_t)M * K;

    /* Decode I2_S -> int8 ternary */
    int8_t *raw = (int8_t *)malloc(n_elements);
    if (!raw) return -1;

    const uint8_t *src = (const uint8_t *)gguf_tensor_data(gguf, t, file_data);
    i2s_decode(src, raw, n_elements);

    /* Pack into TL1 format */
    int32_t pairs = K / 2;
    int32_t bytes_per_row = (pairs + 1) / 2;
    w->indices = (uint8_t *)calloc(M * bytes_per_row, 1);
    w->indices_col = NULL;
    w->M = M;
    w->K = K;
    w->scale = 1.0f;

    if (!w->indices) {
        free(raw);
        return -1;
    }

    tl1_pack_weights(raw, w->indices, M, K);
    tl1_transpose_weights(w);

    free(raw);
    return 0;
}

/* --- Main weight loading function --- */

int model_load_weights(model_t *model, const gguf_context_t *gguf,
                       const uint8_t *file_data, int32_t n_layers_to_load) {
    model_config_t *c = &model->config;
    int32_t dim = c->hidden_size;
    int32_t inter = c->intermediate_size;
    int32_t kv_dim = c->kv_dim;
    int32_t n_layers = (n_layers_to_load > 0 && n_layers_to_load < c->n_layers)
                       ? n_layers_to_load : c->n_layers;

    printf("weight_loader: loading %d/%d layers (dim=%d, inter=%d, kv_dim=%d)\n",
           n_layers, c->n_layers, dim, inter, kv_dim);

    /* Load token embedding (F16 -> F32) */
    int64_t emb_size = (int64_t)c->vocab_size * dim;
    model->token_embedding = load_f16_as_f32(gguf, file_data,
                                              "token_embd.weight", emb_size);
    if (!model->token_embedding) return -1;
    printf("weight_loader: loaded token_embd.weight (%.1f MB)\n",
           emb_size * 4.0f / (1024.0f * 1024.0f));

    /* Load output norm (F32) */
    model->output_norm = load_f32_tensor(gguf, file_data,
                                          "output_norm.weight", dim);
    if (!model->output_norm) return -1;

    /* Load per-layer weights */
    for (int32_t l = 0; l < n_layers; l++) {
        layer_weights_t *lw = &model->layers[l];
        char name[128];

        printf("weight_loader: loading layer %d/%d...\n", l, n_layers);

        /* Norm weights (F32) */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        lw->attn_norm = load_f32_tensor(gguf, file_data, name, dim);
        if (!lw->attn_norm) return -1;

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        lw->ffn_norm = load_f32_tensor(gguf, file_data, name, dim);
        if (!lw->ffn_norm) return -1;

        snprintf(name, sizeof(name), "blk.%d.attn_sub_norm.weight", l);
        lw->attn_sub_norm = load_f32_tensor(gguf, file_data, name, dim);
        if (!lw->attn_sub_norm) return -1;

        snprintf(name, sizeof(name), "blk.%d.ffn_sub_norm.weight", l);
        lw->ffn_sub_norm = load_f32_tensor(gguf, file_data, name, inter);
        if (!lw->ffn_sub_norm) return -1;

        /* Ternary weights (I2_S -> TL1) */
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        if (load_tl1_weight(&lw->attn_q, gguf, file_data, name, dim, dim) != 0)
            return -1;

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        if (load_tl1_weight(&lw->attn_k, gguf, file_data, name, kv_dim, dim) != 0)
            return -1;

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        if (load_tl1_weight(&lw->attn_v, gguf, file_data, name, kv_dim, dim) != 0)
            return -1;

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        if (load_tl1_weight(&lw->attn_o, gguf, file_data, name, dim, dim) != 0)
            return -1;

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        if (load_tl1_weight(&lw->ffn_gate, gguf, file_data, name, inter, dim) != 0)
            return -1;

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        if (load_tl1_weight(&lw->ffn_up, gguf, file_data, name, inter, dim) != 0)
            return -1;

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        if (load_tl1_weight(&lw->ffn_down, gguf, file_data, name, dim, inter) != 0)
            return -1;

        printf("weight_loader: layer %d loaded\n", l);
    }

    printf("weight_loader: all weights loaded successfully\n");
    return 0;
}
