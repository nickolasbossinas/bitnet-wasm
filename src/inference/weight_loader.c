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
    /*
     * I2_S format: 4 groups of QK/4 weights interleaved within each byte.
     * Within blocks of QK_I2S=128 weights (32 bytes):
     *   - Byte b, bits 6-7: weight at group 0, position b  (weight[b])
     *   - Byte b, bits 4-5: weight at group 1, position b  (weight[32+b])
     *   - Byte b, bits 2-3: weight at group 2, position b  (weight[64+b])
     *   - Byte b, bits 0-1: weight at group 3, position b  (weight[96+b])
     * Each 2-bit code maps: 0 -> -1, 1 -> 0, 2 -> +1
     */
    int64_t QK = 128;  /* weights per interleaving block */
    int64_t BPB = 32;  /* bytes per block (QK / 4) */
    int64_t n_blocks = n_elements / QK;
    int64_t tail = n_elements % QK;

    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const uint8_t *blk_data = &data[blk * BPB];
        int8_t *blk_out = &out[blk * QK];

        for (int32_t b = 0; b < BPB; b++) {
            uint8_t byte = blk_data[b];
            blk_out[0 * BPB + b] = (int8_t)((byte >> 6) & 0x03) - 1;
            blk_out[1 * BPB + b] = (int8_t)((byte >> 4) & 0x03) - 1;
            blk_out[2 * BPB + b] = (int8_t)((byte >> 2) & 0x03) - 1;
            blk_out[3 * BPB + b] = (int8_t)((byte >> 0) & 0x03) - 1;
        }
    }

    /* Handle tail elements (< 128) with simple MSB-first sequential decode */
    if (tail > 0) {
        int64_t tail_offset = n_blocks * QK;
        int64_t tail_bytes = (tail + 3) / 4;
        const uint8_t *tail_data = &data[n_blocks * BPB];
        int64_t idx = 0;
        for (int64_t i = 0; i < tail_bytes && idx < tail; i++) {
            uint8_t byte = tail_data[i];
            if (idx < tail) out[tail_offset + idx++] = (int8_t)((byte >> 6) & 0x03) - 1;
            if (idx < tail) out[tail_offset + idx++] = (int8_t)((byte >> 4) & 0x03) - 1;
            if (idx < tail) out[tail_offset + idx++] = (int8_t)((byte >> 2) & 0x03) - 1;
            if (idx < tail) out[tail_offset + idx++] = (int8_t)((byte >> 0) & 0x03) - 1;
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

/* --- Helper: load F16 tensor as raw uint16_t (no conversion) --- */

static uint16_t *load_f16_raw(const gguf_context_t *gguf,
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
    uint16_t *dst = (uint16_t *)malloc(n_elements * sizeof(uint16_t));
    if (!dst) return NULL;
    memcpy(dst, src, n_elements * sizeof(uint16_t));
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

    /* Read per-tensor scale factor appended after packed I2_S data.
     * I2_S format: [ceil(n/4) bytes of packed 2-bit data] [float32 scale]
     * The scale = max(|original_weights|) from the quantization step.
     */
    int64_t packed_bytes = (n_elements + 3) / 4;
    float i2s_scale;
    memcpy(&i2s_scale, &src[packed_bytes], sizeof(float));

    /* Pack into TL1 format */
    int32_t pairs = K / 2;
    int32_t bytes_per_row = (pairs + 1) / 2;
    w->indices = (uint8_t *)calloc(M * bytes_per_row, 1);
    w->indices_col = NULL;
    w->M = M;
    w->K = K;
    w->scale = i2s_scale;

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

    fprintf(stderr, "weight_loader: loading %d/%d layers (dim=%d, inter=%d, kv_dim=%d)\n",
            n_layers, c->n_layers, dim, inter, kv_dim);

    /* Load token embedding as raw F16 (halves memory, 2x faster logits matmul) */
    int64_t emb_size = (int64_t)c->vocab_size * dim;
    model->token_embedding = load_f16_raw(gguf, file_data,
                                           "token_embd.weight", emb_size);
    if (!model->token_embedding) return -1;
    fprintf(stderr, "weight_loader: loaded token_embd.weight F16 (%.1f MB)\n",
            emb_size * 2.0f / (1024.0f * 1024.0f));

    /* Load output norm (F32) */
    model->output_norm = load_f32_tensor(gguf, file_data,
                                          "output_norm.weight", dim);
    if (!model->output_norm) return -1;

    /* Load per-layer weights */
    for (int32_t l = 0; l < n_layers; l++) {
        layer_weights_t *lw = &model->layers[l];
        char name[128];

        fprintf(stderr, "weight_loader: loading layer %d/%d...\n", l, n_layers);

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

        fprintf(stderr, "weight_loader: layer %d loaded\n", l);
    }

    fprintf(stderr, "weight_loader: all weights loaded successfully\n");
    return 0;
}
