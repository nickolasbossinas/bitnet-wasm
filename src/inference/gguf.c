#include "gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * GGUF Parser Implementation
 *
 * Parses the GGUF binary format: header, metadata KV pairs, tensor info.
 * Uses a cursor-based approach reading from a memory buffer.
 */

/* --- Cursor-based binary reader --- */

typedef struct {
    const uint8_t *data;
    size_t         size;
    size_t         pos;
} cursor_t;

static int cursor_check(cursor_t *c, size_t need) {
    return (c->pos + need <= c->size) ? 0 : -1;
}

static uint8_t read_u8(cursor_t *c) {
    uint8_t v = c->data[c->pos];
    c->pos += 1;
    return v;
}

static uint16_t read_u16(cursor_t *c) {
    uint16_t v;
    memcpy(&v, &c->data[c->pos], 2);
    c->pos += 2;
    return v;
}

static uint32_t read_u32(cursor_t *c) {
    uint32_t v;
    memcpy(&v, &c->data[c->pos], 4);
    c->pos += 4;
    return v;
}

static int32_t read_i32(cursor_t *c) {
    int32_t v;
    memcpy(&v, &c->data[c->pos], 4);
    c->pos += 4;
    return v;
}

static uint64_t read_u64(cursor_t *c) {
    uint64_t v;
    memcpy(&v, &c->data[c->pos], 8);
    c->pos += 8;
    return v;
}

static float read_f32(cursor_t *c) {
    float v;
    memcpy(&v, &c->data[c->pos], 4);
    c->pos += 4;
    return v;
}

static double read_f64(cursor_t *c) {
    double v;
    memcpy(&v, &c->data[c->pos], 8);
    c->pos += 8;
    return v;
}

/* Read a GGUF string: uint64 length + chars (NOT null-terminated in file) */
static char *read_string(cursor_t *c) {
    uint64_t len = read_u64(c);
    if (cursor_check(c, len) != 0) return NULL;
    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, &c->data[c->pos], len);
    s[len] = '\0';
    c->pos += len;
    return s;
}

/* Skip a metadata value (used for KV pairs we don't care about) */
static int skip_value(cursor_t *c, uint32_t type);

static int skip_value(cursor_t *c, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            c->pos += 1;
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            c->pos += 2;
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            c->pos += 4;
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            c->pos += 8;
            break;
        case GGUF_TYPE_STRING: {
            char *s = read_string(c);
            free(s);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type = read_u32(c);
            uint64_t arr_len = read_u64(c);
            for (uint64_t i = 0; i < arr_len; i++) {
                if (skip_value(c, arr_type) != 0) return -1;
            }
            break;
        }
        default:
            return -1;
    }
    return cursor_check(c, 0);
}

/* Compute tensor size in bytes for a given type and element count */
static uint64_t tensor_type_size(int32_t type, uint64_t n_elements) {
    switch (type) {
        case GGML_TYPE_F32:  return n_elements * 4;
        case GGML_TYPE_F16:  return n_elements * 2;
        case GGML_TYPE_I8:   return n_elements;
        case GGML_TYPE_I16:  return n_elements * 2;
        case GGML_TYPE_I32:  return n_elements * 4;
        case GGML_TYPE_I64:  return n_elements * 8;
        case GGML_TYPE_F64:  return n_elements * 8;
        case GGML_TYPE_Q8_0: return (n_elements / 32) * 34;   /* 32 int8 + 1 fp16 scale */
        case GGML_TYPE_TQ1_0: return (n_elements / 256) * 54; /* 256 trits in 54 bytes */
        case GGML_TYPE_TQ2_0: return (n_elements / 256) * 66; /* 256 trits in 66 bytes */
        case GGML_TYPE_I2_S:  return (n_elements + 3) / 4;    /* 2 bits per weight, 4 per byte */
        default:
            /* For quantized types we don't handle, estimate conservatively */
            return n_elements;
    }
}

/* --- Public API --- */

int gguf_parse(gguf_context_t *ctx, const uint8_t *data, size_t size) {
    memset(ctx, 0, sizeof(*ctx));

    cursor_t c = { .data = data, .size = size, .pos = 0 };

    /* Header */
    if (cursor_check(&c, 24) != 0) return -1;

    uint32_t magic = read_u32(&c);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic 0x%08x (expected 0x%08x)\n", magic, GGUF_MAGIC);
        return -1;
    }

    ctx->version = read_u32(&c);
    if (ctx->version < 2 || ctx->version > 3) {
        fprintf(stderr, "gguf: unsupported version %u\n", ctx->version);
        return -1;
    }

    ctx->n_tensors = read_u64(&c);
    ctx->n_kv = read_u64(&c);

    printf("gguf: version=%u, tensors=%llu, kv_pairs=%llu\n",
           ctx->version, (unsigned long long)ctx->n_tensors,
           (unsigned long long)ctx->n_kv);

    /* Set defaults for metadata we want to extract */
    ctx->rope_theta = 500000.0f;
    ctx->rms_norm_eps = 1e-5f;
    ctx->max_seq_len = 4096;

    /* Parse metadata KV pairs */
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (cursor_check(&c, 1) != 0) return -1;
        char *key = read_string(&c);
        if (!key) return -1;
        uint32_t value_type = read_u32(&c);

        /*
         * Extract architecture metadata we care about.
         * Use suffix matching since the key prefix varies:
         *   llama.*, bitnet.*, bitnet-b1.58.*, etc.
         */
        if (strstr(key, ".embedding_length") != NULL) {
            ctx->hidden_size = (int32_t)read_u32(&c);
        } else if (strstr(key, ".feed_forward_length") != NULL) {
            ctx->intermediate_size = (int32_t)read_u32(&c);
        } else if (strstr(key, ".block_count") != NULL) {
            ctx->n_layers = (int32_t)read_u32(&c);
        } else if (strstr(key, ".attention.head_count_kv") != NULL) {
            /* Must check head_count_kv before head_count (substring match) */
            ctx->n_kv_heads = (int32_t)read_u32(&c);
        } else if (strstr(key, ".attention.head_count") != NULL) {
            ctx->n_heads = (int32_t)read_u32(&c);
        } else if (strstr(key, ".rope.freq_base") != NULL) {
            ctx->rope_theta = read_f32(&c);
        } else if (strstr(key, ".layer_norm_rms_epsilon") != NULL) {
            ctx->rms_norm_eps = read_f32(&c);
        } else if (strstr(key, ".context_length") != NULL) {
            ctx->max_seq_len = (int32_t)read_u32(&c);
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            /* Array of strings — count gives vocab size */
            /* Type should be GGUF_TYPE_ARRAY */
            if (value_type == GGUF_TYPE_ARRAY) {
                uint32_t arr_type = read_u32(&c);
                uint64_t arr_len = read_u64(&c);
                ctx->vocab_size = (int32_t)arr_len;
                /* Skip the actual token strings */
                for (uint64_t j = 0; j < arr_len; j++) {
                    if (skip_value(&c, arr_type) != 0) { free(key); return -1; }
                }
            } else {
                skip_value(&c, value_type);
            }
        } else {
            /* Skip values we don't need */
            if (skip_value(&c, value_type) != 0) {
                fprintf(stderr, "gguf: failed to skip KV '%s' (type %u)\n", key, value_type);
                free(key);
                return -1;
            }
        }
        free(key);
    }

    /* Compute head_dim */
    if (ctx->n_heads > 0 && ctx->hidden_size > 0) {
        ctx->head_dim = ctx->hidden_size / ctx->n_heads;
    }

    printf("gguf: model config: layers=%d, hidden=%d, intermediate=%d\n",
           ctx->n_layers, ctx->hidden_size, ctx->intermediate_size);
    printf("gguf: heads=%d, kv_heads=%d, head_dim=%d, vocab=%d\n",
           ctx->n_heads, ctx->n_kv_heads, ctx->head_dim, ctx->vocab_size);
    printf("gguf: rope_theta=%.1f, rms_eps=%.1e, max_seq=%d\n",
           ctx->rope_theta, ctx->rms_norm_eps, ctx->max_seq_len);

    /* Parse tensor info */
    ctx->tensors = (gguf_tensor_info_t *)calloc(ctx->n_tensors, sizeof(gguf_tensor_info_t));
    if (!ctx->tensors) return -1;

    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor_info_t *t = &ctx->tensors[i];
        t->name = read_string(&c);
        if (!t->name) return -1;
        t->n_dims = read_u32(&c);
        uint64_t n_elements = 1;
        for (int32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = (int64_t)read_u64(&c);
            n_elements *= t->dims[d];
        }
        t->type = (int32_t)read_u32(&c);
        t->offset = read_u64(&c);
        t->size_bytes = tensor_type_size(t->type, n_elements);
    }

    /* Compute data offset (aligned to 32 bytes by default) */
    uint64_t alignment = 32;
    ctx->data_offset = (c.pos + alignment - 1) & ~(alignment - 1);

    printf("gguf: tensor data starts at offset %llu\n",
           (unsigned long long)ctx->data_offset);

    return 0;
}

void gguf_free(gguf_context_t *ctx) {
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            free(ctx->tensors[i].name);
        }
        free(ctx->tensors);
    }
    memset(ctx, 0, sizeof(*ctx));
}

gguf_tensor_info_t *gguf_find_tensor(const gguf_context_t *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

const void *gguf_tensor_data(const gguf_context_t *ctx,
                             const gguf_tensor_info_t *tensor,
                             const uint8_t *file_data) {
    return &file_data[ctx->data_offset + tensor->offset];
}

const char *ggml_type_name(int32_t type) {
    switch (type) {
        case GGML_TYPE_F32:   return "F32";
        case GGML_TYPE_F16:   return "F16";
        case GGML_TYPE_Q8_0:  return "Q8_0";
        case GGML_TYPE_TQ1_0: return "TQ1_0";
        case GGML_TYPE_TQ2_0: return "TQ2_0";
        case GGML_TYPE_I2_S:  return "I2_S";
        case GGML_TYPE_I32:   return "I32";
        default:              return "unknown";
    }
}
