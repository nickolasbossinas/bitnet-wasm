#ifndef BITNET_GGUF_H
#define BITNET_GGUF_H

#include <stdint.h>
#include <stddef.h>

/*
 * Minimal GGUF parser for BitNet b1.58 model loading.
 *
 * GGUF file layout:
 *   1. Header: magic, version, tensor_count, metadata_kv_count
 *   2. Metadata KV pairs: typed key-value store
 *   3. Tensor info array: name, dims, type, offset per tensor
 *   4. Tensor data: contiguous binary blob (aligned)
 *
 * Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */

#define GGUF_MAGIC 0x46554747  /* "GGUF" in little-endian */
#define GGUF_VERSION 3

/* GGML tensor types we care about */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_I2_S    = 36,  /* BitNet I2_S: 2-bit ternary, 4 weights/byte */
} ggml_type_t;

/* GGUF metadata value types */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type_t;

/* Tensor info extracted from GGUF */
typedef struct {
    char    *name;
    int32_t  n_dims;
    int64_t  dims[4];
    int32_t  type;        /* ggml_type_t */
    uint64_t offset;      /* byte offset into tensor data section */
    uint64_t size_bytes;  /* computed total size */
} gguf_tensor_info_t;

/* Tokenizer metadata extracted from GGUF KV pairs */
typedef struct {
    char    **tokens;          /* [vocab_size] token strings (owned) */
    int32_t  *token_types;     /* [vocab_size] 1=normal, 3=special */
    int32_t   vocab_size;
    char    **merges;          /* [n_merges] "tokenA tokenB" strings (owned) */
    int32_t   n_merges;
    int32_t   bos_token_id;
    int32_t   eos_token_id;
    int32_t   add_bos;
    int32_t   add_eos;
    char     *model_type;      /* "gpt2", "llama", etc. */
} gguf_tokenizer_t;

/* Parsed GGUF context */
typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;

    /* Tensor info array */
    gguf_tensor_info_t *tensors;

    /* Byte offset where tensor data begins in the file */
    uint64_t data_offset;

    /* Model architecture metadata (extracted from KV pairs) */
    int32_t  n_layers;
    int32_t  hidden_size;
    int32_t  intermediate_size;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  head_dim;
    int32_t  vocab_size;
    int32_t  max_seq_len;
    float    rope_theta;
    float    rms_norm_eps;

    /* Tokenizer data */
    gguf_tokenizer_t tokenizer;
} gguf_context_t;

/*
 * Parse a GGUF file from memory.
 * Returns 0 on success, -1 on error.
 */
int gguf_parse(gguf_context_t *ctx, const uint8_t *data, size_t size);

/*
 * Free all allocations in a parsed context.
 */
void gguf_free(gguf_context_t *ctx);

/*
 * Find a tensor by name. Returns NULL if not found.
 */
gguf_tensor_info_t *gguf_find_tensor(const gguf_context_t *ctx, const char *name);

/*
 * Get a pointer to a tensor's raw data in the memory-mapped file.
 */
const void *gguf_tensor_data(const gguf_context_t *ctx,
                             const gguf_tensor_info_t *tensor,
                             const uint8_t *file_data);

/*
 * Return the name of a ggml type.
 */
const char *ggml_type_name(int32_t type);

#endif /* BITNET_GGUF_H */
