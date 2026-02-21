#ifndef BITNET_MODEL_H
#define BITNET_MODEL_H

#include "../kernels/types.h"
#include "gguf.h"

/*
 * BitNet b1.58 2B4T Model — Forward Pass
 *
 * Architecture: LLaMA-like decoder-only transformer
 *   - Ternary weights {-1, 0, 1} in all linear layers
 *   - Squared ReLU activation (max(0,x)^2)
 *   - RMSNorm + SubLN (extra norm before W_o and W_down)
 *   - Grouped Query Attention (GQA): 20 query heads, 5 KV heads
 *   - RoPE positional encoding (theta=500000)
 *   - Tied word embeddings (output projection = embedding^T)
 */

/* Model configuration (matches GGUF metadata) */
typedef struct {
    int32_t n_layers;          /* 30 */
    int32_t hidden_size;       /* 2560 */
    int32_t intermediate_size; /* 6912 */
    int32_t n_heads;           /* 20 (query heads) */
    int32_t n_kv_heads;        /* 5 (KV heads, GQA ratio 4:1) */
    int32_t head_dim;          /* 128 = hidden_size / n_heads */
    int32_t kv_dim;            /* 640 = n_kv_heads * head_dim */
    int32_t vocab_size;        /* 128256 */
    int32_t max_seq_len;       /* 4096 */
    float   rope_theta;        /* 500000.0 */
    float   rms_norm_eps;      /* 1e-5 */
} model_config_t;

/* GEMV scratch buffers (pre-allocated, reused across forward calls).
 * Eliminates 840 malloc/free pairs per token (7 calls × 30 layers × 4 allocs).
 * Also enables LUT caching when multiple projections share the same input. */
typedef struct {
    int8_t  *quant_buf;    /* [max_K] quantized activations */
    int16_t *lut_buf;      /* [max_K/2 * 16] TL1 LUT */
    uint8_t *lut_lo_buf;   /* [max_K/2 * 16] pre-split lo bytes */
    uint8_t *lut_hi_buf;   /* [max_K/2 * 16] pre-split hi bytes */
    float    act_scale;    /* current activation quantization scale */
    int32_t  prepared_K;   /* K dimension of current preparation */
    int32_t  max_K;        /* allocation size (max of hidden, intermediate) */
} gemv_scratch_t;

/* Forward declaration for thread pool */
struct thread_pool_s;

/* Per-layer weights */
typedef struct {
    /* Ternary weights (TL1 packed for SIMD) */
    tl1_weight_t attn_q;       /* [hidden, hidden] = [2560, 2560] */
    tl1_weight_t attn_k;       /* [kv_dim, hidden] = [640, 2560] */
    tl1_weight_t attn_v;       /* [kv_dim, hidden] = [640, 2560] */
    tl1_weight_t attn_o;       /* [hidden, hidden] = [2560, 2560] */
    tl1_weight_t ffn_gate;     /* [intermediate, hidden] = [6912, 2560] */
    tl1_weight_t ffn_up;       /* [intermediate, hidden] = [6912, 2560] */
    tl1_weight_t ffn_down;     /* [hidden, intermediate] = [2560, 6912] */
    /* Norm weights (F32) */
    float *attn_norm;          /* [hidden_size] RMSNorm before attention */
    float *ffn_norm;           /* [hidden_size] RMSNorm before FFN */
    float *attn_sub_norm;      /* [hidden_size] SubLN before W_o */
    float *ffn_sub_norm;       /* [intermediate_size] SubLN before W_down */
} layer_weights_t;

/* Full model state */
typedef struct {
    model_config_t  config;

    /* Global weights */
    uint16_t       *token_embedding;  /* [vocab_size * hidden_size] F16 (freed after INT8 quantization) */
    float          *output_norm;      /* [hidden_size] */
    /* output projection is tied to token_embedding */

    /* INT8 quantized embedding (replaces F16 for logits matmul + embedding lookup) */
    int8_t         *emb_quantized;    /* [vocab_size * hidden_size] */
    float          *emb_row_scales;   /* [vocab_size] per-row scale factors */

    /* Precomputed RoPE sin/cos tables (eliminates per-token trig calls) */
    float          *rope_cos;         /* [max_seq_len * head_dim/2] */
    float          *rope_sin;         /* [max_seq_len * head_dim/2] */

    /* Per-layer weights */
    layer_weights_t *layers;          /* [n_layers] */

    /* KV cache: [n_layers][max_seq_len][n_kv_heads][head_dim] (F16, halves memory) */
    uint16_t       *key_cache;
    uint16_t       *value_cache;

    /* Scratch buffers (pre-allocated for forward pass) */
    float *x;          /* [hidden_size] current hidden state */
    float *xb;         /* [hidden_size] buffer after norm */
    float *q;          /* [hidden_size] = [n_heads * head_dim] */
    float *k;          /* [kv_dim] = [n_kv_heads * head_dim] */
    float *v;          /* [kv_dim] */
    float *att;        /* [n_heads * max_seq_len] attention scores */
    float *xb2;        /* [hidden_size] second buffer */
    float *hb;         /* [intermediate_size] FFN hidden */
    float *hb2;        /* [intermediate_size] FFN hidden 2 */
    float *logits;     /* [vocab_size] output logits */

    /* GEMV scratch (eliminates per-call malloc in bitlinear) */
    gemv_scratch_t scratch;

    /* Thread pool for parallel GEMV/matmul (NULL = single-threaded) */
    struct thread_pool_s *pool;
} model_t;

/*
 * Initialize model config from GGUF context.
 */
void model_config_from_gguf(model_config_t *config, const gguf_context_t *gguf);

/*
 * Allocate scratch buffers and KV cache.
 * Call after config is set. Does NOT load weights.
 */
int model_alloc(model_t *model);

/*
 * Free all model memory.
 */
void model_free(model_t *model);

/*
 * RMSNorm: out = (x / rms(x)) * weight
 */
void rms_norm(float *out, const float *x, const float *weight,
              int32_t dim, float eps);

/*
 * Apply RoPE to query and key vectors in-place.
 * q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
 */
void rope_apply(float *q, float *k, int32_t n_heads, int32_t n_kv_heads,
                int32_t head_dim, int32_t pos, float theta);

/*
 * Softmax in-place.
 */
void softmax(float *x, int32_t size);

/*
 * FP32 matrix-vector multiply: out[M] = W[M,K] * x[K]
 * W is row-major. Used for embedding output projection.
 */
void matmul_f32(float *out, const float *x, const float *W,
                int32_t M, int32_t K);

/*
 * Row-range variant: compute rows [row_start, row_end) only.
 * Used by thread pool for parallel matmul.
 */
void matmul_f32_range(float *out, const float *x, const float *W,
                      int32_t K, int32_t row_start, int32_t row_end);

/*
 * F16->F32 conversion (portable, no hardware FP16 required).
 */
float f16_to_f32(uint16_t h);

/*
 * F16 x F32 matrix-vector multiply: out[M] = W_f16[M,K] * x[K]
 * W is row-major F16 (uint16_t). x and out are F32.
 * Used for logits with F16 token embedding (halves BW vs F32).
 */
void matmul_f16f32(float *out, const float *x, const uint16_t *W,
                    int32_t M, int32_t K);
void matmul_f16f32_range(float *out, const float *x, const uint16_t *W,
                          int32_t K, int32_t row_start, int32_t row_end);

/*
 * INT8 x INT8 matrix-vector multiply: out[M] = W_i8[M,K] · x_i8[K]
 * Both W and x are symmetric int8 quantized. Per-row scales for W.
 * out[i] = dot(W[i], x) * row_scales[i] * x_scale
 * Used for logits matmul (2x less bandwidth than F16).
 */
void matmul_i8(float *out, const int8_t *x_quant, float x_scale,
               const int8_t *W, const float *row_scales,
               int32_t M, int32_t K);
void matmul_i8_range(float *out, const int8_t *x_quant, float x_scale,
                     const int8_t *W, const float *row_scales,
                     int32_t K, int32_t row_start, int32_t row_end);

/*
 * BitLinear: quantize activations -> TL1 GEMV -> scale output.
 * Wraps the existing optimized TL1 SIMD kernel.
 * Legacy interface — allocates/frees per call. Used by benchmark.
 */
void bitlinear(float *out, const float *x, const tl1_weight_t *W);

/*
 * Optimized bitlinear: two-phase interface for the forward pass.
 *
 * bitlinear_prepare: quantize activations + build/presplit LUT into
 * model scratch buffers. Call once per unique input vector.
 *
 * bitlinear_run: execute GEMV using pre-built LUT. No allocation.
 * Call for each weight matrix that shares the same prepared input.
 *
 * Example: Q, K, V projections share the same input after attn_norm,
 * so prepare once + run 3 times (saves 2 quantize+LUT builds).
 */
void bitlinear_prepare(model_t *model, const float *x, int32_t K);
void bitlinear_run(float *out, const tl1_weight_t *W, model_t *model);

/*
 * Forward pass: process one token, update KV cache, return logits.
 * Returns pointer to model->logits (valid until next forward call).
 */
float *forward(model_t *model, int32_t token, int32_t pos);

#endif /* BITNET_MODEL_H */
