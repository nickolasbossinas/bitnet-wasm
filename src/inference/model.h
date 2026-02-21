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
    float          *token_embedding;  /* [vocab_size * hidden_size] */
    float          *output_norm;      /* [hidden_size] */
    /* output projection is tied to token_embedding */

    /* Per-layer weights */
    layer_weights_t *layers;          /* [n_layers] */

    /* KV cache: [n_layers][max_seq_len][n_kv_heads][head_dim] */
    float          *key_cache;
    float          *value_cache;

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
 * BitLinear: quantize activations -> TL1 GEMV -> scale output.
 * Wraps the existing optimized TL1 SIMD kernel.
 */
void bitlinear(float *out, const float *x, const tl1_weight_t *W);

/*
 * Forward pass: process one token, update KV cache, return logits.
 * Returns pointer to model->logits (valid until next forward call).
 */
float *forward(model_t *model, int32_t token, int32_t pos);

#endif /* BITNET_MODEL_H */
