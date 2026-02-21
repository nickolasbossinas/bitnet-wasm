#include "../inference/model.h"
#include "../inference/sampler.h"
#include "../kernels/tl1.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Inference Pipeline Test Harness
 *
 * Tests the forward pass math with small dimensions and random data.
 * No real model weights — validates correctness of each component:
 *   1. RMSNorm
 *   2. Softmax
 *   3. RoPE
 *   4. matmul_f32
 *   5. Sampler (argmax, top-p)
 *   6. Full forward pass (single layer, random TL1 weights)
 */

#define TEST_PASS  0
#define TEST_FAIL  1

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(name) do { \
    tests_run++; \
    printf("  [%d] %-40s ", tests_run, #name); \
    if (test_##name() == TEST_PASS) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

/* ---- Test: RMSNorm ---- */

static int test_rms_norm_basic(void) {
    /* RMSNorm([1,1,1,1], weight=[1,1,1,1], eps=0) should give [1,1,1,1]
     * because rms = sqrt(4/4) = 1, so x/rms = x */
    float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    rms_norm(out, x, w, 4, 0.0f);

    for (int i = 0; i < 4; i++) {
        if (fabsf(out[i] - 1.0f) > 1e-5f) return TEST_FAIL;
    }
    return TEST_PASS;
}

static int test_rms_norm_scaling(void) {
    /* RMSNorm([2,0,0,0], weight=[1,1,1,1], eps=0)
     * rms = sqrt(4/4) = 1, so x * (1/1) * w = [2,0,0,0] */
    float x[] = {2.0f, 0.0f, 0.0f, 0.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    rms_norm(out, x, w, 4, 0.0f);

    /* rms = sqrt((4+0+0+0)/4) = 1.0
     * out = x * (1/1) * w = [2,0,0,0] */
    if (fabsf(out[0] - 2.0f) > 1e-5f) return TEST_FAIL;
    if (fabsf(out[1] - 0.0f) > 1e-5f) return TEST_FAIL;
    return TEST_PASS;
}

static int test_rms_norm_weight(void) {
    /* RMSNorm([1,1], weight=[2,3], eps=0)
     * rms = sqrt((1+1)/2) = 1
     * out = [1*1*2, 1*1*3] = [2, 3] */
    float x[] = {1.0f, 1.0f};
    float w[] = {2.0f, 3.0f};
    float out[2];

    rms_norm(out, x, w, 2, 0.0f);

    if (fabsf(out[0] - 2.0f) > 1e-5f) return TEST_FAIL;
    if (fabsf(out[1] - 3.0f) > 1e-5f) return TEST_FAIL;
    return TEST_PASS;
}

/* ---- Test: Softmax ---- */

static int test_softmax_uniform(void) {
    /* softmax([0,0,0,0]) = [0.25, 0.25, 0.25, 0.25] */
    float x[] = {0.0f, 0.0f, 0.0f, 0.0f};
    softmax(x, 4);

    for (int i = 0; i < 4; i++) {
        if (fabsf(x[i] - 0.25f) > 1e-5f) return TEST_FAIL;
    }
    return TEST_PASS;
}

static int test_softmax_sum(void) {
    /* softmax should always sum to 1 */
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    softmax(x, 5);

    float sum = 0.0f;
    for (int i = 0; i < 5; i++) sum += x[i];
    if (fabsf(sum - 1.0f) > 1e-5f) return TEST_FAIL;

    /* Values should be monotonically increasing */
    for (int i = 1; i < 5; i++) {
        if (x[i] <= x[i-1]) return TEST_FAIL;
    }
    return TEST_PASS;
}

static int test_softmax_dominant(void) {
    /* softmax([100, 0, 0]) should be nearly [1, 0, 0] */
    float x[] = {100.0f, 0.0f, 0.0f};
    softmax(x, 3);

    if (x[0] < 0.999f) return TEST_FAIL;
    if (x[1] > 0.001f) return TEST_FAIL;
    return TEST_PASS;
}

/* ---- Test: matmul_f32 ---- */

static int test_matmul_identity(void) {
    /* [1 0; 0 1] * [3; 7] = [3; 7] */
    float W[] = {1.0f, 0.0f,
                 0.0f, 1.0f};
    float x[] = {3.0f, 7.0f};
    float out[2];

    matmul_f32(out, x, W, 2, 2);

    if (fabsf(out[0] - 3.0f) > 1e-5f) return TEST_FAIL;
    if (fabsf(out[1] - 7.0f) > 1e-5f) return TEST_FAIL;
    return TEST_PASS;
}

static int test_matmul_general(void) {
    /* [1 2 3; 4 5 6] * [1; 1; 1] = [6; 15] */
    float W[] = {1.0f, 2.0f, 3.0f,
                 4.0f, 5.0f, 6.0f};
    float x[] = {1.0f, 1.0f, 1.0f};
    float out[2];

    matmul_f32(out, x, W, 2, 3);

    if (fabsf(out[0] - 6.0f) > 1e-5f) return TEST_FAIL;
    if (fabsf(out[1] - 15.0f) > 1e-5f) return TEST_FAIL;
    return TEST_PASS;
}

/* ---- Test: RoPE ---- */

static int test_rope_position_zero(void) {
    /* At position 0, angle = 0 for all freq, so cos=1, sin=0.
     * RoPE should be identity at pos=0. */
    float q[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float q_orig[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k_orig[] = {5.0f, 6.0f, 7.0f, 8.0f};

    rope_apply(q, k, 1, 1, 4, 0, 10000.0f);

    for (int i = 0; i < 4; i++) {
        if (fabsf(q[i] - q_orig[i]) > 1e-5f) return TEST_FAIL;
        if (fabsf(k[i] - k_orig[i]) > 1e-5f) return TEST_FAIL;
    }
    return TEST_PASS;
}

static int test_rope_preserves_norm(void) {
    /* RoPE rotates pairs — should preserve vector norm */
    float q[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k[] = {1.0f, 1.0f, 1.0f, 1.0f};

    float q_norm_before = 0, k_norm_before = 0;
    for (int i = 0; i < 4; i++) {
        q_norm_before += q[i] * q[i];
        k_norm_before += k[i] * k[i];
    }

    rope_apply(q, k, 1, 1, 4, 42, 500000.0f);

    float q_norm_after = 0, k_norm_after = 0;
    for (int i = 0; i < 4; i++) {
        q_norm_after += q[i] * q[i];
        k_norm_after += k[i] * k[i];
    }

    if (fabsf(q_norm_before - q_norm_after) > 1e-4f) return TEST_FAIL;
    if (fabsf(k_norm_before - k_norm_after) > 1e-4f) return TEST_FAIL;
    return TEST_PASS;
}

/* ---- Test: Sampler ---- */

static int test_argmax(void) {
    float logits[] = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f};
    int32_t idx = sample_argmax(logits, 5);
    if (idx != 3) return TEST_FAIL;
    return TEST_PASS;
}

static int test_top_p_deterministic(void) {
    /* With one dominant logit and low temperature, should always pick it */
    float logits[] = {-100.0f, -100.0f, 100.0f, -100.0f};
    uint32_t rng = 12345;
    int32_t idx = sample_top_p(logits, 4, 0.9f, 1.0f, &rng);
    if (idx != 2) return TEST_FAIL;
    return TEST_PASS;
}

/* ---- Test: Full forward pass (tiny model, random weights) ---- */

static uint32_t test_rng = 42;

static uint32_t test_xorshift(void) {
    uint32_t x = test_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    test_rng = x;
    return x;
}

static int8_t test_rand_ternary(void) {
    return (int8_t)(test_xorshift() % 3) - 1;
}

static float test_rand_float(void) {
    return ((float)(test_xorshift() % 10000) / 5000.0f) - 1.0f;
}

/*
 * Create a small model with random weights to test the forward pass.
 * Using tiny dimensions to keep it fast:
 *   hidden=64, intermediate=128, heads=4, kv_heads=2, head_dim=16
 *   vocab=32, max_seq=16, 1 layer
 */
static int test_forward_pass(void) {
    model_t model;
    memset(&model, 0, sizeof(model));

    /* Config: tiny model */
    model.config.n_layers = 1;
    model.config.hidden_size = 64;
    model.config.intermediate_size = 128;
    model.config.n_heads = 4;
    model.config.n_kv_heads = 2;
    model.config.head_dim = 16;
    model.config.kv_dim = 2 * 16; /* 32 */
    model.config.vocab_size = 32;
    model.config.max_seq_len = 16;
    model.config.rope_theta = 500000.0f;
    model.config.rms_norm_eps = 1e-5f;

    int32_t dim = 64;
    int32_t kv_dim = 32;
    int32_t inter = 128;
    int32_t vocab = 32;

    /* Allocate scratch buffers and KV cache */
    if (model_alloc(&model) != 0) {
        fprintf(stderr, "  model_alloc failed\n");
        return TEST_FAIL;
    }

    /* Allocate and fill token embedding with random data */
    model.token_embedding = (float *)calloc(vocab * dim, sizeof(float));
    model.output_norm = (float *)calloc(dim, sizeof(float));
    if (!model.token_embedding || !model.output_norm) {
        model_free(&model);
        return TEST_FAIL;
    }

    for (int i = 0; i < vocab * dim; i++) {
        model.token_embedding[i] = test_rand_float() * 0.1f;
    }
    for (int i = 0; i < dim; i++) {
        model.output_norm[i] = 1.0f + test_rand_float() * 0.01f;
    }

    /* Set up layer weights */
    layer_weights_t *lw = &model.layers[0];

    /* Norm weights (all ~1.0) */
    lw->attn_norm = (float *)malloc(dim * sizeof(float));
    lw->ffn_norm = (float *)malloc(dim * sizeof(float));
    lw->attn_sub_norm = (float *)malloc(model.config.head_dim * sizeof(float));
    lw->ffn_sub_norm = (float *)malloc(inter * sizeof(float));

    for (int i = 0; i < dim; i++) {
        lw->attn_norm[i] = 1.0f;
        lw->ffn_norm[i] = 1.0f;
    }
    for (int i = 0; i < model.config.head_dim; i++) {
        lw->attn_sub_norm[i] = 1.0f;
    }
    for (int i = 0; i < inter; i++) {
        lw->ffn_sub_norm[i] = 1.0f;
    }

    /* Create random ternary weights and pack as TL1 */
    /* Helper: allocate raw ternary, pack, transpose */
    struct {
        tl1_weight_t *w;
        int32_t M, K;
    } weight_specs[] = {
        { &lw->attn_q,    dim,    dim },
        { &lw->attn_k,    kv_dim, dim },
        { &lw->attn_v,    kv_dim, dim },
        { &lw->attn_o,    dim,    dim },
        { &lw->ffn_gate,  inter,  dim },
        { &lw->ffn_up,    inter,  dim },
        { &lw->ffn_down,  dim,    inter },
    };
    int n_weights = sizeof(weight_specs) / sizeof(weight_specs[0]);

    for (int w = 0; w < n_weights; w++) {
        int32_t M = weight_specs[w].M;
        int32_t K = weight_specs[w].K;
        tl1_weight_t *tw = weight_specs[w].w;

        /* Generate random ternary matrix */
        int8_t *raw = (int8_t *)malloc(M * K);
        for (int i = 0; i < M * K; i++) {
            raw[i] = test_rand_ternary();
        }

        /* Pack into TL1 format */
        int32_t pairs = K / 2;
        int32_t bytes_per_row = (pairs + 1) / 2;
        tw->indices = (uint8_t *)calloc(M * bytes_per_row, 1);
        tw->indices_col = NULL;
        tw->M = M;
        tw->K = K;
        tw->scale = 1.0f;

        tl1_pack_weights(raw, tw->indices, M, K);
        tl1_transpose_weights(tw);

        free(raw);
    }

    /* Run forward pass for a few positions */
    for (int32_t pos = 0; pos < 4; pos++) {
        int32_t token = pos % vocab;
        float *logits = forward(&model, token, pos);

        if (!logits) {
            fprintf(stderr, "  forward returned NULL at pos %d\n", pos);
            /* Clean up TL1 weight data */
            for (int w = 0; w < n_weights; w++) {
                free(weight_specs[w].w->indices);
                free(weight_specs[w].w->indices_col);
            }
            model_free(&model);
            return TEST_FAIL;
        }

        /* Verify logits are finite */
        int has_nan = 0;
        int has_inf = 0;
        for (int32_t i = 0; i < vocab; i++) {
            if (isnan(logits[i])) has_nan = 1;
            if (isinf(logits[i])) has_inf = 1;
        }
        if (has_nan || has_inf) {
            fprintf(stderr, "  logits contain NaN/Inf at pos %d\n", pos);
            for (int w = 0; w < n_weights; w++) {
                free(weight_specs[w].w->indices);
                free(weight_specs[w].w->indices_col);
            }
            model_free(&model);
            return TEST_FAIL;
        }

        /* Verify softmax of logits sums to ~1 */
        float logits_copy[32];
        memcpy(logits_copy, logits, vocab * sizeof(float));
        softmax(logits_copy, vocab);
        float sum = 0.0f;
        for (int32_t i = 0; i < vocab; i++) sum += logits_copy[i];
        if (fabsf(sum - 1.0f) > 1e-4f) {
            fprintf(stderr, "  softmax(logits) sum = %f at pos %d\n", sum, pos);
            for (int w = 0; w < n_weights; w++) {
                free(weight_specs[w].w->indices);
                free(weight_specs[w].w->indices_col);
            }
            model_free(&model);
            return TEST_FAIL;
        }

        /* Verify argmax works on the logits */
        int32_t best = sample_argmax(logits, vocab);
        if (best < 0 || best >= vocab) {
            fprintf(stderr, "  argmax out of range: %d\n", best);
            for (int w = 0; w < n_weights; w++) {
                free(weight_specs[w].w->indices);
                free(weight_specs[w].w->indices_col);
            }
            model_free(&model);
            return TEST_FAIL;
        }
    }

    /* Clean up TL1 weight data */
    for (int w = 0; w < n_weights; w++) {
        free(weight_specs[w].w->indices);
        weight_specs[w].w->indices = NULL;
        free(weight_specs[w].w->indices_col);
        weight_specs[w].w->indices_col = NULL;
    }
    model_free(&model);
    return TEST_PASS;
}

/* ---- Main ---- */

int main(void) {
    printf("===========================================\n");
    printf(" BitNet Inference Pipeline Tests\n");
    printf("===========================================\n\n");

    printf(" RMSNorm:\n");
    RUN_TEST(rms_norm_basic);
    RUN_TEST(rms_norm_scaling);
    RUN_TEST(rms_norm_weight);

    printf("\n Softmax:\n");
    RUN_TEST(softmax_uniform);
    RUN_TEST(softmax_sum);
    RUN_TEST(softmax_dominant);

    printf("\n Matrix multiply:\n");
    RUN_TEST(matmul_identity);
    RUN_TEST(matmul_general);

    printf("\n RoPE:\n");
    RUN_TEST(rope_position_zero);
    RUN_TEST(rope_preserves_norm);

    printf("\n Sampler:\n");
    RUN_TEST(argmax);
    RUN_TEST(top_p_deterministic);

    printf("\n Forward pass:\n");
    RUN_TEST(forward_pass);

    printf("\n===========================================\n");
    printf(" Results: %d/%d passed\n", tests_passed, tests_run);
    printf("===========================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
