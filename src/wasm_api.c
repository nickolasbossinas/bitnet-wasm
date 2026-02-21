#include <emscripten.h>
#include "inference/gguf.h"
#include "inference/model.h"
#include "inference/thread_pool.h"
#include "inference/weight_loader.h"
#include "inference/tokenizer.h"
#include "inference/generate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * WASM API for browser-based inference.
 *
 * Provides three entry points callable from JavaScript:
 *   bitnet_init()     — load model from GGUF buffer
 *   bitnet_generate() — generate text, streaming tokens via postMessage
 *   bitnet_free()     — release all model memory
 *
 * Token output is streamed to the main thread via EM_ASM + self.postMessage().
 * All inference runs in a Web Worker to avoid blocking the UI.
 */

/* Global state — single model instance per worker */
static model_t g_model;
static tokenizer_t g_tok;
static thread_pool_t g_pool;
static int g_initialized = 0;

/* Token callback: stream each token to main thread via postMessage */
static void wasm_token_callback(const char *piece, int32_t token_id,
                                 void *user_data) {
    (void)user_data;
    EM_ASM({
        self.postMessage({
            type: 'token',
            piece: UTF8ToString($0),
            id: $1
        });
    }, piece, token_id);
}

/*
 * Load model from GGUF data already in WASM heap.
 *
 * gguf_data: pointer to GGUF file bytes (allocated by JS via _malloc)
 * size:      byte length of GGUF data
 * n_layers:  number of layers to load (-1 or 0 = all)
 * n_threads: number of worker threads for parallel GEMV (0 = single-threaded)
 *
 * Returns: 0 on success, negative error code on failure
 *   -1: GGUF parse failed
 *   -2: tokenizer init failed
 *   -3: model alloc failed
 *   -4: weight loading failed
 *
 * Note: caller should _free(gguf_data) after this returns — weights are
 * copied out during loading.
 */
EMSCRIPTEN_KEEPALIVE
int bitnet_init(const uint8_t *gguf_data, size_t size, int32_t n_layers,
                int32_t n_threads) {
    if (g_initialized) {
        fprintf(stderr, "bitnet_init: already initialized, call bitnet_free first\n");
        return -5;
    }

    /* 1. Parse GGUF */
    gguf_context_t gguf;
    if (gguf_parse(&gguf, gguf_data, size) != 0) {
        fprintf(stderr, "bitnet_init: GGUF parse failed\n");
        return -1;
    }
    fprintf(stderr, "Model: %d layers, %d hidden, %d vocab\n",
            gguf.n_layers, gguf.hidden_size, gguf.vocab_size);

    /* 2. Init tokenizer */
    if (tokenizer_init(&g_tok, &gguf.tokenizer) != 0) {
        fprintf(stderr, "bitnet_init: tokenizer init failed\n");
        gguf_free(&gguf);
        return -2;
    }
    fprintf(stderr, "Tokenizer: %d vocab, %d merges\n",
            g_tok.vocab_size, g_tok.n_merges);

    /* 3. Configure and allocate model */
    memset(&g_model, 0, sizeof(g_model));
    model_config_from_gguf(&g_model.config, &gguf);

    int32_t layers_to_load = n_layers;
    if (layers_to_load > 0 && layers_to_load < g_model.config.n_layers) {
        g_model.config.n_layers = layers_to_load;
        fprintf(stderr, "Loading %d of %d layers\n", layers_to_load, gguf.n_layers);
    } else {
        layers_to_load = g_model.config.n_layers;
    }

    if (model_alloc(&g_model) != 0) {
        fprintf(stderr, "bitnet_init: model alloc failed\n");
        tokenizer_free(&g_tok);
        gguf_free(&gguf);
        return -3;
    }

    /* 4. Load weights */
    fprintf(stderr, "Loading weights...\n");
    if (model_load_weights(&g_model, &gguf, gguf_data, layers_to_load) != 0) {
        fprintf(stderr, "bitnet_init: weight loading failed\n");
        model_free(&g_model);
        tokenizer_free(&g_tok);
        gguf_free(&gguf);
        return -4;
    }

    /* 5. Free GGUF metadata (weights are copied out) */
    gguf_free(&gguf);

    /* 6. Initialize thread pool for parallel GEMV/matmul */
    if (n_threads > 0) {
        if (thread_pool_init(&g_pool, n_threads) == 0) {
            g_model.pool = &g_pool;
        } else {
            fprintf(stderr, "bitnet_init: thread pool init failed, using single-threaded\n");
            g_model.pool = NULL;
        }
    } else {
        g_model.pool = NULL;
    }

    g_initialized = 1;
    fprintf(stderr, "Model loaded successfully.\n");
    return 0;
}

/*
 * Generate text from a prompt.
 *
 * Each generated token is streamed via postMessage({type:'token', piece, id}).
 * Returns number of tokens generated, or -1 on error.
 */
EMSCRIPTEN_KEEPALIVE
int bitnet_generate(const char *prompt, int32_t max_tokens,
                     float temperature, float top_p, uint32_t seed,
                     float repetition_penalty) {
    if (!g_initialized) {
        fprintf(stderr, "bitnet_generate: model not initialized\n");
        return -1;
    }

    generate_params_t params = {
        .temperature = temperature,
        .top_p = top_p,
        .max_tokens = max_tokens,
        .seed = seed,
        .repetition_penalty = repetition_penalty,
    };

    return generate(&g_model, &g_tok, prompt, &params,
                     wasm_token_callback, NULL);
}

/*
 * Free all model memory.
 */
EMSCRIPTEN_KEEPALIVE
void bitnet_free(void) {
    if (!g_initialized) return;

    /* Destroy thread pool first (workers reference model data) */
    if (g_model.pool) {
        thread_pool_destroy(&g_pool);
        g_model.pool = NULL;
    }

    /* Free TL1 weight index allocations */
    for (int32_t l = 0; l < g_model.config.n_layers; l++) {
        layer_weights_t *lw = &g_model.layers[l];
        free(lw->attn_q.indices);    free(lw->attn_q.indices_col);
        free(lw->attn_k.indices);    free(lw->attn_k.indices_col);
        free(lw->attn_v.indices);    free(lw->attn_v.indices_col);
        free(lw->attn_o.indices);    free(lw->attn_o.indices_col);
        free(lw->ffn_gate.indices);  free(lw->ffn_gate.indices_col);
        free(lw->ffn_up.indices);    free(lw->ffn_up.indices_col);
        free(lw->ffn_down.indices);  free(lw->ffn_down.indices_col);
    }
    model_free(&g_model);
    tokenizer_free(&g_tok);
    g_initialized = 0;
    fprintf(stderr, "Model freed.\n");
}
