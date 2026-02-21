#include "generate.h"
#include "sampler.h"
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

int generate(model_t *model, tokenizer_t *tok, const char *prompt,
             const generate_params_t *params, token_callback_t callback,
             void *user_data) {
    /* Encode prompt */
    int32_t prompt_tokens[4096];
    int32_t n_prompt = tokenizer_encode(tok, prompt, prompt_tokens, 4096, 1);
    if (n_prompt <= 0) {
        fprintf(stderr, "generate: tokenizer_encode failed\n");
        return -1;
    }

    int32_t max_seq = model->config.max_seq_len;
    if (n_prompt >= max_seq) {
        fprintf(stderr, "generate: prompt too long (%d tokens, max %d)\n",
                n_prompt, max_seq);
        return -1;
    }

    int32_t max_new = params->max_tokens;
    if (n_prompt + max_new > max_seq) {
        max_new = max_seq - n_prompt;
    }

    uint32_t rng_state = params->seed;

    fprintf(stderr, "[generate] prompt: %d tokens, generating up to %d new tokens\n",
            n_prompt, max_new);

    /* Prefill: process all prompt tokens to build KV cache */
    double t_prefill_start = get_time_ms();
    float *logits = NULL;
    for (int32_t i = 0; i < n_prompt; i++) {
        logits = forward(model, prompt_tokens[i], i);
        if (!logits) {
            fprintf(stderr, "generate: forward failed at prompt pos %d\n", i);
            return -1;
        }
    }
    double t_prefill_end = get_time_ms();

    /* Decode: generate new tokens one at a time */
    double t_decode_start = get_time_ms();
    int32_t n_generated = 0;

    for (int32_t i = 0; i < max_new; i++) {
        /* Sample next token from logits */
        int32_t next_token;
        if (params->temperature == 0.0f) {
            next_token = sample_argmax(logits, model->config.vocab_size);
        } else {
            next_token = sample_top_p(logits, model->config.vocab_size,
                                       params->top_p, params->temperature,
                                       &rng_state);
        }

        /* Stop on EOS */
        if (next_token == tok->eos_id) {
            break;
        }

        n_generated++;

        /* Decode and emit token */
        if (callback) {
            const char *piece = tokenizer_decode_token(tok, next_token);
            if (piece) {
                callback(piece, next_token, user_data);
            }
        }

        /* Forward pass for next position */
        int32_t pos = n_prompt + i;
        logits = forward(model, next_token, pos);
        if (!logits) {
            fprintf(stderr, "generate: forward failed at decode pos %d\n", pos);
            return -1;
        }
    }
    double t_decode_end = get_time_ms();

    /* Print timing stats */
    double prefill_ms = t_prefill_end - t_prefill_start;
    double decode_ms = t_decode_end - t_decode_start;
    double prefill_tps = (prefill_ms > 0) ? (n_prompt / (prefill_ms / 1000.0)) : 0;
    double decode_tps = (decode_ms > 0 && n_generated > 0)
                         ? (n_generated / (decode_ms / 1000.0)) : 0;

    fprintf(stderr, "\n[generate] prefill: %d tokens, %.1f ms (%.1f tok/s)\n",
            n_prompt, prefill_ms, prefill_tps);
    fprintf(stderr, "[generate] decode:  %d tokens, %.1f ms (%.1f tok/s)\n",
            n_generated, decode_ms, decode_tps);
    fprintf(stderr, "[generate] total:   %.1f ms\n", prefill_ms + decode_ms);

    return n_generated;
}
