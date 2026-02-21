#include "generate.h"
#include "sampler.h"
#include <stdio.h>
#include <stdlib.h>
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

    /* Token history for repetition penalty and loop detection */
    int32_t *history = (int32_t *)malloc((n_prompt + max_new) * sizeof(int32_t));
    if (!history) {
        fprintf(stderr, "generate: failed to allocate token history\n");
        return -1;
    }
    memcpy(history, prompt_tokens, n_prompt * sizeof(int32_t));
    int32_t history_len = n_prompt;

    float rep_penalty = params->repetition_penalty;
    int32_t consecutive_newlines = 0;

    for (int32_t i = 0; i < max_new; i++) {
        /* Apply repetition penalty to logits for previously seen tokens */
        if (rep_penalty != 1.0f) {
            for (int32_t h = 0; h < history_len; h++) {
                int32_t tid = history[h];
                if (logits[tid] > 0.0f)
                    logits[tid] /= rep_penalty;
                else
                    logits[tid] *= rep_penalty;
            }
        }

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
        history[history_len++] = next_token;

        /* Stop on repeated 8-gram (model is looping) */
        if (history_len >= 16) {
            int32_t ngram = 8;
            int32_t *recent = &history[history_len - ngram];
            int32_t *prior  = &history[history_len - 2 * ngram];
            if (memcmp(recent, prior, ngram * sizeof(int32_t)) == 0) {
                fprintf(stderr, "[generate] stopped: repeated %d-gram detected\n", ngram);
                break;
            }
        }

        /* Decode and emit token (GPT-2 byte decode → raw bytes) */
        {
            char piece_buf[256];
            int32_t len = tokenizer_decode(tok, &next_token, 1,
                                            piece_buf, sizeof(piece_buf));
            if (len > 0) {
                if (callback) {
                    callback(piece_buf, next_token, user_data);
                }

                /* Track consecutive newlines for \n\n stop */
                for (int32_t c = 0; c < len; c++) {
                    if (piece_buf[c] == '\n')
                        consecutive_newlines++;
                    else
                        consecutive_newlines = 0;
                }
                if (consecutive_newlines >= 2) {
                    break;
                }
            }
        }

        /* Forward pass for next position */
        int32_t pos = n_prompt + i;
        logits = forward(model, next_token, pos);
        if (!logits) {
            fprintf(stderr, "generate: forward failed at decode pos %d\n", pos);
            free(history);
            return -1;
        }
    }
    double t_decode_end = get_time_ms();
    free(history);

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
