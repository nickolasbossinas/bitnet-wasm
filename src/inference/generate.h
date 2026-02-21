#ifndef BITNET_GENERATE_H
#define BITNET_GENERATE_H

#include "model.h"
#include "tokenizer.h"
#include <stdint.h>

/*
 * End-to-end text generation.
 *
 * Wires together: tokenizer -> forward pass -> sampler -> detokenizer.
 * Supports streaming output via callback.
 */

typedef struct {
    float    temperature;  /* 0 = greedy/argmax */
    float    top_p;        /* nucleus sampling threshold */
    int32_t  max_tokens;   /* max new tokens to generate */
    uint32_t seed;         /* RNG seed for sampling */
} generate_params_t;

/*
 * Called for each generated token.
 * piece:    decoded text fragment (may be partial UTF-8 for multi-byte chars)
 * token_id: the sampled token ID
 * user_data: opaque pointer passed through from generate()
 */
typedef void (*token_callback_t)(const char *piece, int32_t token_id,
                                  void *user_data);

/*
 * Generate text from a prompt.
 *
 * model:     initialized model with weights loaded
 * tok:       initialized tokenizer
 * prompt:    input text
 * params:    generation parameters
 * callback:  called for each new token (NULL to suppress output)
 * user_data: passed to callback
 *
 * Returns number of tokens generated, or -1 on error.
 */
int generate(model_t *model, tokenizer_t *tok, const char *prompt,
             const generate_params_t *params, token_callback_t callback,
             void *user_data);

#endif /* BITNET_GENERATE_H */
