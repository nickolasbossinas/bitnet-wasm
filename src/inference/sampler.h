#ifndef BITNET_SAMPLER_H
#define BITNET_SAMPLER_H

#include <stdint.h>

/*
 * Token Sampling
 *
 * sample_argmax: deterministic, picks highest logit
 * sample_top_p:  nucleus sampling with temperature
 */

/*
 * Return index of the maximum value in logits.
 */
int32_t sample_argmax(const float *logits, int32_t vocab_size);

/*
 * Top-p (nucleus) sampling with temperature.
 * Modifies logits in-place (applies temperature + softmax).
 * Returns sampled token index.
 *
 * rng_state: pointer to xorshift32 state (updated in place)
 */
int32_t sample_top_p(float *logits, int32_t vocab_size,
                     float top_p, float temperature, uint32_t *rng_state);

#endif /* BITNET_SAMPLER_H */
