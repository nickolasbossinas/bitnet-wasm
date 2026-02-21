#include "sampler.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int32_t sample_argmax(const float *logits, int32_t vocab_size) {
    int32_t best = 0;
    float best_val = logits[0];
    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* Simple xorshift32 PRNG */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Random float in [0, 1) */
static float rand_f32(uint32_t *state) {
    return (float)(xorshift32(state) >> 8) / 16777216.0f;
}

/* Comparison for qsort: descending by probability */
typedef struct {
    float prob;
    int32_t index;
} prob_index_t;

static int prob_index_cmp(const void *a, const void *b) {
    const prob_index_t *pa = (const prob_index_t *)a;
    const prob_index_t *pb = (const prob_index_t *)b;
    if (pb->prob > pa->prob) return 1;
    if (pb->prob < pa->prob) return -1;
    return 0;
}

int32_t sample_top_p(float *logits, int32_t vocab_size,
                     float top_p, float temperature, uint32_t *rng_state) {
    /* Apply temperature */
    if (temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        for (int32_t i = 0; i < vocab_size; i++) {
            logits[i] *= inv_temp;
        }
    }

    /* Softmax */
    float max_val = logits[0];
    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int32_t i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int32_t i = 0; i < vocab_size; i++) {
        logits[i] *= inv_sum;
    }

    /* Build sorted probability-index pairs */
    prob_index_t *sorted = (prob_index_t *)malloc(vocab_size * sizeof(prob_index_t));
    for (int32_t i = 0; i < vocab_size; i++) {
        sorted[i].prob = logits[i];
        sorted[i].index = i;
    }
    qsort(sorted, vocab_size, sizeof(prob_index_t), prob_index_cmp);

    /* Find top-p cutoff */
    float cumulative = 0.0f;
    int32_t cutoff = vocab_size;
    for (int32_t i = 0; i < vocab_size; i++) {
        cumulative += sorted[i].prob;
        if (cumulative > top_p) {
            cutoff = i + 1;
            break;
        }
    }

    /* Sample from the top-p set */
    float r = rand_f32(rng_state) * cumulative;
    float running = 0.0f;
    int32_t result = sorted[0].index;
    for (int32_t i = 0; i < cutoff; i++) {
        running += sorted[i].prob;
        if (running >= r) {
            result = sorted[i].index;
            break;
        }
    }

    free(sorted);
    return result;
}
