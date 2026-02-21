#include "../kernels/gemv.h"
#include "../kernels/i2s.h"
#include "../kernels/tl1.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

/*
 * BitNet WASM SIMD Kernel Benchmark
 *
 * Tests all kernel variants (I2_S scalar/SIMD, TL1 scalar/SIMD)
 * against the same random ternary weight matrix and activation vector.
 *
 * Reports:
 *   - Time per GEMV (ms)
 *   - Throughput (GOPS — giga operations per second)
 *   - Speedup vs scalar baseline
 *   - Correctness (max error vs scalar reference)
 */

#define NUM_WARMUP   5
#define NUM_ITERS   50

/* Simple xorshift32 PRNG (deterministic, no rand() dependency) */
static uint32_t rng_state = 42;

static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

/* Generate random ternary weight: {-1, 0, 1} */
static int8_t rand_ternary(void) {
    return (int8_t)(xorshift32() % 3) - 1;
}

/* Generate random float activation in [-1, 1] */
static float rand_activation(void) {
    return ((float)(xorshift32() % 10000) / 5000.0f) - 1.0f;
}

/* Max absolute error between two float arrays */
static float max_error(const float *a, const float *b, int32_t len) {
    float max_err = 0.0f;
    for (int32_t i = 0; i < len; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

typedef struct {
    const char *name;
    double avg_ms;
    double gops;
    double speedup;
    float  max_err;
} bench_result_t;

static void print_header(int32_t M, int32_t K) {
    printf("===========================================\n");
    printf(" BitNet WASM SIMD Kernel Benchmark\n");
    printf("===========================================\n");
    printf(" Matrix: %d x %d (%.1f KB packed I2S, %.1f KB packed TL1)\n",
           M, K,
           (float)(M * ((K + 3) / 4)) / 1024.0f,
           (float)(M * ((K / 2 + 1) / 2)) / 1024.0f);
    printf(" Operations per GEMV: %lld\n", (long long)M * K * 2);
    printf(" Warmup: %d, Iterations: %d\n", NUM_WARMUP, NUM_ITERS);
    printf("-------------------------------------------\n");
}

static void print_result(bench_result_t *r) {
    printf(" %-20s  %8.3f ms  %6.2f GOPS  %5.2fx  err=%.2e\n",
           r->name, r->avg_ms, r->gops, r->speedup, r->max_err);
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int run_benchmark(int32_t M, int32_t K) {
    printf("\n");
    print_header(M, K);

    /* Allocate raw ternary weights */
    int8_t *raw_weights = (int8_t *)malloc(M * K * sizeof(int8_t));
    for (int32_t i = 0; i < M * K; i++) {
        raw_weights[i] = rand_ternary();
    }

    /* Allocate activations */
    float *activations = (float *)malloc(K * sizeof(float));
    for (int32_t i = 0; i < K; i++) {
        activations[i] = rand_activation();
    }

    /* Pack weights for I2_S */
    int32_t i2s_packed_K = (K + 3) / 4;
    uint8_t *i2s_packed = (uint8_t *)calloc(M * i2s_packed_K, sizeof(uint8_t));
    i2s_pack_weights(raw_weights, i2s_packed, M, K);

    i2s_weight_t i2s_w = {
        .data  = i2s_packed,
        .M     = M,
        .K     = K,
        .scale = 1.0f   /* simplified: no per-tensor scale for benchmark */
    };

    /* Pack weights for TL1 */
    int32_t tl1_pairs = K / 2;
    int32_t tl1_bytes = (tl1_pairs + 1) / 2;
    uint8_t *tl1_packed = (uint8_t *)calloc(M * tl1_bytes, sizeof(uint8_t));
    tl1_pack_weights(raw_weights, tl1_packed, M, K);

    tl1_weight_t tl1_w = {
        .indices = tl1_packed,
        .M       = M,
        .K       = K,
        .scale   = 1.0f
    };

    /* Output buffers */
    float *out_ref  = (float *)calloc(M, sizeof(float));
    float *out_test = (float *)calloc(M, sizeof(float));

    /* Benchmark each kernel */
    bench_result_t results[KERNEL_COUNT];
    double baseline_ms = 0.0;

    for (int32_t k = 0; k < KERNEL_COUNT; k++) {
        kernel_type_t kt = (kernel_type_t)k;
        const void *w = (kt <= KERNEL_I2S_SIMD)
                        ? (const void *)&i2s_w
                        : (const void *)&tl1_w;

        /* Warmup */
        for (int32_t iter = 0; iter < NUM_WARMUP; iter++) {
            gemv_run(kt, w, activations, out_test, M, K);
        }

        /* Timed iterations */
        double total_ms = 0.0;
        for (int32_t iter = 0; iter < NUM_ITERS; iter++) {
            double ms = gemv_run(kt, w, activations, out_test, M, K);
            total_ms += ms;
        }

        double avg_ms = total_ms / NUM_ITERS;
        double ops = (double)M * K * 2;  /* multiply + add per element */
        double gops = (ops / (avg_ms / 1000.0)) / 1e9;

        /* Reference: first kernel (I2_S scalar) */
        if (k == 0) {
            baseline_ms = avg_ms;
            memcpy(out_ref, out_test, M * sizeof(float));
        }

        results[k].name    = kernel_name(kt);
        results[k].avg_ms  = avg_ms;
        results[k].gops    = gops;
        results[k].speedup = baseline_ms / avg_ms;
        results[k].max_err = max_error(out_ref, out_test, M);

        print_result(&results[k]);
    }

    printf("-------------------------------------------\n");
    printf(" Done.\n\n");

    /* Cleanup */
    free(raw_weights);
    free(activations);
    free(i2s_packed);
    free(tl1_packed);
    free(out_ref);
    free(out_test);

    return 0;
}

int main(void) {
    /* Typical BitNet 2B layer dimensions */
    printf(">>> Small test (256 x 256)\n");
    run_benchmark(256, 256);

    printf(">>> Medium test (2048 x 2048)\n");
    run_benchmark(2048, 2048);

    printf(">>> Large test (4096 x 2048) — typical FFN layer\n");
    run_benchmark(4096, 2048);

    return 0;
}
