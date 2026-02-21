#ifndef BITNET_THREAD_POOL_H
#define BITNET_THREAD_POOL_H

/*
 * Thread pool for parallel GEMV and matmul operations.
 *
 * Uses Emscripten pthreads (SharedArrayBuffer-backed Web Workers) on WASM.
 * Falls back to single-threaded stubs on native builds.
 *
 * Design: barrier-based synchronization. Worker threads sleep on
 * barrier_start, wake to execute their work slice, then wait on
 * barrier_done. This gives very low overhead per dispatch (~2 barrier
 * waits = microseconds), which is critical since we dispatch 211 times
 * per token (7 GEMV × 30 layers + 1 logits matmul).
 */

#include "../kernels/types.h"
#include <stdint.h>

#if defined(__EMSCRIPTEN__) && defined(__EMSCRIPTEN_PTHREADS__)
#define BITNET_THREADED 1
#include <pthread.h>
#else
#define BITNET_THREADED 0
#endif

#define BITNET_MAX_THREADS 8

typedef enum {
    WORK_NONE = 0,
    WORK_GEMV_TL1,
    WORK_MATMUL_F32,
    WORK_EXIT
} work_type_t;

typedef struct {
    work_type_t type;
    int32_t row_start;
    int32_t row_end;

    /* GEMV TL1 params */
    const tl1_weight_t *W;
    const int16_t *lut;
    const uint8_t *lut_lo;
    const uint8_t *lut_hi;
    float scale;

    /* Matmul F32 params */
    const float *x;
    const float *W_f32;
    int32_t K;

    /* Output (shared by both GEMV and matmul) */
    float *out;
} work_item_t;

typedef struct {
    int32_t pool_idx;           /* thread index in pool */
    struct thread_pool_s *pool; /* back-pointer */
} worker_ctx_t;

typedef struct thread_pool_s {
    int32_t n_threads;  /* number of worker threads (not counting main) */
#if BITNET_THREADED
    pthread_t threads[BITNET_MAX_THREADS];
    worker_ctx_t ctxs[BITNET_MAX_THREADS];
    work_item_t work[BITNET_MAX_THREADS + 1]; /* +1 for main thread's slice */
    pthread_barrier_t barrier_start;
    pthread_barrier_t barrier_done;
#else
    work_item_t work[1]; /* unused, for struct validity */
#endif
} thread_pool_t;

/*
 * Create thread pool with n worker threads.
 * Returns 0 on success, -1 on failure.
 * n_threads=0 creates the struct but no workers (single-threaded).
 */
int thread_pool_init(thread_pool_t *pool, int32_t n_threads);

/*
 * Destroy thread pool. Signals workers to exit and joins them.
 */
void thread_pool_destroy(thread_pool_t *pool);

/*
 * Dispatch parallel TL1 GEMV: split W->M output rows across threads.
 * Main thread participates in computation (not idle during dispatch).
 * Requires pool->n_threads > 0.
 */
void thread_pool_gemv(thread_pool_t *pool,
                      const tl1_weight_t *W,
                      const int16_t *lut,
                      const uint8_t *lut_lo, const uint8_t *lut_hi,
                      float scale, float *out);

/*
 * Dispatch parallel F32 matmul: split M output rows across threads.
 * Main thread participates in computation.
 * Requires pool->n_threads > 0.
 */
void thread_pool_matmul(thread_pool_t *pool,
                        float *out, const float *x, const float *W,
                        int32_t M, int32_t K);

#endif /* BITNET_THREAD_POOL_H */
