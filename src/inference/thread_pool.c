#include "thread_pool.h"
#include "../kernels/tl1.h"
#include "model.h"
#include <string.h>
#include <stdio.h>

/*
 * Helper: evenly split M rows across 'total' workers.
 * Worker 'idx' gets rows [*start, *end).
 * First (M % total) workers get one extra row.
 */
static void split_rows(int32_t M, int32_t total, int32_t idx,
                       int32_t *start, int32_t *end) {
    int32_t base = M / total;
    int32_t rem = M % total;
    *start = idx * base + (idx < rem ? idx : rem);
    *end = *start + base + (idx < rem ? 1 : 0);
}

#if BITNET_THREADED

/* Execute a single work item (called by both workers and main thread) */
static void execute_work(work_item_t *w) {
    switch (w->type) {
    case WORK_GEMV_TL1:
        tl1_gemv_simd_fast_range(w->W, w->lut, w->lut_lo, w->lut_hi,
                                  w->scale, w->out,
                                  w->row_start, w->row_end);
        break;
    case WORK_MATMUL_F32:
        matmul_f32_range(w->out, w->x, w->W_f32, w->K,
                          w->row_start, w->row_end);
        break;
    case WORK_MATMUL_F16F32:
        matmul_f16f32_range(w->out, w->x, w->W_f16, w->K,
                             w->row_start, w->row_end);
        break;
    default:
        break;
    }
}

/* Worker thread entry point */
static void *worker_func(void *arg) {
    worker_ctx_t *ctx = (worker_ctx_t *)arg;
    thread_pool_t *pool = ctx->pool;
    int32_t idx = ctx->pool_idx;

    while (1) {
        pthread_barrier_wait(&pool->barrier_start);
        work_item_t *w = &pool->work[idx];
        if (w->type == WORK_EXIT) break;
        execute_work(w);
        pthread_barrier_wait(&pool->barrier_done);
    }
    return NULL;
}

int thread_pool_init(thread_pool_t *pool, int32_t n_threads) {
    memset(pool, 0, sizeof(*pool));

    if (n_threads < 1) n_threads = 0;
    if (n_threads > BITNET_MAX_THREADS) n_threads = BITNET_MAX_THREADS;
    pool->n_threads = n_threads;

    if (n_threads == 0) return 0;

    /* Barriers include main thread (+1) */
    if (pthread_barrier_init(&pool->barrier_start, NULL, n_threads + 1) != 0) {
        fprintf(stderr, "thread_pool: barrier_start init failed\n");
        return -1;
    }
    if (pthread_barrier_init(&pool->barrier_done, NULL, n_threads + 1) != 0) {
        fprintf(stderr, "thread_pool: barrier_done init failed\n");
        pthread_barrier_destroy(&pool->barrier_start);
        return -1;
    }

    /* Create worker threads */
    for (int32_t i = 0; i < n_threads; i++) {
        pool->ctxs[i].pool_idx = i;
        pool->ctxs[i].pool = pool;
        if (pthread_create(&pool->threads[i], NULL, worker_func, &pool->ctxs[i]) != 0) {
            fprintf(stderr, "thread_pool: failed to create thread %d\n", i);
            /* Clean up already-created threads */
            for (int32_t j = 0; j < i; j++) {
                pool->work[j].type = WORK_EXIT;
            }
            if (i > 0) {
                /* Reinit barriers for fewer threads */
                pthread_barrier_destroy(&pool->barrier_start);
                pthread_barrier_destroy(&pool->barrier_done);
                pthread_barrier_init(&pool->barrier_start, NULL, i + 1);
                pthread_barrier_wait(&pool->barrier_start);
                for (int32_t j = 0; j < i; j++) {
                    pthread_join(pool->threads[j], NULL);
                }
                pthread_barrier_destroy(&pool->barrier_start);
            } else {
                pthread_barrier_destroy(&pool->barrier_start);
                pthread_barrier_destroy(&pool->barrier_done);
            }
            pool->n_threads = 0;
            return -1;
        }
    }

    fprintf(stderr, "Thread pool: %d workers + main = %d total threads\n",
            n_threads, n_threads + 1);
    return 0;
}

void thread_pool_destroy(thread_pool_t *pool) {
    if (pool->n_threads <= 0) return;

    /* Signal all workers to exit */
    for (int32_t i = 0; i < pool->n_threads; i++) {
        pool->work[i].type = WORK_EXIT;
    }

    /* Wake workers (they see WORK_EXIT and break out of loop) */
    pthread_barrier_wait(&pool->barrier_start);

    /* Wait for all workers to finish */
    for (int32_t i = 0; i < pool->n_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    pthread_barrier_destroy(&pool->barrier_start);
    pthread_barrier_destroy(&pool->barrier_done);
    pool->n_threads = 0;
}

void thread_pool_gemv(thread_pool_t *pool,
                      const tl1_weight_t *W,
                      const int16_t *lut,
                      const uint8_t *lut_lo, const uint8_t *lut_hi,
                      float scale, float *out) {
    int32_t n = pool->n_threads;
    int32_t total = n + 1;  /* workers + main */
    int32_t M = W->M;

    /* Fill work items for worker threads (indices 0..n-1) */
    for (int32_t i = 0; i < n; i++) {
        int32_t rs, re;
        split_rows(M, total, i, &rs, &re);
        pool->work[i] = (work_item_t){
            .type = WORK_GEMV_TL1,
            .row_start = rs, .row_end = re,
            .W = W, .lut = lut, .lut_lo = lut_lo, .lut_hi = lut_hi,
            .scale = scale, .out = out
        };
    }

    /* Main thread's slice (index n) */
    int32_t main_start, main_end;
    split_rows(M, total, n, &main_start, &main_end);

    /* Wake workers */
    pthread_barrier_wait(&pool->barrier_start);

    /* Main thread computes its slice while workers run */
    tl1_gemv_simd_fast_range(W, lut, lut_lo, lut_hi, scale, out,
                              main_start, main_end);

    /* Wait for all workers to finish */
    pthread_barrier_wait(&pool->barrier_done);
}

void thread_pool_matmul(thread_pool_t *pool,
                        float *out, const float *x, const float *W,
                        int32_t M, int32_t K) {
    int32_t n = pool->n_threads;
    int32_t total = n + 1;

    /* Fill work items for worker threads */
    for (int32_t i = 0; i < n; i++) {
        int32_t rs, re;
        split_rows(M, total, i, &rs, &re);
        pool->work[i] = (work_item_t){
            .type = WORK_MATMUL_F32,
            .row_start = rs, .row_end = re,
            .x = x, .W_f32 = W, .K = K, .out = out
        };
    }

    int32_t main_start, main_end;
    split_rows(M, total, n, &main_start, &main_end);

    pthread_barrier_wait(&pool->barrier_start);
    matmul_f32_range(out, x, W, K, main_start, main_end);
    pthread_barrier_wait(&pool->barrier_done);
}

void thread_pool_matmul_f16(thread_pool_t *pool,
                             float *out, const float *x, const uint16_t *W,
                             int32_t M, int32_t K) {
    int32_t n = pool->n_threads;
    int32_t total = n + 1;

    for (int32_t i = 0; i < n; i++) {
        int32_t rs, re;
        split_rows(M, total, i, &rs, &re);
        pool->work[i] = (work_item_t){
            .type = WORK_MATMUL_F16F32,
            .row_start = rs, .row_end = re,
            .x = x, .W_f16 = W, .K = K, .out = out
        };
    }

    int32_t main_start, main_end;
    split_rows(M, total, n, &main_start, &main_end);

    pthread_barrier_wait(&pool->barrier_start);
    matmul_f16f32_range(out, x, W, K, main_start, main_end);
    pthread_barrier_wait(&pool->barrier_done);
}

#else /* !BITNET_THREADED */

/* Non-threaded stubs — call range functions directly (single-threaded) */

int thread_pool_init(thread_pool_t *pool, int32_t n_threads) {
    (void)n_threads;
    memset(pool, 0, sizeof(*pool));
    pool->n_threads = 0;
    return 0;
}

void thread_pool_destroy(thread_pool_t *pool) {
    (void)pool;
}

void thread_pool_gemv(thread_pool_t *pool,
                      const tl1_weight_t *W,
                      const int16_t *lut,
                      const uint8_t *lut_lo, const uint8_t *lut_hi,
                      float scale, float *out) {
    (void)pool;
    tl1_gemv_simd_fast_range(W, lut, lut_lo, lut_hi, scale, out, 0, W->M);
}

void thread_pool_matmul(thread_pool_t *pool,
                        float *out, const float *x, const float *W,
                        int32_t M, int32_t K) {
    (void)pool;
    matmul_f32_range(out, x, W, K, 0, M);
}

void thread_pool_matmul_f16(thread_pool_t *pool,
                             float *out, const float *x, const uint16_t *W,
                             int32_t M, int32_t K) {
    (void)pool;
    matmul_f16f32_range(out, x, W, K, 0, M);
}

#endif /* BITNET_THREADED */
