/*
 * Block sparsity analysis of BitNet ternary weights.
 *
 * For each I2_S tensor, divides weights into blocks of a given size and
 * counts how many non-zero weights each block has. Reports:
 *   - Histogram of block densities (nnz per block)
 *   - How many blocks are below various density thresholds
 *   - Estimated compression from zeroing low-density blocks
 *   - Per-tensor breakdown
 *
 * Usage: ./block_sparsity <model.gguf> [block_size]
 *        Default block_size = 128 (matches I2_S block)
 */

#include "../inference/gguf.h"
#include "../inference/weight_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int32_t block_size;
    int64_t total_blocks;
    int64_t total_weights;
    int64_t total_nnz;
    int64_t *histogram;  /* histogram[i] = number of blocks with exactly i non-zero weights */
} sparsity_stats_t;

static void init_stats(sparsity_stats_t *s, int32_t block_size) {
    s->block_size = block_size;
    s->total_blocks = 0;
    s->total_weights = 0;
    s->total_nnz = 0;
    s->histogram = (int64_t *)calloc(block_size + 1, sizeof(int64_t));
}

static void accumulate_tensor(sparsity_stats_t *s, const int8_t *weights,
                               int64_t n_elements) {
    int32_t bs = s->block_size;
    int64_t n_blocks = n_elements / bs;
    int64_t tail = n_elements % bs;

    for (int64_t b = 0; b < n_blocks; b++) {
        const int8_t *blk = &weights[b * bs];
        int32_t nnz = 0;
        for (int32_t i = 0; i < bs; i++) {
            if (blk[i] != 0) nnz++;
        }
        s->histogram[nnz]++;
        s->total_blocks++;
        s->total_nnz += nnz;
    }

    /* Handle tail block (partial) — count but don't include in histogram */
    if (tail > 0) {
        const int8_t *blk = &weights[n_blocks * bs];
        int32_t nnz = 0;
        for (int64_t i = 0; i < tail; i++) {
            if (blk[i] != 0) nnz++;
        }
        s->total_nnz += nnz;
    }

    s->total_weights += n_elements;
}

static void print_histogram(const sparsity_stats_t *s) {
    int32_t bs = s->block_size;

    printf("\n--- Block Density Histogram (block_size=%d) ---\n\n", bs);
    printf("NNZ Range     Blocks       %%Blocks    CumulBlocks  Cumul%%\n");
    printf("---------     ------       -------    -----------  ------\n");

    /* Print in buckets of block_size/16 */
    int32_t bucket_size = bs / 16;
    if (bucket_size < 1) bucket_size = 1;

    int64_t cumul = 0;
    for (int32_t lo = 0; lo <= bs; lo += bucket_size) {
        int32_t hi = lo + bucket_size - 1;
        if (hi > bs) hi = bs;

        int64_t count = 0;
        for (int32_t i = lo; i <= hi && i <= bs; i++) {
            count += s->histogram[i];
        }
        cumul += count;

        if (count > 0) {
            printf("%3d - %3d     %8lld     %6.2f%%    %11lld  %6.2f%%\n",
                   lo, hi,
                   (long long)count,
                   100.0 * count / s->total_blocks,
                   (long long)cumul,
                   100.0 * cumul / s->total_blocks);
        }
    }
}

static void print_pruning_analysis(const sparsity_stats_t *s) {
    int32_t bs = s->block_size;
    double total_i2s_mb = s->total_weights * 2.0 / 8.0 / 1024.0 / 1024.0;
    double baseline_zero_pct = 100.0 * (s->total_weights - s->total_nnz) / s->total_weights;

    printf("\n--- Pruning Analysis ---\n\n");
    printf("Total weights:      %lld\n", (long long)s->total_weights);
    printf("Total non-zero:     %lld (%.2f%%)\n",
           (long long)s->total_nnz, 100.0 * s->total_nnz / s->total_weights);
    printf("Total zero:         %lld (%.2f%%)\n",
           (long long)(s->total_weights - s->total_nnz), baseline_zero_pct);
    printf("Total blocks:       %lld (block_size=%d)\n",
           (long long)s->total_blocks, bs);
    printf("I2_S size:          %.1f MB\n\n", total_i2s_mb);

    printf("Threshold   ZeroBlocks   %%Blocks   WeightsZeroed   NewZero%%   ");
    printf("FileSave   SaveMB   SpeedupEst\n");
    printf("---------   ----------   -------   -------------   --------   ");
    printf("--------   ------   ----------\n");

    /* For each density threshold: if block has <= threshold nnz, zero it */
    int thresholds[] = {0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, -1};

    for (int ti = 0; thresholds[ti] >= 0; ti++) {
        int thresh = thresholds[ti];
        if (thresh > bs) break;

        int64_t blocks_zeroed = 0;
        int64_t weights_zeroed = 0;  /* additional non-zero weights forced to zero */

        for (int32_t nnz = 0; nnz <= thresh && nnz <= bs; nnz++) {
            blocks_zeroed += s->histogram[nnz];
            weights_zeroed += s->histogram[nnz] * nnz;  /* force these nnz weights to zero */
        }

        int64_t new_total_zero = (s->total_weights - s->total_nnz) + weights_zeroed;
        double new_zero_pct = 100.0 * new_total_zero / s->total_weights;

        /* File savings: zeroed blocks don't need to be stored */
        double blocks_saved_pct = 100.0 * blocks_zeroed / s->total_blocks;
        double saved_mb = blocks_zeroed * bs * 2.0 / 8.0 / 1024.0 / 1024.0;

        /* Speedup estimate: proportional to blocks skipped */
        double speedup = 1.0 / (1.0 - (double)blocks_zeroed / s->total_blocks);

        printf("nnz <= %3d   %10lld   %6.2f%%   %13lld   %7.2f%%   "
               "%6.2f%%   %5.1f   %.2fx\n",
               thresh,
               (long long)blocks_zeroed,
               blocks_saved_pct,
               (long long)weights_zeroed,
               new_zero_pct,
               blocks_saved_pct,
               saved_mb,
               speedup);
    }
}

static void print_per_tensor_summary(const gguf_context_t *gguf,
                                      const uint8_t *file_data,
                                      int32_t block_size) {
    printf("\n--- Per-Tensor Block Sparsity (blocks with 0 nnz / blocks with <=16 nnz) ---\n\n");
    printf("%-40s %10s %8s %8s %8s\n",
           "Tensor", "Blocks", "0-nnz%%", "<=16nnz%%", "MeanNNZ");
    printf("%-40s %10s %8s %8s %8s\n",
           "------", "------", "------", "--------", "-------");

    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        gguf_tensor_info_t *t = &gguf->tensors[i];
        if (t->type != GGML_TYPE_I2_S) continue;

        int64_t n_elements = 1;
        for (int d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];

        int8_t *weights = (int8_t *)malloc(n_elements);
        if (!weights) continue;

        const uint8_t *src = (const uint8_t *)gguf_tensor_data(gguf, t, file_data);
        i2s_decode(src, weights, n_elements);

        int64_t n_blocks = n_elements / block_size;
        int64_t zero_blocks = 0;
        int64_t low_blocks = 0;  /* <=16 nnz */
        int64_t total_nnz = 0;

        for (int64_t b = 0; b < n_blocks; b++) {
            const int8_t *blk = &weights[b * block_size];
            int32_t nnz = 0;
            for (int32_t j = 0; j < block_size; j++) {
                if (blk[j] != 0) nnz++;
            }
            if (nnz == 0) zero_blocks++;
            if (nnz <= 16) low_blocks++;
            total_nnz += nnz;
        }

        double mean_nnz = n_blocks > 0 ? (double)total_nnz / n_blocks : 0;

        printf("%-40s %10lld %7.2f%% %7.2f%% %7.1f\n",
               t->name,
               (long long)n_blocks,
               100.0 * zero_blocks / n_blocks,
               100.0 * low_blocks / n_blocks,
               mean_nnz);

        free(weights);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [block_size]\n", argv[0]);
        return 1;
    }

    int32_t block_size = 128;
    if (argc >= 3) block_size = atoi(argv[2]);
    if (block_size < 1 || block_size > 1024) {
        fprintf(stderr, "Invalid block_size %d\n", block_size);
        return 1;
    }

    /* Read GGUF file */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(file_size);
    if (!data) { fprintf(stderr, "OOM\n"); fclose(f); return 1; }
    fread(data, 1, file_size, f);
    fclose(f);

    gguf_context_t gguf;
    if (gguf_parse(&gguf, data, file_size) != 0) {
        fprintf(stderr, "GGUF parse failed\n");
        free(data);
        return 1;
    }

    printf("=== Block Sparsity Analysis ===\n");
    printf("Block size: %d\n", block_size);

    /* Global stats */
    sparsity_stats_t stats;
    init_stats(&stats, block_size);

    for (uint64_t i = 0; i < gguf.n_tensors; i++) {
        gguf_tensor_info_t *t = &gguf.tensors[i];
        if (t->type != GGML_TYPE_I2_S) continue;

        int64_t n_elements = 1;
        for (int d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];

        int8_t *weights = (int8_t *)malloc(n_elements);
        if (!weights) {
            fprintf(stderr, "OOM decoding %s\n", t->name);
            continue;
        }

        const uint8_t *src = (const uint8_t *)gguf_tensor_data(&gguf, t, data);
        i2s_decode(src, weights, n_elements);
        accumulate_tensor(&stats, weights, n_elements);

        free(weights);
    }

    print_histogram(&stats);
    print_pruning_analysis(&stats);
    print_per_tensor_summary(&gguf, data, block_size);

    free(stats.histogram);
    gguf_free(&gguf);
    free(data);
    return 0;
}
