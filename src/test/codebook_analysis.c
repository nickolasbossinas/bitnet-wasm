/*
 * Codebook compression analysis for BitNet ternary weights.
 *
 * Analyzes the feasibility of dictionary compression on raw I2_S bytes.
 * Tests 2-byte sub-blocks (8 weights each) — the sweet spot where the
 * pattern space (65536) is small enough for a direct frequency array.
 *
 * Also tests 1-byte sub-blocks (4 weights, 256 patterns) as baseline.
 *
 * Usage: ./codebook_analysis <model.gguf>
 */

#include "../inference/gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int cmp_desc(const void *a, const void *b) {
    int64_t va = *(const int64_t *)a;
    int64_t vb = *(const int64_t *)b;
    return (vb > va) - (vb < va);
}

/* Collect all I2_S raw bytes across tensors */
static void get_i2s_info(const gguf_context_t *gguf, const uint8_t *file_data,
                          const uint8_t ***tensor_ptrs, int64_t **tensor_sizes,
                          int *n_tensors_out) {
    int count = 0;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (gguf->tensors[i].type == GGML_TYPE_I2_S) count++;
    }

    *tensor_ptrs = (const uint8_t **)malloc(count * sizeof(uint8_t *));
    *tensor_sizes = (int64_t *)malloc(count * sizeof(int64_t));
    *n_tensors_out = count;

    int idx = 0;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        gguf_tensor_info_t *t = &gguf->tensors[i];
        if (t->type != GGML_TYPE_I2_S) continue;

        int64_t n_el = 1;
        for (int d = 0; d < t->n_dims; d++) n_el *= t->dims[d];

        (*tensor_ptrs)[idx] = (const uint8_t *)gguf_tensor_data(gguf, t, file_data);
        (*tensor_sizes)[idx] = (n_el + 3) / 4;  /* packed bytes */
        idx++;
    }
}

static void analyze_1byte(const uint8_t **ptrs, const int64_t *sizes, int n_tensors) {
    printf("\n=== Sub-block: 1 byte (4 weights) ===\n\n");

    int64_t counts[256] = {0};
    int64_t total = 0;

    for (int t = 0; t < n_tensors; t++) {
        for (int64_t j = 0; j < sizes[t]; j++) {
            counts[ptrs[t][j]]++;
        }
        total += sizes[t];
    }

    int unique = 0;
    for (int j = 0; j < 256; j++) {
        if (counts[j] > 0) unique++;
    }

    int64_t sorted[256];
    memcpy(sorted, counts, sizeof(sorted));
    qsort(sorted, 256, sizeof(int64_t), cmp_desc);

    printf("Total sub-blocks:  %lld\n", (long long)total);
    printf("Unique patterns:   %d / 256\n", unique);
    printf("(Only 81 ternary combos exist for 4 weights)\n\n");

    printf("Top 10:\n");
    for (int k = 0; k < 10; k++) {
        printf("  #%d: %lld (%.2f%%)\n", k + 1, (long long)sorted[k],
               100.0 * sorted[k] / total);
    }

    /* Entropy */
    double H = 0;
    for (int k = 0; k < 256; k++) {
        if (counts[k] > 0) {
            double p = (double)counts[k] / total;
            H -= p * log2(p);
        }
    }
    printf("\nEntropy: %.4f bits/byte = %.4f bits/weight\n", H, H / 4.0);
    printf("Minimum size: %.1f MB (vs %.1f MB raw)\n",
           H * total / 8.0 / (1024.0 * 1024.0),
           (double)total / (1024.0 * 1024.0));
}

static void analyze_2byte(const uint8_t **ptrs, const int64_t *sizes, int n_tensors) {
    printf("\n=== Sub-block: 2 bytes (8 weights) ===\n\n");

    int64_t *counts = (int64_t *)calloc(65536, sizeof(int64_t));
    int64_t total = 0;

    for (int t = 0; t < n_tensors; t++) {
        int64_t n_sub = sizes[t] / 2;
        const uint8_t *src = ptrs[t];
        for (int64_t s = 0; s < n_sub; s++) {
            uint16_t key;
            memcpy(&key, &src[s * 2], 2);
            counts[key]++;
        }
        total += n_sub;
    }

    int64_t unique = 0;
    for (int j = 0; j < 65536; j++) {
        if (counts[j] > 0) unique++;
    }

    int64_t *sorted = (int64_t *)malloc(unique * sizeof(int64_t));
    int64_t si = 0;
    for (int j = 0; j < 65536; j++) {
        if (counts[j] > 0) sorted[si++] = counts[j];
    }
    qsort(sorted, unique, sizeof(int64_t), cmp_desc);

    printf("Total sub-blocks:  %lld\n", (long long)total);
    printf("Unique patterns:   %lld / 65536 possible\n", (long long)unique);
    printf("(Ternary combos for 8 weights: 3^8 = 6561)\n");
    printf("Repetition ratio:  %.0fx\n\n", (double)total / unique);

    printf("Top 20:\n");
    for (int k = 0; k < 20 && k < unique; k++) {
        printf("  #%d: %lld (%.3f%%)\n", k + 1, (long long)sorted[k],
               100.0 * sorted[k] / total);
    }

    /* Coverage at various codebook sizes */
    double raw_mb = (double)total * 2 / (1024.0 * 1024.0);
    printf("\nCodebook coverage (raw = %.1f MB):\n", raw_mb);
    printf("  CB Size   IdxBits   Coverage   Compressed    Ratio    Bits/wt\n");

    int cb_sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, -1};
    for (int ci = 0; cb_sizes[ci] > 0; ci++) {
        int64_t cb = cb_sizes[ci];
        if (cb > unique) cb = unique;

        int64_t covered = 0;
        for (int64_t k = 0; k < cb; k++) covered += sorted[k];
        double cov = 100.0 * covered / total;

        int idx_bits = 1;
        while ((1LL << idx_bits) < cb) idx_bits++;

        int64_t uncov = total - covered;
        /* Each sub-block: 1 flag bit + (idx_bits if in CB, or 16 raw bits if not) */
        double bits = total + covered * (double)idx_bits + uncov * 16.0;
        double cb_overhead = (double)cb * 2;  /* codebook table in bytes */
        double comp_mb = bits / 8.0 / (1024.0 * 1024.0) + cb_overhead / (1024.0 * 1024.0);
        double bpw = bits / (total * 8.0);

        printf("  %6d    %5d     %5.1f%%    %7.1f MB    %.2fx    %.4f\n",
               cb_sizes[ci], idx_bits, cov, comp_mb, raw_mb / comp_mb, bpw);
        if (cb == unique) break;
    }

    /* What if we only use codebook (no escape — lossy for rare patterns)? */
    printf("\nPure codebook (no escape, assign nearest):\n");
    printf("  CB Size   IdxBits   Size       Ratio    Bits/wt\n");
    int pure_cb[] = {256, 4096, 65536, -1};
    for (int ci = 0; pure_cb[ci] > 0; ci++) {
        int cb = pure_cb[ci];
        int idx_bits = 1;
        while ((1 << idx_bits) < cb) idx_bits++;
        double bits = total * (double)idx_bits;
        double sz_mb = bits / 8.0 / (1024.0 * 1024.0) + (double)cb * 2 / (1024.0 * 1024.0);
        double bpw = (double)idx_bits / 8.0;
        printf("  %6d    %5d     %5.1f MB    %.2fx    %.4f\n",
               cb, idx_bits, sz_mb, raw_mb / sz_mb, bpw);
    }

    /* Entropy */
    double H = 0;
    for (int64_t k = 0; k < unique; k++) {
        double p = (double)sorted[k] / total;
        H -= p * log2(p);
    }
    printf("\nEntropy: %.4f bits/sub-block = %.4f bits/weight\n", H, H / 8.0);
    printf("Entropy floor: %.1f MB\n", H * total / 8.0 / (1024.0 * 1024.0));

    free(sorted);
    free(counts);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

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

    printf("=== Codebook Compression Analysis ===\n");
    printf("Model: %s (%.1f MB)\n", argv[1], file_size / (1024.0 * 1024.0));

    const uint8_t **ptrs;
    int64_t *sizes;
    int n_tensors;
    get_i2s_info(&gguf, data, &ptrs, &sizes, &n_tensors);
    printf("I2_S tensors: %d\n", n_tensors);

    analyze_1byte(ptrs, sizes, n_tensors);
    analyze_2byte(ptrs, sizes, n_tensors);

    free(ptrs);
    free(sizes);
    gguf_free(&gguf);
    free(data);
    return 0;
}
