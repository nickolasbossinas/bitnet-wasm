/*
 * Entropy analysis of BitNet ternary weights.
 *
 * Reads the GGUF model file, decodes all I2_S tensors, and reports:
 *   - Per-tensor {-1, 0, +1} distribution
 *   - Per-tensor entropy (bits/weight)
 *   - Global aggregate statistics
 *   - Theoretical minimum file size vs actual I2_S size
 *
 * Build: part of native CMake build (see CMakeLists.txt)
 * Usage: ./entropy_analysis <model.gguf>
 */

#include "../inference/gguf.h"
#include "../inference/weight_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    /* Read GGUF file into memory */
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *data = (uint8_t *)malloc(file_size);
    if (!data) {
        fprintf(stderr, "Cannot allocate %ld bytes\n", file_size);
        fclose(f);
        return 1;
    }
    fread(data, 1, file_size, f);
    fclose(f);

    /* Parse GGUF */
    gguf_context_t gguf;
    if (gguf_parse(&gguf, data, file_size) != 0) {
        fprintf(stderr, "GGUF parse failed\n");
        free(data);
        return 1;
    }

    printf("=== BitNet Ternary Weight Entropy Analysis ===\n\n");
    printf("File: %s (%.1f MB)\n", argv[1], file_size / (1024.0 * 1024.0));
    printf("Tensors: %llu\n\n", (unsigned long long)gguf.n_tensors);

    /* Global counters */
    int64_t global_neg = 0, global_zero = 0, global_pos = 0;
    int64_t total_i2s_packed_bytes = 0;
    int n_i2s_tensors = 0;

    printf("%-40s %12s %8s %8s %8s %10s\n",
           "Tensor", "Elements", "%-1", "%0", "%+1", "H (bits)");
    printf("%-40s %12s %8s %8s %8s %10s\n",
           "------", "--------", "---", "--", "---", "--------");

    for (uint64_t i = 0; i < gguf.n_tensors; i++) {
        gguf_tensor_info_t *t = &gguf.tensors[i];
        if (t->type != GGML_TYPE_I2_S) continue;

        n_i2s_tensors++;

        int64_t n_elements = 1;
        for (int d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];

        /* Decode I2_S */
        int8_t *weights = (int8_t *)malloc(n_elements);
        if (!weights) {
            fprintf(stderr, "OOM decoding %s\n", t->name);
            continue;
        }

        const uint8_t *src = (const uint8_t *)gguf_tensor_data(&gguf, t, data);
        i2s_decode(src, weights, n_elements);

        /* Count values */
        int64_t cnt_neg = 0, cnt_zero = 0, cnt_pos = 0;
        for (int64_t j = 0; j < n_elements; j++) {
            if (weights[j] == -1) cnt_neg++;
            else if (weights[j] == 0) cnt_zero++;
            else if (weights[j] == 1) cnt_pos++;
        }

        /* Compute entropy */
        double p_neg  = (double)cnt_neg  / n_elements;
        double p_zero = (double)cnt_zero / n_elements;
        double p_pos  = (double)cnt_pos  / n_elements;

        double H = 0.0;
        if (p_neg  > 0) H -= p_neg  * log2(p_neg);
        if (p_zero > 0) H -= p_zero * log2(p_zero);
        if (p_pos  > 0) H -= p_pos  * log2(p_pos);

        printf("%-40s %12lld %7.2f%% %7.2f%% %7.2f%% %9.4f\n",
               t->name,
               (long long)n_elements,
               p_neg * 100.0, p_zero * 100.0, p_pos * 100.0,
               H);

        global_neg  += cnt_neg;
        global_zero += cnt_zero;
        global_pos  += cnt_pos;
        total_i2s_packed_bytes += (n_elements + 3) / 4;

        free(weights);
    }

    /* Global summary */
    int64_t total_weights = global_neg + global_zero + global_pos;
    double gp_neg  = (double)global_neg  / total_weights;
    double gp_zero = (double)global_zero / total_weights;
    double gp_pos  = (double)global_pos  / total_weights;

    double gH = 0.0;
    if (gp_neg  > 0) gH -= gp_neg  * log2(gp_neg);
    if (gp_zero > 0) gH -= gp_zero * log2(gp_zero);
    if (gp_pos  > 0) gH -= gp_pos  * log2(gp_pos);

    double i2s_size_mb = total_i2s_packed_bytes / (1024.0 * 1024.0);
    double entropy_size_bits = gH * total_weights;
    double entropy_size_mb = entropy_size_bits / 8.0 / (1024.0 * 1024.0);
    double savings_mb = i2s_size_mb - entropy_size_mb;
    double uniform_H = log2(3.0);

    printf("\n=== Global Summary ===\n\n");
    printf("I2_S tensors:       %d\n", n_i2s_tensors);
    printf("Total weights:      %lld (%.1f M)\n",
           (long long)total_weights, total_weights / 1e6);
    printf("\n");
    printf("Distribution:\n");
    printf("  -1:  %lld (%.2f%%)\n", (long long)global_neg, gp_neg * 100.0);
    printf("   0:  %lld (%.2f%%)\n", (long long)global_zero, gp_zero * 100.0);
    printf("  +1:  %lld (%.2f%%)\n", (long long)global_pos, gp_pos * 100.0);
    printf("\n");
    printf("Entropy:            %.4f bits/weight\n", gH);
    printf("Uniform entropy:    %.4f bits/weight (log2(3))\n", uniform_H);
    printf("I2_S encoding:      2.0000 bits/weight\n");
    printf("\n");
    printf("=== Size Comparison ===\n\n");
    printf("I2_S (2 bit):       %.1f MB\n", i2s_size_mb);
    printf("Entropy limit:      %.1f MB (%.4f bits/weight)\n", entropy_size_mb, gH);
    printf("Potential savings:  %.1f MB (%.1f%%)\n",
           savings_mb, savings_mb / i2s_size_mb * 100.0);
    printf("Uniform limit:      %.1f MB (%.4f bits/weight)\n",
           uniform_H * total_weights / 8.0 / (1024.0 * 1024.0), uniform_H);

    gguf_free(&gguf);
    free(data);
    return 0;
}
