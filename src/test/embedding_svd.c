/*
 * SVD analysis of the embedding table.
 *
 * Computes the covariance matrix of embedding vectors and finds eigenvalues
 * to determine the effective rank of the embedding space.
 *
 * Since we don't want a LAPACK dependency, we use power iteration to find
 * eigenvalues of the covariance matrix (E^T @ E), which gives us the
 * squared singular values of E.
 *
 * Reports:
 *   - Singular value spectrum (energy per dimension)
 *   - Cumulative energy (how many dims capture 90/95/99% of variance)
 *   - Effective rank
 *   - Per-dimension variance of embedding vectors
 *
 * Usage: ./embedding_svd <model.gguf>
 */

#include "../inference/gguf.h"
#include "../inference/weight_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Compute C = E^T @ E  (dim x dim covariance-like matrix)
 * E is [vocab_size x dim] stored as F16 */
static void compute_covariance(const uint16_t *emb_f16, int32_t vocab_size,
                                int32_t dim, double *C) {
    /* Zero out */
    memset(C, 0, (size_t)dim * dim * sizeof(double));

    /* Accumulate E^T @ E row by row to keep memory usage low */
    float *row = (float *)malloc(dim * sizeof(float));

    for (int32_t v = 0; v < vocab_size; v++) {
        const uint16_t *src = &emb_f16[(int64_t)v * dim];
        for (int32_t j = 0; j < dim; j++) {
            row[j] = f16_to_f32(src[j]);
        }

        /* Outer product accumulation: C += row^T @ row */
        for (int32_t i = 0; i < dim; i++) {
            double ri = row[i];
            for (int32_t j = i; j < dim; j++) {
                C[(int64_t)i * dim + j] += ri * (double)row[j];
            }
        }

        if (v % 10000 == 0) {
            fprintf(stderr, "\rCovariance: %d/%d tokens (%.0f%%)",
                    v, vocab_size, 100.0 * v / vocab_size);
        }
    }
    fprintf(stderr, "\rCovariance: done.                          \n");

    /* Fill lower triangle (symmetric) */
    for (int32_t i = 0; i < dim; i++) {
        for (int32_t j = 0; j < i; j++) {
            C[(int64_t)i * dim + j] = C[(int64_t)j * dim + i];
        }
    }

    free(row);
}

/* Power iteration with deflation to find top-k eigenvalues of symmetric matrix.
 * Returns eigenvalues in descending order. */
static void find_eigenvalues(double *C, int32_t dim, double *eigenvalues,
                              int32_t k, int32_t max_iter) {
    double *v = (double *)malloc(dim * sizeof(double));
    double *Av = (double *)malloc(dim * sizeof(double));

    for (int32_t e = 0; e < k; e++) {
        /* Random init */
        for (int32_t i = 0; i < dim; i++) {
            v[i] = ((double)rand() / RAND_MAX) - 0.5;
        }

        /* Normalize */
        double norm = 0;
        for (int32_t i = 0; i < dim; i++) norm += v[i] * v[i];
        norm = sqrt(norm);
        for (int32_t i = 0; i < dim; i++) v[i] /= norm;

        double lambda = 0;
        for (int32_t iter = 0; iter < max_iter; iter++) {
            /* Av = C @ v */
            for (int32_t i = 0; i < dim; i++) {
                double sum = 0;
                for (int32_t j = 0; j < dim; j++) {
                    sum += C[(int64_t)i * dim + j] * v[j];
                }
                Av[i] = sum;
            }

            /* Eigenvalue estimate = v^T @ Av */
            lambda = 0;
            for (int32_t i = 0; i < dim; i++) lambda += v[i] * Av[i];

            /* Normalize Av -> v */
            norm = 0;
            for (int32_t i = 0; i < dim; i++) norm += Av[i] * Av[i];
            norm = sqrt(norm);
            if (norm < 1e-15) break;
            for (int32_t i = 0; i < dim; i++) v[i] = Av[i] / norm;
        }

        eigenvalues[e] = lambda;

        /* Deflate: C = C - lambda * v @ v^T */
        for (int32_t i = 0; i < dim; i++) {
            for (int32_t j = i; j < dim; j++) {
                double d = lambda * v[i] * v[j];
                C[(int64_t)i * dim + j] -= d;
                if (j != i) C[(int64_t)j * dim + i] -= d;
            }
        }

        if (e % 50 == 0 || e == k - 1) {
            fprintf(stderr, "\rEigenvalues: %d/%d", e + 1, k);
        }
    }
    fprintf(stderr, "\rEigenvalues: done.            \n");

    free(v);
    free(Av);
}

/* Compute per-dimension variance directly */
static void compute_dim_variance(const uint16_t *emb_f16, int32_t vocab_size,
                                  int32_t dim, double *means, double *variances) {
    memset(means, 0, dim * sizeof(double));
    memset(variances, 0, dim * sizeof(double));

    for (int32_t v = 0; v < vocab_size; v++) {
        const uint16_t *src = &emb_f16[(int64_t)v * dim];
        for (int32_t j = 0; j < dim; j++) {
            double val = f16_to_f32(src[j]);
            means[j] += val;
        }
    }
    for (int32_t j = 0; j < dim; j++) means[j] /= vocab_size;

    for (int32_t v = 0; v < vocab_size; v++) {
        const uint16_t *src = &emb_f16[(int64_t)v * dim];
        for (int32_t j = 0; j < dim; j++) {
            double val = f16_to_f32(src[j]);
            double d = val - means[j];
            variances[j] += d * d;
        }
    }
    for (int32_t j = 0; j < dim; j++) variances[j] /= vocab_size;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
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

    int32_t dim = gguf.hidden_size;
    int32_t vocab_size = gguf.vocab_size;

    printf("=== Embedding Table SVD Analysis ===\n\n");
    printf("Vocab size: %d\n", vocab_size);
    printf("Hidden dim: %d\n", dim);
    printf("Embedding size: %.1f MB (F16)\n",
           (int64_t)vocab_size * dim * 2.0 / (1024.0 * 1024.0));

    /* Load embedding as raw F16 */
    gguf_tensor_info_t *t = gguf_find_tensor(&gguf, "token_embd.weight");
    if (!t || t->type != GGML_TYPE_F16) {
        fprintf(stderr, "token_embd.weight not found or not F16\n");
        free(data);
        return 1;
    }
    const uint16_t *emb_f16 = (const uint16_t *)gguf_tensor_data(&gguf, t, data);

    /* 1. Per-dimension variance */
    printf("\n--- Per-Dimension Variance ---\n\n");
    double *means = (double *)calloc(dim, sizeof(double));
    double *variances = (double *)calloc(dim, sizeof(double));
    compute_dim_variance(emb_f16, vocab_size, dim, means, variances);

    /* Sort variances to show distribution */
    double *sorted_var = (double *)malloc(dim * sizeof(double));
    memcpy(sorted_var, variances, dim * sizeof(double));
    /* Simple sort (dim=2560, fine for insertion sort) */
    for (int32_t i = 1; i < dim; i++) {
        double key = sorted_var[i];
        int32_t j = i - 1;
        while (j >= 0 && sorted_var[j] < key) {
            sorted_var[j + 1] = sorted_var[j];
            j--;
        }
        sorted_var[j + 1] = key;
    }

    double total_var = 0;
    for (int32_t j = 0; j < dim; j++) total_var += sorted_var[j];

    printf("Total variance: %.4f\n", total_var);
    printf("Top 10 dim variances: ");
    for (int i = 0; i < 10 && i < dim; i++) printf("%.4f ", sorted_var[i]);
    printf("\n");
    printf("Bottom 10 dim variances: ");
    for (int i = dim - 10; i < dim; i++) printf("%.6f ", sorted_var[i]);
    printf("\n\n");

    /* Cumulative variance by raw dimensions */
    double cum = 0;
    int p90_raw = -1, p95_raw = -1, p99_raw = -1;
    for (int32_t j = 0; j < dim; j++) {
        cum += sorted_var[j];
        if (p90_raw < 0 && cum >= 0.90 * total_var) p90_raw = j + 1;
        if (p95_raw < 0 && cum >= 0.95 * total_var) p95_raw = j + 1;
        if (p99_raw < 0 && cum >= 0.99 * total_var) p99_raw = j + 1;
    }
    printf("Dims for 90%% variance: %d / %d (%.1f%%)\n", p90_raw, dim, 100.0 * p90_raw / dim);
    printf("Dims for 95%% variance: %d / %d (%.1f%%)\n", p95_raw, dim, 100.0 * p95_raw / dim);
    printf("Dims for 99%% variance: %d / %d (%.1f%%)\n", p99_raw, dim, 100.0 * p99_raw / dim);

    /* Count low-variance dimensions */
    double var_thresh = total_var / dim * 0.01;  /* 1% of mean variance */
    int n_low_var = 0;
    for (int32_t j = 0; j < dim; j++) {
        if (variances[j] < var_thresh) n_low_var++;
    }
    printf("Dims with <1%% of mean variance: %d / %d\n\n", n_low_var, dim);

    /* 2. SVD via eigenvalues of E^T @ E */
    printf("--- Singular Value Spectrum (via E^T @ E eigenvalues) ---\n\n");

    double *C = (double *)malloc((int64_t)dim * dim * sizeof(double));
    if (!C) {
        fprintf(stderr, "Cannot allocate %dx%d covariance matrix (%.0f MB)\n",
                dim, dim, (double)dim * dim * 8 / 1024 / 1024);
        free(data);
        return 1;
    }

    compute_covariance(emb_f16, vocab_size, dim, C);

    /* Find all eigenvalues */
    double *eigenvalues = (double *)malloc(dim * sizeof(double));
    find_eigenvalues(C, dim, eigenvalues, dim, 200);

    /* Singular values = sqrt(eigenvalues) */
    double total_energy = 0;
    for (int32_t i = 0; i < dim; i++) {
        if (eigenvalues[i] < 0) eigenvalues[i] = 0;
        total_energy += eigenvalues[i];
    }

    /* Print spectrum */
    printf("Rank  Eigenvalue     SingVal   Energy%%  Cumul%%\n");
    printf("----  ----------     -------   -------  ------\n");
    cum = 0;
    int p90 = -1, p95 = -1, p99 = -1;
    for (int32_t i = 0; i < dim; i++) {
        cum += eigenvalues[i];
        double pct = 100.0 * eigenvalues[i] / total_energy;
        double cum_pct = 100.0 * cum / total_energy;

        if (p90 < 0 && cum_pct >= 90.0) p90 = i + 1;
        if (p95 < 0 && cum_pct >= 95.0) p95 = i + 1;
        if (p99 < 0 && cum_pct >= 99.0) p99 = i + 1;

        /* Print first 30, then every 100th, then last 10 */
        if (i < 30 || i % 100 == 0 || i >= dim - 10 ||
            (p90 == i + 1) || (p95 == i + 1) || (p99 == i + 1)) {
            printf("%4d  %12.2f  %9.2f  %6.3f%%  %6.2f%%\n",
                   i + 1, eigenvalues[i], sqrt(eigenvalues[i]), pct, cum_pct);
        }
    }

    printf("\n=== Summary ===\n\n");
    printf("Effective rank (90%% energy): %d / %d dims\n", p90, dim);
    printf("Effective rank (95%% energy): %d / %d dims\n", p95, dim);
    printf("Effective rank (99%% energy): %d / %d dims\n", p99, dim);

    double ratio_90 = (double)p90 / dim;
    double ratio_99 = (double)p99 / dim;
    printf("\nImplication for weight pruning:\n");
    printf("  If 90%% energy in %d/%d dims, ~%.0f%% of weight connections\n",
           p90, dim, (1.0 - ratio_90) * 100);
    printf("  operate on low-energy subspace and may be prunable.\n");
    printf("  At 99%%: %d/%d dims → ~%.0f%% potentially prunable.\n",
           p99, dim, (1.0 - ratio_99) * 100);

    /* Estimate ternary weight savings */
    int64_t total_ternary = 2084044800LL;  /* from entropy analysis */
    double prunable_90 = total_ternary * (1.0 - ratio_90) * 2.0 / 8.0 / 1024.0 / 1024.0;
    double prunable_99 = total_ternary * (1.0 - ratio_99) * 2.0 / 8.0 / 1024.0 / 1024.0;
    printf("\n  Weight savings (I2_S) if pruning low-energy dims:\n");
    printf("    90%% cutoff: zero ~%.0f MB of weights (of 497 MB)\n", prunable_90);
    printf("    99%% cutoff: zero ~%.0f MB of weights (of 497 MB)\n", prunable_99);

    free(eigenvalues);
    free(C);
    free(means);
    free(variances);
    free(sorted_var);
    gguf_free(&gguf);
    free(data);
    return 0;
}
