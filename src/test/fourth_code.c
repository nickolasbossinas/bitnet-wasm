/*
 * Fourth code analysis: what should the unused 2-bit code `11` represent?
 *
 * Analyzes decoded ternary weight sequences to evaluate strategies:
 *   1. "Same as previous" — run-length on any value
 *   2. "Zero run" — next N weights are zero (various N encodings)
 *   3. "Repeat previous pair" — copy last 2 weights
 *   4. Combined: context-dependent encoding
 *
 * For each strategy, estimates how many 2-bit slots would be freed
 * and the resulting compressed size.
 *
 * Usage: ./fourth_code <model.gguf>
 */

#include "../inference/gguf.h"
#include "../inference/weight_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Collect all decoded ternary weights into one big array */
static int8_t *decode_all_weights(const gguf_context_t *gguf,
                                   const uint8_t *file_data,
                                   int64_t *total_out) {
    /* First pass: count total */
    int64_t total = 0;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        gguf_tensor_info_t *t = &gguf->tensors[i];
        if (t->type != GGML_TYPE_I2_S) continue;
        int64_t n = 1;
        for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
        total += n;
    }

    int8_t *all = (int8_t *)malloc(total);
    if (!all) return NULL;

    /* Second pass: decode */
    int64_t offset = 0;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        gguf_tensor_info_t *t = &gguf->tensors[i];
        if (t->type != GGML_TYPE_I2_S) continue;
        int64_t n = 1;
        for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
        const uint8_t *src = (const uint8_t *)gguf_tensor_data(gguf, t, file_data);
        i2s_decode(src, &all[offset], n);
        offset += n;
    }

    *total_out = total;
    return all;
}

/* Strategy 1: "Same as previous" */
static void analyze_same_as_prev(const int8_t *w, int64_t n) {
    printf("\n=== Strategy 1: 11 = \"Same as previous\" ===\n\n");

    int64_t same_count = 0;
    int64_t run_lengths[64] = {0};  /* histogram of consecutive same-as-prev runs */

    int64_t i = 1;
    while (i < n) {
        if (w[i] == w[i - 1]) {
            int64_t run = 0;
            while (i < n && w[i] == w[i - 1]) {
                run++;
                i++;
            }
            same_count += run;
            int bucket = run < 63 ? run : 63;
            run_lengths[bucket]++;
        } else {
            i++;
        }
    }

    double pct = 100.0 * same_count / n;
    printf("Consecutive same-as-prev pairs: %lld (%.2f%% of all weights)\n",
           (long long)same_count, pct);

    /* With 11=same_as_prev, each such weight costs 2 bits instead of 2 bits,
     * BUT we can now encode 3 values + repeat in 2 bits.
     * The saving comes from being able to pack more efficiently afterward. */

    /* Simple model: each "same" saves nothing directly (still 2 bits),
     * but it means those slots use code 11, making the distribution
     * over 4 codes more even, which helps entropy coding on top. */

    /* Better model: use 11 to represent repeat, freeing slots.
     * Without 11: need 2 bits for {-1,0,+1}. With 11=repeat:
     * first weight in a run: 2 bits, subsequent: 2 bits (code 11).
     * No direct saving, but enables RLE on top. */

    printf("\nRun length distribution (consecutive same values):\n");
    printf("  Length    Runs        Total weights saved by RLE\n");
    for (int r = 1; r < 64; r++) {
        if (run_lengths[r] > 0) {
            /* With simple 11=same: each repeat is 2 bits, same as raw. No saving.
             * With RLE using 11: "11 + count" could encode long runs.
             * E.g., 11 alone = repeat 1x, 11-11 = repeat 2x, etc. */
            printf("  %3d       %8lld\n", r, (long long)run_lengths[r]);
        }
    }

    /* Encoding: if 11 means "same as prev", we save 0 bits per occurrence
     * but we gain the ability to use arithmetic/Huffman coding more effectively
     * because the 4th symbol concentrates probability mass. */
    double p_same = (double)same_count / n;
    double p_other = 1.0 - p_same;
    /* After splitting: 3 symbols share (1-p_same) probability, plus p_same for "same" */
    /* Effective entropy with 4 symbols vs 3: */
    /* This doesn't save bits directly with fixed 2-bit coding, but shows potential. */
    printf("\nSame-as-prev probability: %.4f\n", p_same);
    printf("If entropy-coded with 4th symbol: would reduce entropy further.\n");
}

/* Strategy 2: "Zero run" — various encodings */
static void analyze_zero_runs(const int8_t *w, int64_t n) {
    printf("\n=== Strategy 2: 11 = \"Zero run\" marker ===\n\n");

    /* Find all zero runs */
    int64_t run_hist[256] = {0};
    int64_t total_zeros_in_runs = 0;
    int64_t n_runs = 0;

    int64_t i = 0;
    while (i < n) {
        if (w[i] == 0) {
            int64_t start = i;
            while (i < n && w[i] == 0) i++;
            int64_t len = i - start;
            int bucket = len < 255 ? (int)len : 255;
            run_hist[bucket]++;
            total_zeros_in_runs += len;
            n_runs++;
        } else {
            i++;
        }
    }

    printf("Total zeros: %lld (%.2f%%)\n",
           (long long)total_zeros_in_runs, 100.0 * total_zeros_in_runs / n);
    printf("Zero runs:   %lld\n", (long long)n_runs);
    if (n_runs > 0)
        printf("Avg run len: %.1f\n", (double)total_zeros_in_runs / n_runs);

    printf("\nZero run length distribution:\n");
    printf("  Length    Runs         Zeros\n");
    for (int r = 1; r < 256; r++) {
        if (run_hist[r] > 0) {
            printf("  %3d       %8lld     %8lld\n",
                   r, (long long)run_hist[r], (long long)run_hist[r] * r);
        }
    }

    /*
     * Fixed-meaning schemes: 11 = "next N weights are all zero"
     * One 2-bit code replaces N×2 bits of zeros → saves (N×2 - 2) bits.
     * Greedily consume runs: each run of length L produces floor(L/N) codes.
     */
    printf("\n--- Fixed N-zero schemes (11 = N zeros) ---\n\n");
    printf("  N    Codes used      Bits saved     Saved MB    New size MB    Bits/wt\n");
    printf("  --   ----------      ----------     --------    -----------    -------\n");

    for (int N = 2; N <= 10; N++) {
        int64_t codes = 0;
        for (int r = N; r < 256; r++) {
            codes += run_hist[r] * (r / N);
        }
        int64_t saved_bits = codes * (N * 2 - 2);
        double saved_mb = saved_bits / 8.0 / (1024.0 * 1024.0);
        double new_size = (n * 2.0 - saved_bits) / 8.0 / (1024.0 * 1024.0);
        double bpw = (n * 2.0 - saved_bits) / n;
        printf("  %2d   %12lld   %12lld     %6.1f      %8.1f       %.4f\n",
               N, (long long)codes, (long long)saved_bits, saved_mb, new_size, bpw);
    }

    /* Mixed: use two codes if we could have two 4th-code meanings.
     * But we only have one unused code (11). What if we use it adaptively?
     *
     * Better idea: greedy optimal. For each run of length L, pick the
     * best single N that maximizes total savings for that specific run.
     * savings(N,L) = floor(L/N) × (2N-2)
     */
    printf("\n--- Optimal N per run (greedy best N for each run length) ---\n\n");

    int64_t total_saved_optimal = 0;
    int64_t best_N_hist[11] = {0};

    for (int r = 2; r < 256; r++) {
        if (run_hist[r] == 0) continue;

        /* Find N that maximizes savings for this run length */
        int best_N = 2;
        int64_t best_savings = 0;
        for (int N = 2; N <= 10; N++) {
            int64_t codes = r / N;
            int64_t savings = codes * (N * 2 - 2);
            if (savings > best_savings) {
                best_savings = savings;
                best_N = N;
            }
        }
        total_saved_optimal += best_savings * run_hist[r];
        best_N_hist[best_N] += run_hist[r];
    }

    double opt_saved_mb = total_saved_optimal / 8.0 / (1024.0 * 1024.0);
    double opt_new_size = (n * 2.0 - total_saved_optimal) / 8.0 / (1024.0 * 1024.0);
    printf("If we could pick best N per run: saves %.1f MB → %.1f MB\n",
           opt_saved_mb, opt_new_size);
    printf("Best N distribution:\n");
    for (int N = 2; N <= 10; N++) {
        if (best_N_hist[N] > 0) {
            printf("  N=%d: %lld runs\n", N, (long long)best_N_hist[N]);
        }
    }
}

/* Strategy 3: general "repeat previous N" */
static void analyze_repeat_patterns(const int8_t *w, int64_t n) {
    printf("\n=== Strategy 3: Repeat patterns ===\n\n");

    /* Count consecutive pairs (w[i]==w[i-1] && w[i+1]==w[i-2]) */
    int64_t pair_repeats = 0;
    for (int64_t i = 2; i < n - 1; i += 2) {
        if (w[i] == w[i - 2] && w[i + 1] == w[i - 1]) {
            pair_repeats++;
        }
    }
    printf("Consecutive pair repeats: %lld (%.2f%% of pair slots)\n",
           (long long)pair_repeats, 100.0 * pair_repeats / (n / 2));

    /* Count transitions */
    int64_t trans[3][3] = {{0}};
    for (int64_t i = 1; i < n; i++) {
        int from = w[i - 1] + 1;  /* map -1,0,+1 -> 0,1,2 */
        int to = w[i] + 1;
        trans[from][to]++;
    }

    printf("\nTransition matrix (from row -> to col):\n");
    printf("         -1        0       +1\n");
    const char *labels[] = {"-1", " 0", "+1"};
    for (int r = 0; r < 3; r++) {
        int64_t row_total = trans[r][0] + trans[r][1] + trans[r][2];
        printf("  %s   %5.1f%%   %5.1f%%   %5.1f%%\n",
               labels[r],
               100.0 * trans[r][0] / row_total,
               100.0 * trans[r][1] / row_total,
               100.0 * trans[r][2] / row_total);
    }

    /* Self-transition rate (same as previous) */
    int64_t self = trans[0][0] + trans[1][1] + trans[2][2];
    printf("\nSelf-transition rate: %.2f%% (same as previous)\n",
           100.0 * self / (n - 1));
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

    printf("=== Fourth Code (11) Analysis ===\n");

    int64_t total;
    int8_t *weights = decode_all_weights(&gguf, data, &total);
    if (!weights) {
        fprintf(stderr, "OOM decoding weights\n");
        free(data);
        return 1;
    }

    printf("Total ternary weights: %lld (%.1f MB at 2 bits)\n\n",
           (long long)total, total * 2.0 / 8.0 / (1024.0 * 1024.0));

    analyze_same_as_prev(weights, total);
    analyze_zero_runs(weights, total);
    analyze_repeat_patterns(weights, total);

    free(weights);
    gguf_free(&gguf);
    free(data);
    return 0;
}
