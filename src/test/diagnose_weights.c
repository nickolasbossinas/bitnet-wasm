#include "../inference/gguf.h"
#include "../inference/weight_loader.h"
#include "../kernels/tl1.h"
#include "../kernels/gemv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Diagnostic tool to verify I2_S weight loading and TL1 GEMV correctness.
 *
 * Checks:
 *   1. Whether _scale tensors exist in GGUF (per-tensor weight scales)
 *   2. Raw I2_S byte analysis (hex dump + decode with LSB and MSB ordering)
 *   3. Weight value distribution
 *   4. Single-row dot product verification (TL1 vs naive float)
 *   5. First-layer forward pass output sample
 */

/* Decode I2_S with MSB-first ordering (bits 6-7 = weight 0) */
static void i2s_decode_msb(const uint8_t *data, int8_t *out, int64_t n_elements) {
    int64_t full_bytes = n_elements / 4;
    int64_t remainder = n_elements % 4;

    for (int64_t i = 0; i < full_bytes; i++) {
        uint8_t byte = data[i];
        out[i * 4 + 0] = (int8_t)((byte >> 6) & 0x03) - 1;
        out[i * 4 + 1] = (int8_t)((byte >> 4) & 0x03) - 1;
        out[i * 4 + 2] = (int8_t)((byte >> 2) & 0x03) - 1;
        out[i * 4 + 3] = (int8_t)((byte >> 0) & 0x03) - 1;
    }

    if (remainder > 0) {
        uint8_t byte = data[full_bytes];
        for (int64_t j = 0; j < remainder; j++) {
            out[full_bytes * 4 + j] = (int8_t)((byte >> (6 - j * 2)) & 0x03) - 1;
        }
    }
}

/* Decode I2_S with interleaved block format (QK=128, 4 groups of 32) */
static void i2s_decode_interleaved(const uint8_t *data, int8_t *out,
                                    int64_t n_elements) {
    int64_t n_blocks = n_elements / 128;
    int64_t remainder = n_elements % 128;

    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const uint8_t *blk_data = &data[blk * 32];
        int8_t *blk_out = &out[blk * 128];

        for (int32_t b = 0; b < 32; b++) {
            uint8_t byte = blk_data[b];
            /* Group 0 (weights 0..31): bits 6-7 */
            blk_out[0 * 32 + b] = (int8_t)((byte >> 6) & 0x03) - 1;
            /* Group 1 (weights 32..63): bits 4-5 */
            blk_out[1 * 32 + b] = (int8_t)((byte >> 4) & 0x03) - 1;
            /* Group 2 (weights 64..95): bits 2-3 */
            blk_out[2 * 32 + b] = (int8_t)((byte >> 2) & 0x03) - 1;
            /* Group 3 (weights 96..127): bits 0-1 */
            blk_out[3 * 32 + b] = (int8_t)((byte >> 0) & 0x03) - 1;
        }
    }

    /* Handle remaining elements (simple sequential for tail) */
    if (remainder > 0) {
        int64_t tail_offset = n_blocks * 128;
        int64_t tail_bytes = (remainder + 3) / 4;
        const uint8_t *tail_data = &data[n_blocks * 32];
        for (int64_t i = 0; i < tail_bytes && tail_offset < n_elements; i++) {
            uint8_t byte = tail_data[i];
            for (int j = 0; j < 4 && tail_offset < n_elements; j++) {
                out[tail_offset++] = (int8_t)((byte >> (6 - j * 2)) & 0x03) - 1;
            }
        }
    }
}

/* Naive float dot product: sum(w[j] * x[j]) for j=0..K-1 */
static float naive_dot(const int8_t *weights, const float *activations,
                       int32_t K) {
    float sum = 0.0f;
    for (int32_t j = 0; j < K; j++) {
        sum += (float)weights[j] * activations[j];
    }
    return sum;
}

/* Check if a scale tensor exists for a given weight tensor name */
static gguf_tensor_info_t *find_scale_tensor(const gguf_context_t *ctx,
                                              const char *weight_name) {
    char scale_name[256];
    snprintf(scale_name, sizeof(scale_name), "%s_scale", weight_name);
    return gguf_find_tensor(ctx, scale_name);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <gguf-file>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", argv[1]);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *data = (uint8_t *)malloc(file_size);
    if (!data) {
        fprintf(stderr, "Failed to allocate %ld bytes\n", file_size);
        fclose(f);
        return 1;
    }
    fread(data, 1, file_size, f);
    fclose(f);

    gguf_context_t ctx;
    if (gguf_parse(&ctx, data, file_size) != 0) {
        fprintf(stderr, "Failed to parse GGUF\n");
        free(data);
        return 1;
    }

    printf("=== I2_S Weight Diagnostic ===\n\n");

    /* ============================================
     * 1. Check for _scale tensors
     * ============================================ */
    printf("--- 1. Scale tensor search ---\n");
    int n_scale_found = 0;
    int n_i2s_found = 0;
    for (uint64_t i = 0; i < ctx.n_tensors; i++) {
        gguf_tensor_info_t *t = &ctx.tensors[i];
        if (t->type == GGML_TYPE_I2_S) {
            n_i2s_found++;
            /* Check for corresponding _scale tensor */
            gguf_tensor_info_t *st = find_scale_tensor(&ctx, t->name);
            if (st) {
                n_scale_found++;
                if (n_scale_found <= 3) {
                    const float *scale_data = (const float *)gguf_tensor_data(&ctx, st, data);
                    printf("  FOUND: %s_scale = %f (type %s)\n",
                           t->name, *scale_data, ggml_type_name(st->type));
                }
            }
        }
    }
    printf("  I2_S tensors: %d, scale tensors found: %d\n", n_i2s_found, n_scale_found);
    if (n_scale_found == 0) {
        printf("  WARNING: No _scale tensors found! Per-tensor scales may be missing.\n");
    }

    /* Also check for any tensor with "_scale" in the name */
    printf("\n  All tensors containing '_scale':\n");
    int any_scale = 0;
    for (uint64_t i = 0; i < ctx.n_tensors; i++) {
        if (strstr(ctx.tensors[i].name, "scale") != NULL ||
            strstr(ctx.tensors[i].name, "Scale") != NULL) {
            printf("    %s (type=%s)\n", ctx.tensors[i].name,
                   ggml_type_name(ctx.tensors[i].type));
            any_scale = 1;
        }
    }
    if (!any_scale) printf("    (none found)\n");

    /* ============================================
     * 2. Raw I2_S byte analysis
     * ============================================ */
    printf("\n--- 2. Raw I2_S byte analysis ---\n");

    /* Find first I2_S tensor (should be blk.0.attn_q.weight) */
    gguf_tensor_info_t *first_i2s = NULL;
    for (uint64_t i = 0; i < ctx.n_tensors; i++) {
        if (ctx.tensors[i].type == GGML_TYPE_I2_S) {
            first_i2s = &ctx.tensors[i];
            break;
        }
    }

    if (!first_i2s) {
        printf("  No I2_S tensors found!\n");
        gguf_free(&ctx);
        free(data);
        return 0;
    }

    printf("  First I2_S tensor: %s\n", first_i2s->name);
    printf("  Dims: [%lld", (long long)first_i2s->dims[0]);
    for (int d = 1; d < first_i2s->n_dims; d++)
        printf(", %lld", (long long)first_i2s->dims[d]);
    printf("]\n");

    int64_t n_elements = 1;
    for (int d = 0; d < first_i2s->n_dims; d++)
        n_elements *= first_i2s->dims[d];

    int64_t packed_bytes = (n_elements + 3) / 4;
    printf("  Total elements: %lld, packed bytes: %lld\n",
           (long long)n_elements, (long long)packed_bytes);

    const uint8_t *i2s_data = (const uint8_t *)gguf_tensor_data(&ctx, first_i2s, data);

    /* Check if there's a float scale appended after the packed data */
    printf("\n  Checking for appended scale (4 bytes after packed data):\n");
    if (first_i2s->offset + packed_bytes + 4 <=
        (uint64_t)file_size - ctx.data_offset) {
        float appended_scale;
        memcpy(&appended_scale, &i2s_data[packed_bytes], sizeof(float));
        printf("    Bytes at offset %lld: %02x %02x %02x %02x\n",
               (long long)packed_bytes,
               i2s_data[packed_bytes], i2s_data[packed_bytes+1],
               i2s_data[packed_bytes+2], i2s_data[packed_bytes+3]);
        printf("    As float: %f\n", appended_scale);
        if (appended_scale > 0.0f && appended_scale < 100.0f) {
            printf("    >>> LIKELY A VALID SCALE FACTOR! <<<\n");
        }
    }

    /* Print first 32 raw bytes */
    printf("\n  First 32 raw bytes (hex):\n    ");
    for (int i = 0; i < 32 && i < packed_bytes; i++) {
        printf("%02x ", i2s_data[i]);
        if ((i + 1) % 16 == 0) printf("\n    ");
    }
    printf("\n");

    /* ============================================
     * 3. Decode comparison (3 methods)
     * ============================================ */
    printf("\n--- 3. Decode comparison (first 32 weights) ---\n");

    int32_t sample_n = 128;
    int8_t *dec_lsb = (int8_t *)malloc(sample_n);
    int8_t *dec_msb = (int8_t *)malloc(sample_n);
    int8_t *dec_interleaved = (int8_t *)malloc(sample_n);

    i2s_decode(i2s_data, dec_lsb, sample_n);
    i2s_decode_msb(i2s_data, dec_msb, sample_n);
    i2s_decode_interleaved(i2s_data, dec_interleaved, sample_n);

    printf("  Position:     ");
    for (int i = 0; i < 32; i++) printf("%3d", i);
    printf("\n");

    printf("  LSB-first:    ");
    for (int i = 0; i < 32; i++) printf("%3d", dec_lsb[i]);
    printf("\n");

    printf("  MSB-first:    ");
    for (int i = 0; i < 32; i++) printf("%3d", dec_msb[i]);
    printf("\n");

    printf("  Interleaved:  ");
    for (int i = 0; i < 32; i++) printf("%3d", dec_interleaved[i]);
    printf("\n");

    /* Check if any method produces values outside {-1, 0, 1} */
    printf("\n  Value check (first 128 weights):\n");
    for (int method = 0; method < 3; method++) {
        int8_t *dec = (method == 0) ? dec_lsb :
                      (method == 1) ? dec_msb : dec_interleaved;
        const char *name = (method == 0) ? "LSB-first" :
                           (method == 1) ? "MSB-first" : "Interleaved";
        int count[5] = {0};  /* -1, 0, 1, other_neg, other_pos */
        for (int i = 0; i < sample_n; i++) {
            if (dec[i] == -1) count[0]++;
            else if (dec[i] == 0) count[1]++;
            else if (dec[i] == 1) count[2]++;
            else if (dec[i] < 0) count[3]++;
            else count[4]++;
        }
        printf("    %-12s: -1=%d, 0=%d, +1=%d, other=%d\n",
               name, count[0], count[1], count[2], count[3] + count[4]);
    }

    /* ============================================
     * 4. Full tensor weight distribution
     * ============================================ */
    printf("\n--- 4. Full tensor weight distribution ---\n");

    int8_t *full_lsb = (int8_t *)malloc(n_elements);
    int8_t *full_msb = (int8_t *)malloc(n_elements);

    if (full_lsb && full_msb) {
        i2s_decode(i2s_data, full_lsb, n_elements);
        i2s_decode_msb(i2s_data, full_msb, n_elements);

        for (int method = 0; method < 2; method++) {
            int8_t *dec = (method == 0) ? full_lsb : full_msb;
            const char *name = (method == 0) ? "LSB-first" : "MSB-first";
            int64_t cnt_neg = 0, cnt_zero = 0, cnt_pos = 0, cnt_other = 0;
            for (int64_t i = 0; i < n_elements; i++) {
                if (dec[i] == -1) cnt_neg++;
                else if (dec[i] == 0) cnt_zero++;
                else if (dec[i] == 1) cnt_pos++;
                else cnt_other++;
            }
            printf("  %-12s: -1=%.1f%%, 0=%.1f%%, +1=%.1f%%, other=%lld\n",
                   name,
                   100.0 * cnt_neg / n_elements,
                   100.0 * cnt_zero / n_elements,
                   100.0 * cnt_pos / n_elements,
                   (long long)cnt_other);
        }
    }

    /* ============================================
     * 5. Dot product verification
     * ============================================ */
    printf("\n--- 5. Dot product verification ---\n");

    /* Use first row of the first I2_S tensor */
    int32_t K = (int32_t)first_i2s->dims[0];  /* input dim (GGML dims[0]) */
    int32_t M = (first_i2s->n_dims > 1) ? (int32_t)first_i2s->dims[1] : 1;
    printf("  Tensor dims: M=%d (output), K=%d (input)\n", M, K);

    /* Create simple test activation: [1, 2, 3, 4, 5, ...] */
    float *test_act = (float *)calloc(K, sizeof(float));
    for (int32_t j = 0; j < K; j++) {
        test_act[j] = (float)(j % 17) - 8.0f;  /* range [-8, 8] */
    }

    /* Decode full tensor to get weights for all rows */
    int8_t *full_dec = full_lsb;  /* reuse */
    int8_t *full_dec_msb_ptr = full_msb;

    if (full_dec && full_dec_msb_ptr) {
        /* Compute naive dot product for first 4 rows, 3 decode methods */
        int32_t check_rows = (M < 4) ? M : 4;

        printf("  Dot products (first %d rows of %s):\n", check_rows, first_i2s->name);
        printf("    %-10s  %-14s  %-14s\n", "Row", "LSB-first", "MSB-first");

        for (int32_t row = 0; row < check_rows; row++) {
            float dot_lsb = naive_dot(&full_dec[row * K], test_act, K);
            float dot_msb = naive_dot(&full_dec_msb_ptr[row * K], test_act, K);
            printf("    row %-5d  %14.4f  %14.4f\n", row, dot_lsb, dot_msb);
        }

        /* Also run through TL1 pipeline and compare */
        printf("\n  TL1 pipeline verification (row 0):\n");

        /* Pack weights using LSB decode */
        tl1_weight_t tl1_w;
        int32_t pairs = K / 2;
        int32_t bpr = (pairs + 1) / 2;
        tl1_w.indices = (uint8_t *)calloc(1 * bpr, 1);  /* just 1 row */
        tl1_w.indices_col = NULL;
        tl1_w.M = 1;
        tl1_w.K = K;
        tl1_w.scale = 1.0f;
        tl1_pack_weights(full_dec, tl1_w.indices, 1, K);

        /* Quantize activations (same as gemv_run) */
        int8_t *quant_act = (int8_t *)malloc(K);
        float a_scale;
        float absmax = 0.0f;
        for (int32_t j = 0; j < K; j++) {
            float a = fabsf(test_act[j]);
            if (a > absmax) absmax = a;
        }
        a_scale = (absmax > 0) ? (absmax / 127.0f) : 1.0f;
        for (int32_t j = 0; j < K; j++) {
            float v = test_act[j] / a_scale;
            if (v > 127.0f) v = 127.0f;
            if (v < -128.0f) v = -128.0f;
            quant_act[j] = (int8_t)roundf(v);
        }

        /* Build LUT */
        int32_t lut_size = pairs * 16;
        int16_t *lut = (int16_t *)calloc(lut_size, sizeof(int16_t));
        tl1_build_lut(lut, quant_act, K);

        /* Run scalar TL1 GEMV */
        activation_t act = { .data = quant_act, .len = K, .scale = a_scale };
        float tl1_result;
        output_t out = { .data = &tl1_result, .len = 1 };
        tl1_gemv_scalar(&tl1_w, lut, &act, &out);

        /* Compute expected (naive float dot product) */
        float naive_result = naive_dot(full_dec, test_act, K);

        printf("    Naive float dot:     %14.4f\n", naive_result);
        printf("    TL1 pipeline result: %14.4f\n", tl1_result);
        printf("    Difference:          %14.4f (%.2f%%)\n",
               fabsf(naive_result - tl1_result),
               (fabsf(naive_result) > 0) ?
               100.0f * fabsf(naive_result - tl1_result) / fabsf(naive_result) : 0.0f);

        free(tl1_w.indices);
        free(quant_act);
        free(lut);
    }

    /* ============================================
     * 6. Check for alignment issues
     * ============================================ */
    printf("\n--- 6. GGUF data alignment check ---\n");
    printf("  Data offset: %llu\n", (unsigned long long)ctx.data_offset);
    printf("  Assumed alignment: 32\n");

    /* Check for general.alignment KV */
    printf("  (Note: general.alignment KV not parsed; using default 32)\n");

    /* Check a few tensor offsets */
    printf("\n  First 5 tensor offsets:\n");
    for (uint64_t i = 0; i < ctx.n_tensors && i < 5; i++) {
        gguf_tensor_info_t *t = &ctx.tensors[i];
        uint64_t abs_offset = ctx.data_offset + t->offset;
        printf("    %s: relative=%llu, absolute=%llu, aligned_32=%s\n",
               t->name,
               (unsigned long long)t->offset,
               (unsigned long long)abs_offset,
               (abs_offset % 32 == 0) ? "yes" : "NO");
    }

    free(full_lsb);
    free(full_msb);
    free(dec_lsb);
    free(dec_msb);
    free(dec_interleaved);
    free(test_act);
    gguf_free(&ctx);
    free(data);

    printf("\n=== Diagnostic complete ===\n");
    return 0;
}
