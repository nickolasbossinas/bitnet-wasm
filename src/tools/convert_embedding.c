/*
 * GGUF Embedding Converter: F16 → INT8
 *
 * Reads a GGUF model file and writes a new one where token_embd.weight
 * is stored as INT8 (per-row symmetric quantization) instead of F16.
 * Per-row F32 scales are stored in a new tensor token_embd.scales.
 *
 * This halves the embedding from ~626 MB to ~314 MB on disk.
 *
 * Usage: ./convert_embedding <input.gguf> <output.gguf>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* --- Minimal binary reader (matches gguf.c cursor) --- */

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
} cursor_t;

static uint32_t read_u32(cursor_t *c) {
    uint32_t v; memcpy(&v, &c->data[c->pos], 4); c->pos += 4; return v;
}
static uint64_t read_u64(cursor_t *c) {
    uint64_t v; memcpy(&v, &c->data[c->pos], 8); c->pos += 8; return v;
}

/* Read GGUF string, return length and pointer (no copy) */
static size_t read_str_len(cursor_t *c) {
    uint64_t len = read_u64(c);
    return (size_t)len;
}

static void skip_str(cursor_t *c) {
    size_t len = read_str_len(c);
    c->pos += len;
}

/* Skip a metadata value */
static void skip_value(cursor_t *c, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: c->pos += 1; break;  /* u8, i8, bool */
        case 2: case 3: c->pos += 2; break;           /* u16, i16 */
        case 4: case 5: case 6: c->pos += 4; break;   /* u32, i32, f32 */
        case 10: case 11: case 12: c->pos += 8; break; /* u64, i64, f64 */
        case 8: skip_str(c); break;                    /* string */
        case 9: {                                      /* array */
            uint32_t arr_type = read_u32(c);
            uint64_t arr_len = read_u64(c);
            for (uint64_t i = 0; i < arr_len; i++) skip_value(c, arr_type);
            break;
        }
    }
}

/* --- F16 → F32 conversion --- */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) {
            float f; memcpy(&f, &sign, 4); return f;
        }
        exp = 1;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3FF;
        exp = exp + (127 - 15);
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | ((uint32_t)mant << 13);
        float f; memcpy(&f, &bits, 4); return f;
    } else {
        exp = exp + (127 - 15);
    }
    uint32_t bits = sign | ((uint32_t)exp << 23) | ((uint32_t)mant << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* --- Tensor info for tracking --- */

typedef struct {
    char    name[256];
    int32_t n_dims;
    int64_t dims[4];
    int32_t type;
    uint64_t offset;      /* original offset in input data section */
    uint64_t size_bytes;  /* original size */
} tensor_entry_t;

/* Compute tensor size (same as gguf.c) */
static uint64_t tensor_type_size(int32_t type, uint64_t n_elements) {
    switch (type) {
        case 0:  return n_elements * 4;  /* F32 */
        case 1:  return n_elements * 2;  /* F16 */
        case 24: return n_elements;      /* I8 */
        case 36: return (n_elements + 3) / 4;  /* I2_S */
        default: return n_elements;
    }
}

/* --- Binary write helpers --- */

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void write_i32(FILE *f, int32_t v)  { fwrite(&v, 4, 1, f); }
static void write_i64(FILE *f, int64_t v)  { fwrite(&v, 8, 1, f); }

/* Write GGUF string: u64 length + chars */
static void write_gguf_str(FILE *f, const char *s) {
    uint64_t len = strlen(s);
    write_u64(f, len);
    fwrite(s, 1, len, f);
}

/* Pad file to alignment */
static void write_padding(FILE *f, uint64_t alignment) {
    long pos = ftell(f);
    uint64_t aligned = (pos + alignment - 1) & ~(alignment - 1);
    uint64_t pad = aligned - pos;
    if (pad > 0) {
        uint8_t zeros[64] = {0};
        while (pad > 0) {
            uint64_t chunk = pad > 64 ? 64 : pad;
            fwrite(zeros, 1, chunk, f);
            pad -= chunk;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf>\n", argv[0]);
        return 1;
    }

    /* Read input file */
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(file_size);
    if (!data) { fprintf(stderr, "OOM\n"); fclose(fin); return 1; }
    fread(data, 1, file_size, fin);
    fclose(fin);

    printf("Input: %s (%.1f MB)\n", argv[1], file_size / (1024.0 * 1024.0));

    cursor_t c = { .data = data, .size = file_size, .pos = 0 };

    /* Parse header */
    uint32_t magic = read_u32(&c);
    uint32_t version = read_u32(&c);
    uint64_t n_tensors = read_u64(&c);
    uint64_t n_kv = read_u64(&c);

    if (magic != 0x46554747) {
        fprintf(stderr, "Bad GGUF magic\n");
        free(data);
        return 1;
    }

    printf("GGUF v%u: %llu tensors, %llu KV pairs\n",
           version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    /* Skip all KV pairs (we'll copy their raw bytes) */
    size_t kv_start = c.pos;
    for (uint64_t i = 0; i < n_kv; i++) {
        skip_str(&c);           /* key */
        uint32_t vtype = read_u32(&c);  /* value type */
        skip_value(&c, vtype);  /* value */
    }
    size_t kv_end = c.pos;
    printf("KV section: bytes %zu - %zu (%zu bytes)\n", kv_start, kv_end, kv_end - kv_start);

    /* Parse tensor info */
    tensor_entry_t *tensors = (tensor_entry_t *)calloc(n_tensors, sizeof(tensor_entry_t));
    int embd_idx = -1;

    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_entry_t *t = &tensors[i];
        size_t name_len = read_str_len(&c);
        if (name_len >= sizeof(t->name)) name_len = sizeof(t->name) - 1;
        memcpy(t->name, &c.data[c.pos], name_len);
        t->name[name_len] = '\0';
        c.pos += name_len;

        t->n_dims = (int32_t)read_u32(&c);
        uint64_t n_elements = 1;
        for (int32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = (int64_t)read_u64(&c);
            n_elements *= t->dims[d];
        }
        t->type = (int32_t)read_u32(&c);
        t->offset = read_u64(&c);
        t->size_bytes = tensor_type_size(t->type, n_elements);

        if (strcmp(t->name, "token_embd.weight") == 0) {
            embd_idx = (int)i;
            printf("Found token_embd.weight: type=%d (F16), dims=[%lld, %lld], %.1f MB\n",
                   t->type, (long long)t->dims[0], (long long)t->dims[1],
                   t->size_bytes / (1024.0 * 1024.0));
        }
    }

    if (embd_idx < 0) {
        fprintf(stderr, "token_embd.weight not found!\n");
        free(tensors);
        free(data);
        return 1;
    }

    /* Compute data_offset (32-byte aligned after tensor info) */
    uint64_t alignment = 32;
    uint64_t data_offset = (c.pos + alignment - 1) & ~(alignment - 1);
    printf("Tensor data offset: %llu\n", (unsigned long long)data_offset);

    /* Compute embedding dimensions.
     * GGUF dims: [inner, outer] = [hidden_size, vocab_size]
     * Data layout: vocab_size rows of hidden_size elements each. */
    tensor_entry_t *embd = &tensors[embd_idx];
    int32_t dim = (int32_t)embd->dims[0];          /* hidden_size = 2560 */
    int32_t vocab_size = (int32_t)embd->dims[1];    /* vocab_size = 128256 */
    int64_t n_elements = (int64_t)vocab_size * dim;

    printf("\nQuantizing embedding: %d × %d = %lld elements\n", vocab_size, dim,
           (long long)n_elements);
    printf("F16 size: %.1f MB → INT8 size: %.1f MB + scales %.1f MB = %.1f MB\n",
           n_elements * 2.0 / 1024 / 1024,
           n_elements * 1.0 / 1024 / 1024,
           vocab_size * 4.0 / 1024 / 1024,
           (n_elements + vocab_size * 4.0) / 1024 / 1024);

    /* Quantize F16 → INT8 */
    const uint16_t *f16_data = (const uint16_t *)&data[data_offset + embd->offset];
    int8_t *int8_data = (int8_t *)malloc(n_elements);
    float *scales = (float *)malloc(vocab_size * sizeof(float));
    if (!int8_data || !scales) {
        fprintf(stderr, "OOM for quantization\n");
        free(data); return 1;
    }

    double total_mse = 0;
    for (int32_t r = 0; r < vocab_size; r++) {
        const uint16_t *row = &f16_data[(int64_t)r * dim];
        int8_t *out_row = &int8_data[(int64_t)r * dim];

        float max_abs = 0.0f;
        for (int32_t j = 0; j < dim; j++) {
            float v = f16_to_f32(row[j]);
            float av = v > 0 ? v : -v;
            if (av > max_abs) max_abs = av;
        }

        scales[r] = max_abs / 127.0f;

        if (max_abs == 0.0f) {
            memset(out_row, 0, dim);
        } else {
            float inv_scale = 127.0f / max_abs;
            for (int32_t j = 0; j < dim; j++) {
                float v = f16_to_f32(row[j]);
                int32_t q = (int32_t)roundf(v * inv_scale);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                out_row[j] = (int8_t)q;

                float dq = q * scales[r];
                double d = v - dq;
                total_mse += d * d;
            }
        }

        if (r % 20000 == 0) {
            fprintf(stderr, "\rQuantizing: %d/%d (%.0f%%)", r, vocab_size,
                    100.0 * r / vocab_size);
        }
    }
    fprintf(stderr, "\rQuantizing: done.                    \n");

    double rmse = sqrt(total_mse / n_elements);
    printf("Quantization RMSE: %.6f\n\n", rmse);

    /* --- Write output GGUF --- */
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { fprintf(stderr, "Cannot create %s\n", argv[2]); free(data); return 1; }

    /* 1. Header: same as input but n_tensors += 1 (for token_embd.scales) */
    write_u32(fout, magic);
    write_u32(fout, version);
    write_u64(fout, n_tensors + 1);  /* +1 for scales tensor */
    write_u64(fout, n_kv);

    /* 2. KV pairs: copy verbatim from input */
    fwrite(&data[kv_start], 1, kv_end - kv_start, fout);

    /* 3. Tensor info section */
    /* Compute new sizes and offsets.
     * New embedding size: n_elements * 1 (INT8)
     * New scales tensor: vocab_size * 4 (F32)
     * Size difference: old_embd_size - new_embd_size */
    uint64_t old_embd_size = embd->size_bytes;
    uint64_t new_embd_size = (uint64_t)n_elements;  /* INT8: 1 byte each */
    uint64_t scales_size = (uint64_t)vocab_size * 4;  /* F32 scales */
    int64_t size_delta = (int64_t)old_embd_size - (int64_t)new_embd_size - (int64_t)scales_size;

    printf("Size delta: %lld bytes (%.1f MB saved)\n",
           (long long)size_delta, size_delta / (1024.0 * 1024.0));

    /* Sort tensors by offset to determine data layout order */
    int *order = (int *)malloc(n_tensors * sizeof(int));
    for (uint64_t i = 0; i < n_tensors; i++) order[i] = (int)i;
    /* Simple insertion sort by offset */
    for (uint64_t i = 1; i < n_tensors; i++) {
        int key = order[i];
        int64_t j = (int64_t)i - 1;
        while (j >= 0 && tensors[order[j]].offset > tensors[key].offset) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    /* Compute ACTUAL on-disk sizes from offset gaps in the original file.
     * tensor_type_size is wrong for I2_S (doesn't include the appended 4-byte scale).
     * The true size of each tensor is: next_tensor_offset - this_tensor_offset. */
    uint64_t *actual_sizes = (uint64_t *)calloc(n_tensors, sizeof(uint64_t));
    for (uint64_t oi = 0; oi < n_tensors; oi++) {
        int idx = order[oi];
        if (oi + 1 < n_tensors) {
            int next_idx = order[oi + 1];
            actual_sizes[idx] = tensors[next_idx].offset - tensors[idx].offset;
        } else {
            /* Last tensor: size extends to end of file */
            actual_sizes[idx] = (uint64_t)file_size - data_offset - tensors[idx].offset;
        }
    }

    /* Compute new offsets: walk tensors in data order, accumulating.
     * Insert scales tensor right after the embedding. */
    uint64_t *new_offsets = (uint64_t *)calloc(n_tensors, sizeof(uint64_t));
    uint64_t scales_offset = 0;
    uint64_t cur_offset = 0;

    for (uint64_t oi = 0; oi < n_tensors; oi++) {
        int idx = order[oi];
        new_offsets[idx] = cur_offset;

        if (idx == embd_idx) {
            /* Embedding: now INT8 */
            cur_offset += new_embd_size;
            /* Scales tensor goes right after */
            scales_offset = cur_offset;
            cur_offset += scales_size;
        } else {
            cur_offset += actual_sizes[idx];
        }
    }

    /* Write tensor info for all original tensors */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_entry_t *t = &tensors[i];

        /* Write name */
        write_gguf_str(fout, t->name);

        /* Write n_dims + dims */
        write_u32(fout, (uint32_t)t->n_dims);
        for (int32_t d = 0; d < t->n_dims; d++) {
            write_u64(fout, (uint64_t)t->dims[d]);
        }

        /* Write type (I8 for embedding, unchanged for others) */
        if ((int)i == embd_idx) {
            write_u32(fout, 24);  /* GGML_TYPE_I8 */
        } else {
            write_u32(fout, (uint32_t)t->type);
        }

        /* Write new offset */
        write_u64(fout, new_offsets[i]);
    }

    /* Write the new scales tensor info */
    write_gguf_str(fout, "token_embd.scales");
    write_u32(fout, 1);  /* n_dims = 1 */
    write_u64(fout, (uint64_t)vocab_size);  /* dims[0] */
    write_u32(fout, 0);  /* GGML_TYPE_F32 */
    write_u64(fout, scales_offset);

    /* 4. Pad to 32-byte alignment */
    write_padding(fout, 32);

    printf("New data offset: %ld\n", ftell(fout));

    /* 5. Write tensor data */
    for (uint64_t oi = 0; oi < n_tensors; oi++) {
        int idx = order[oi];
        tensor_entry_t *t = &tensors[idx];

        if (idx == embd_idx) {
            /* Write INT8 embedding */
            fwrite(int8_data, 1, n_elements, fout);
            /* Write F32 scales */
            fwrite(scales, sizeof(float), vocab_size, fout);
            fprintf(stderr, "Wrote INT8 embedding + scales (%.1f MB)\n",
                    (n_elements + vocab_size * 4.0) / 1024 / 1024);
        } else {
            /* Copy original data (use actual_sizes to include any appended metadata
             * like the I2_S per-tensor scale that tensor_type_size omits) */
            const uint8_t *src = &data[data_offset + t->offset];
            fwrite(src, 1, actual_sizes[idx], fout);
        }
    }

    long output_size = ftell(fout);
    fclose(fout);

    printf("\nOutput: %s (%.1f MB)\n", argv[2], output_size / (1024.0 * 1024.0));
    printf("Saved: %.1f MB (%.1f%%)\n",
           (file_size - output_size) / (1024.0 * 1024.0),
           100.0 * (file_size - output_size) / file_size);

    free(actual_sizes);
    free(order);
    free(new_offsets);
    free(int8_data);
    free(scales);
    free(tensors);
    free(data);

    return 0;
}
