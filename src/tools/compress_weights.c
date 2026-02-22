/*
 * GGUF Weight Compressor: I2_S → TRIT5
 *
 * Reads a GGUF model file and re-encodes all I2_S (2-bit ternary) tensors
 * to TRIT5 format (5 ternary weights per byte, base-3 encoding).
 *
 * I2_S: 2 bits/weight, 4 weights/byte → 497 MB for 2B weights
 * TRIT5: ~1.6 bits/weight, 5 weights/byte → 398 MB (saves ~99 MB)
 *
 * Lossless: decoded ternary values are identical.
 *
 * Usage: ./compress_weights <input.gguf> <output.gguf>
 */

#include "../inference/gguf.h"
#include "../inference/weight_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- Minimal binary reader --- */

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
} cursor_t;

static uint32_t rd_u32(cursor_t *c) {
    uint32_t v; memcpy(&v, &c->data[c->pos], 4); c->pos += 4; return v;
}
static uint64_t rd_u64(cursor_t *c) {
    uint64_t v; memcpy(&v, &c->data[c->pos], 8); c->pos += 8; return v;
}

static void skip_str(cursor_t *c) {
    uint64_t len = rd_u64(c);
    c->pos += len;
}

static void skip_value(cursor_t *c, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: c->pos += 1; break;
        case 2: case 3: c->pos += 2; break;
        case 4: case 5: case 6: c->pos += 4; break;
        case 10: case 11: case 12: c->pos += 8; break;
        case 8: skip_str(c); break;
        case 9: {
            uint32_t arr_type = rd_u32(c);
            uint64_t arr_len = rd_u64(c);
            for (uint64_t i = 0; i < arr_len; i++) skip_value(c, arr_type);
            break;
        }
    }
}

/* --- Binary write helpers --- */

static void wr_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wr_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }

static void wr_gguf_str(FILE *f, const char *s) {
    uint64_t len = strlen(s);
    wr_u64(f, len);
    fwrite(s, 1, len, f);
}

static void wr_padding(FILE *f, uint64_t alignment) {
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

/* --- Tensor info --- */

typedef struct {
    char     name[256];
    int32_t  n_dims;
    int64_t  dims[4];
    int32_t  type;
    uint64_t offset;
} tensor_entry_t;

/* --- TRIT5 encoder --- */

/*
 * Encode int8 ternary values {-1, 0, +1} to TRIT5 packed bytes.
 * Maps: -1→0, 0→1, +1→2. Then byte = t0*81 + t1*27 + t2*9 + t3*3 + t4.
 * Returns number of packed bytes written.
 */
static int64_t trit5_encode(const int8_t *weights, uint8_t *out, int64_t n_elements) {
    int64_t n_full = n_elements / 5;
    int64_t tail = n_elements % 5;

    for (int64_t i = 0; i < n_full; i++) {
        const int8_t *w = &weights[i * 5];
        uint8_t byte = (uint8_t)(
            (w[0] + 1) * 81 +
            (w[1] + 1) * 27 +
            (w[2] + 1) * 9 +
            (w[3] + 1) * 3 +
            (w[4] + 1)
        );
        out[i] = byte;
    }

    /* Encode tail (< 5 weights) */
    if (tail > 0) {
        int8_t tmp[5] = {0, 0, 0, 0, 0};  /* pad with zeros (→ code 1) */
        for (int64_t j = 0; j < tail; j++) {
            tmp[j] = weights[n_full * 5 + j];
        }
        uint8_t byte = (uint8_t)(
            (tmp[0] + 1) * 81 +
            (tmp[1] + 1) * 27 +
            (tmp[2] + 1) * 9 +
            (tmp[3] + 1) * 3 +
            (tmp[4] + 1)
        );
        out[n_full] = byte;
    }

    return (n_elements + 4) / 5;
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
    uint32_t magic = rd_u32(&c);
    uint32_t version = rd_u32(&c);
    uint64_t n_tensors = rd_u64(&c);
    uint64_t n_kv = rd_u64(&c);

    if (magic != 0x46554747) {
        fprintf(stderr, "Bad GGUF magic\n");
        free(data); return 1;
    }

    printf("GGUF v%u: %llu tensors, %llu KV pairs\n",
           version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    /* Skip KV pairs (copy raw bytes later) */
    size_t kv_start = c.pos;
    for (uint64_t i = 0; i < n_kv; i++) {
        skip_str(&c);
        uint32_t vtype = rd_u32(&c);
        skip_value(&c, vtype);
    }
    size_t kv_end = c.pos;

    /* Parse tensor info */
    tensor_entry_t *tensors = (tensor_entry_t *)calloc(n_tensors, sizeof(tensor_entry_t));
    int n_i2s = 0;

    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_entry_t *t = &tensors[i];
        uint64_t name_len = rd_u64(&c);
        if (name_len >= sizeof(t->name)) name_len = sizeof(t->name) - 1;
        memcpy(t->name, &c.data[c.pos], name_len);
        t->name[name_len] = '\0';
        c.pos += name_len;

        t->n_dims = (int32_t)rd_u32(&c);
        for (int32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = (int64_t)rd_u64(&c);
        }
        t->type = (int32_t)rd_u32(&c);
        t->offset = rd_u64(&c);

        if (t->type == GGML_TYPE_I2_S) n_i2s++;
    }

    uint64_t alignment = 32;
    uint64_t data_offset = (c.pos + alignment - 1) & ~(alignment - 1);

    printf("Found %d I2_S tensors to compress\n", n_i2s);
    printf("Original data offset: %llu\n\n", (unsigned long long)data_offset);

    /* Sort tensors by offset */
    int *order = (int *)malloc(n_tensors * sizeof(int));
    for (uint64_t i = 0; i < n_tensors; i++) order[i] = (int)i;
    for (uint64_t i = 1; i < n_tensors; i++) {
        int key = order[i];
        int64_t j = (int64_t)i - 1;
        while (j >= 0 && tensors[order[j]].offset > tensors[key].offset) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    /* Compute actual on-disk sizes from offset gaps */
    uint64_t *actual_sizes = (uint64_t *)calloc(n_tensors, sizeof(uint64_t));
    for (uint64_t oi = 0; oi < n_tensors; oi++) {
        int idx = order[oi];
        if (oi + 1 < n_tensors) {
            int next_idx = order[oi + 1];
            actual_sizes[idx] = tensors[next_idx].offset - tensors[idx].offset;
        } else {
            actual_sizes[idx] = (uint64_t)file_size - data_offset - tensors[idx].offset;
        }
    }

    /* For each I2_S tensor, compute the TRIT5 packed size.
     * TRIT5: ceil(n/5) bytes of packed data + 4 bytes for scale.
     * I2_S:  ceil(n/4) bytes of packed data + 4 bytes for scale.
     * The actual_size includes the scale, so: actual_i2s = ceil(n/4) + 4 */

    /* Compute new offsets */
    uint64_t *new_offsets = (uint64_t *)calloc(n_tensors, sizeof(uint64_t));
    uint64_t *new_sizes = (uint64_t *)calloc(n_tensors, sizeof(uint64_t));
    uint64_t cur_offset = 0;
    int64_t total_saved = 0;

    for (uint64_t oi = 0; oi < n_tensors; oi++) {
        int idx = order[oi];
        new_offsets[idx] = cur_offset;

        if (tensors[idx].type == GGML_TYPE_I2_S) {
            /* Compute n_elements from dims */
            uint64_t n_el = 1;
            for (int32_t d = 0; d < tensors[idx].n_dims; d++)
                n_el *= tensors[idx].dims[d];

            uint64_t trit5_packed = (n_el + 4) / 5;
            uint64_t trit5_total = trit5_packed + 4;  /* + scale */
            new_sizes[idx] = trit5_total;
            cur_offset += trit5_total;
            total_saved += (int64_t)actual_sizes[idx] - (int64_t)trit5_total;
        } else {
            new_sizes[idx] = actual_sizes[idx];
            cur_offset += actual_sizes[idx];
        }
    }

    printf("Weight compression: saves %lld bytes (%.1f MB)\n",
           (long long)total_saved, total_saved / (1024.0 * 1024.0));

    /* --- Write output GGUF --- */
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { fprintf(stderr, "Cannot create %s\n", argv[2]); free(data); return 1; }

    /* 1. Header (same tensor count) */
    wr_u32(fout, magic);
    wr_u32(fout, version);
    wr_u64(fout, n_tensors);
    wr_u64(fout, n_kv);

    /* 2. KV pairs: copy verbatim */
    fwrite(&data[kv_start], 1, kv_end - kv_start, fout);

    /* 3. Tensor info */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_entry_t *t = &tensors[i];
        wr_gguf_str(fout, t->name);
        wr_u32(fout, (uint32_t)t->n_dims);
        for (int32_t d = 0; d < t->n_dims; d++) {
            wr_u64(fout, (uint64_t)t->dims[d]);
        }
        /* Change type for I2_S tensors */
        if (t->type == GGML_TYPE_I2_S) {
            wr_u32(fout, GGML_TYPE_TRIT5);
        } else {
            wr_u32(fout, (uint32_t)t->type);
        }
        wr_u64(fout, new_offsets[i]);
    }

    /* 4. Pad to alignment */
    wr_padding(fout, 32);
    printf("New data offset: %ld\n", ftell(fout));

    /* 5. Write tensor data */
    int tensors_compressed = 0;
    int64_t bytes_saved_verify = 0;

    for (uint64_t oi = 0; oi < n_tensors; oi++) {
        int idx = order[oi];
        tensor_entry_t *t = &tensors[idx];

        if (t->type == GGML_TYPE_I2_S) {
            /* Decode I2_S → int8 → encode TRIT5 */
            uint64_t n_el = 1;
            for (int32_t d = 0; d < t->n_dims; d++) n_el *= t->dims[d];

            const uint8_t *src = &data[data_offset + t->offset];

            /* Decode I2_S to int8 */
            int8_t *raw = (int8_t *)malloc(n_el);
            if (!raw) { fprintf(stderr, "OOM\n"); fclose(fout); return 1; }
            i2s_decode(src, raw, n_el);

            /* Read original per-tensor scale */
            int64_t i2s_packed = (n_el + 3) / 4;
            float scale;
            memcpy(&scale, &src[i2s_packed], sizeof(float));

            /* Encode to TRIT5 */
            int64_t trit5_packed = (n_el + 4) / 5;
            uint8_t *packed = (uint8_t *)calloc(trit5_packed, 1);
            if (!packed) { fprintf(stderr, "OOM\n"); free(raw); fclose(fout); return 1; }
            trit5_encode(raw, packed, n_el);

            /* Verify roundtrip */
            int8_t *verify = (int8_t *)malloc(n_el);
            trit5_decode(packed, verify, n_el);
            int mismatch = 0;
            for (uint64_t j = 0; j < n_el; j++) {
                if (raw[j] != verify[j]) { mismatch = 1; break; }
            }
            free(verify);
            if (mismatch) {
                fprintf(stderr, "ERROR: roundtrip mismatch for %s!\n", t->name);
                free(raw); free(packed); fclose(fout); return 1;
            }

            /* Write TRIT5 data + scale */
            fwrite(packed, 1, trit5_packed, fout);
            fwrite(&scale, sizeof(float), 1, fout);

            bytes_saved_verify += (int64_t)actual_sizes[idx] - (trit5_packed + 4);
            tensors_compressed++;

            if (tensors_compressed % 30 == 0 || tensors_compressed == n_i2s) {
                fprintf(stderr, "\rCompressed %d/%d tensors", tensors_compressed, n_i2s);
            }

            free(packed);
            free(raw);
        } else {
            /* Copy non-I2_S data verbatim */
            const uint8_t *src = &data[data_offset + t->offset];
            fwrite(src, 1, actual_sizes[idx], fout);
        }
    }
    fprintf(stderr, "\rCompressed %d/%d tensors — done.          \n", tensors_compressed, n_i2s);

    long output_size = ftell(fout);
    fclose(fout);

    printf("\nOutput: %s (%.1f MB)\n", argv[2], output_size / (1024.0 * 1024.0));
    printf("Saved: %.1f MB (%.1f%%)\n",
           (file_size - output_size) / (1024.0 * 1024.0),
           100.0 * (file_size - output_size) / file_size);
    printf("All %d tensors verified lossless roundtrip.\n", tensors_compressed);

    free(new_sizes);
    free(actual_sizes);
    free(order);
    free(new_offsets);
    free(tensors);
    free(data);

    return 0;
}
