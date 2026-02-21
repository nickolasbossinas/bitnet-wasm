#include "../inference/gguf.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * Quick GGUF inspector: parse and print all tensor info.
 * Usage: inspect_gguf <path-to-gguf>
 */

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

    printf("File: %s (%.2f MB)\n", argv[1], file_size / (1024.0 * 1024.0));

    /* Read entire file — for inspection we only need first few MB but
     * let's read it all to validate data_offset computation */
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

    printf("\n--- Model Config ---\n");
    printf("layers=%d, hidden=%d, intermediate=%d\n",
           ctx.n_layers, ctx.hidden_size, ctx.intermediate_size);
    printf("heads=%d, kv_heads=%d, head_dim=%d, vocab=%d\n",
           ctx.n_heads, ctx.n_kv_heads, ctx.head_dim, ctx.vocab_size);

    printf("\n--- Tensors (%llu total) ---\n", (unsigned long long)ctx.n_tensors);
    printf("%-45s  %-8s  %-30s  %12s  %12s\n",
           "Name", "Type", "Dimensions", "Offset", "Size (KB)");
    printf("%-45s  %-8s  %-30s  %12s  %12s\n",
           "----", "----", "----------", "------", "---------");

    for (uint64_t i = 0; i < ctx.n_tensors; i++) {
        gguf_tensor_info_t *t = &ctx.tensors[i];

        /* Format dimensions */
        char dims_str[64] = "";
        int pos = 0;
        pos += snprintf(dims_str + pos, sizeof(dims_str) - pos, "[");
        for (int d = 0; d < t->n_dims; d++) {
            if (d > 0) pos += snprintf(dims_str + pos, sizeof(dims_str) - pos, ", ");
            pos += snprintf(dims_str + pos, sizeof(dims_str) - pos, "%lld", (long long)t->dims[d]);
        }
        snprintf(dims_str + pos, sizeof(dims_str) - pos, "]");

        printf("%-45s  %-8s(%2d)  %-30s  %12llu  %12.1f\n",
               t->name,
               ggml_type_name(t->type),
               t->type,
               dims_str,
               (unsigned long long)t->offset,
               t->size_bytes / 1024.0);
    }

    printf("\n--- Summary ---\n");
    printf("Data offset: %llu\n", (unsigned long long)ctx.data_offset);
    printf("File size:   %ld\n", file_size);

    gguf_free(&ctx);
    free(data);
    return 0;
}
