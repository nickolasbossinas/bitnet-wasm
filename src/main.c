#include "inference/gguf.h"
#include "inference/model.h"
#include "inference/weight_loader.h"
#include "inference/tokenizer.h"
#include "inference/generate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * bitnet-cli: Standalone text generation tool.
 *
 * Usage: bitnet-cli <model.gguf> [options]
 *   -p "prompt"       Input prompt (default: "Hello")
 *   -n 32             Max tokens to generate (default: 32)
 *   -t 0.0            Temperature (default: 0.0 = greedy)
 *   --top-p 0.9       Top-p sampling (default: 0.9)
 *   --layers N        Load only N layers (default: all)
 *   --seed N          RNG seed (default: 42)
 */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [options]\n"
        "  -p \"prompt\"       Input prompt (default: \"Hello\")\n"
        "  -n 32             Max tokens to generate (default: 32)\n"
        "  -t 0.0            Temperature (default: 0.0 = greedy)\n"
        "  --top-p 0.9       Top-p sampling (default: 0.9)\n"
        "  --layers N        Load only N layers (default: all)\n"
        "  --seed N          RNG seed (default: 42)\n",
        prog);
}

/* Streaming callback: print each token piece as it's generated */
static void stream_token(const char *piece, int32_t token_id, void *user_data) {
    (void)token_id;
    (void)user_data;
    printf("%s", piece);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Defaults */
    const char *model_path = argv[1];
    const char *prompt = "Hello";
    int32_t max_tokens = 32;
    float temperature = 0.0f;
    float top_p = 0.9f;
    int32_t n_layers = -1;  /* all */
    uint32_t seed = 42;

    /* Parse args */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint32_t)atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* 1. Read GGUF file */
    fprintf(stderr, "Loading model: %s\n", model_path);
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", model_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    fprintf(stderr, "File size: %.2f MB\n", file_size / (1024.0 * 1024.0));

    uint8_t *file_data = (uint8_t *)malloc(file_size);
    if (!file_data) {
        fprintf(stderr, "Failed to allocate %ld bytes for file\n", file_size);
        fclose(f);
        return 1;
    }
    fread(file_data, 1, file_size, f);
    fclose(f);

    /* 2. Parse GGUF metadata */
    gguf_context_t gguf;
    if (gguf_parse(&gguf, file_data, file_size) != 0) {
        fprintf(stderr, "Failed to parse GGUF\n");
        free(file_data);
        return 1;
    }
    fprintf(stderr, "Model: %d layers, %d hidden, %d vocab\n",
            gguf.n_layers, gguf.hidden_size, gguf.vocab_size);

    /* 3. Init tokenizer */
    tokenizer_t tok;
    if (tokenizer_init(&tok, &gguf.tokenizer) != 0) {
        fprintf(stderr, "Failed to init tokenizer\n");
        gguf_free(&gguf);
        free(file_data);
        return 1;
    }
    fprintf(stderr, "Tokenizer: %d vocab, %d merges\n",
            tok.vocab_size, tok.n_merges);

    /* 4. Alloc model and load weights */
    model_t model;
    memset(&model, 0, sizeof(model));
    model_config_from_gguf(&model.config, &gguf);

    int32_t layers_to_load = n_layers;
    if (layers_to_load > 0) {
        model.config.n_layers = layers_to_load;
        fprintf(stderr, "Loading %d of %d layers\n", layers_to_load, gguf.n_layers);
    } else {
        layers_to_load = model.config.n_layers;
    }

    if (model_alloc(&model) != 0) {
        fprintf(stderr, "Failed to allocate model\n");
        tokenizer_free(&tok);
        gguf_free(&gguf);
        free(file_data);
        return 1;
    }

    fprintf(stderr, "Loading weights...\n");
    if (model_load_weights(&model, &gguf, file_data, layers_to_load) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        model_free(&model);
        tokenizer_free(&tok);
        gguf_free(&gguf);
        free(file_data);
        return 1;
    }

    /* 5. Free GGUF file buffer (weights are copied out) */
    gguf_free(&gguf);
    free(file_data);
    file_data = NULL;
    fprintf(stderr, "Model loaded. Freed GGUF buffer.\n\n");

    /* 6. Generate */
    generate_params_t params = {
        .temperature = temperature,
        .top_p = top_p,
        .max_tokens = max_tokens,
        .seed = seed,
    };

    fprintf(stderr, "Prompt: \"%s\"\n", prompt);
    fprintf(stderr, "Params: temp=%.2f, top_p=%.2f, max_tokens=%d, seed=%u\n\n",
            temperature, top_p, max_tokens, seed);

    int n = generate(&model, &tok, prompt, &params, stream_token, NULL);
    printf("\n");

    if (n < 0) {
        fprintf(stderr, "Generation failed\n");
    } else {
        fprintf(stderr, "\nGenerated %d tokens.\n", n);
    }

    /* 7. Cleanup */
    /* Free layer TL1 weight allocations */
    for (int32_t l = 0; l < model.config.n_layers; l++) {
        layer_weights_t *lw = &model.layers[l];
        free(lw->attn_q.indices);    free(lw->attn_q.indices_col);
        free(lw->attn_k.indices);    free(lw->attn_k.indices_col);
        free(lw->attn_v.indices);    free(lw->attn_v.indices_col);
        free(lw->attn_o.indices);    free(lw->attn_o.indices_col);
        free(lw->ffn_gate.indices);  free(lw->ffn_gate.indices_col);
        free(lw->ffn_up.indices);    free(lw->ffn_up.indices_col);
        free(lw->ffn_down.indices);  free(lw->ffn_down.indices_col);
    }
    model_free(&model);
    tokenizer_free(&tok);

    return (n >= 0) ? 0 : 1;
}
