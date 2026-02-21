#ifndef BITNET_TOKENIZER_H
#define BITNET_TOKENIZER_H

#include "gguf.h"
#include <stdint.h>

/*
 * BPE Tokenizer for Llama 3 / BitNet b1.58
 *
 * GPT-2 style byte-level BPE: every byte maps to a printable Unicode
 * character (the "GPT-2 byte encoding"). Vocab and merge rules from GGUF
 * use this encoding. Pretokenization splits text into chunks before BPE.
 */

/* Hash table entry for string -> token ID lookup */
typedef struct vocab_entry {
    char                *key;
    int32_t              id;
    struct vocab_entry  *next;
} vocab_entry_t;

typedef struct {
    vocab_entry_t **buckets;
    int32_t         n_buckets;
} vocab_hash_t;

/* Merge pair -> rank hash table */
typedef struct merge_entry {
    char                *pair_key;   /* "a\xFFb" key */
    int32_t              key_len;
    int32_t              rank;
    struct merge_entry  *next;
} merge_entry_t;

typedef struct {
    merge_entry_t **buckets;
    int32_t         n_buckets;
} merge_hash_t;

/* Tokenizer state */
typedef struct {
    /* Vocab: token_id -> string (owned copies) */
    char          **vocab;
    int32_t        *token_types;
    int32_t         vocab_size;

    /* String -> token ID */
    vocab_hash_t    vocab_hash;

    /* Merge pair -> rank */
    merge_hash_t    merge_hash;
    int32_t         n_merges;

    /* Special tokens */
    int32_t         bos_id;
    int32_t         eos_id;
    int32_t         add_bos;
    int32_t         add_eos;

    /* GPT-2 byte encoding tables */
    int32_t         byte_to_unicode[256];    /* byte -> Unicode code point */
    uint8_t         unicode_to_byte[324];    /* code point -> byte (max cp = 323) */
    uint8_t         byte_to_utf8[256][3];    /* byte -> UTF-8 encoded GPT-2 char */
    uint8_t         byte_to_utf8_len[256];   /* length of UTF-8 encoding */
} tokenizer_t;

int tokenizer_init(tokenizer_t *tok, const gguf_tokenizer_t *gguf_tok);
void tokenizer_free(tokenizer_t *tok);

int32_t tokenizer_encode(const tokenizer_t *tok, const char *text,
                          int32_t *tokens_out, int32_t max_tokens,
                          int32_t add_special);

const char *tokenizer_decode_token(const tokenizer_t *tok, int32_t token_id);

int32_t tokenizer_decode(const tokenizer_t *tok, const int32_t *tokens,
                          int32_t n_tokens, char *buf, int32_t buf_size);

#endif /* BITNET_TOKENIZER_H */
