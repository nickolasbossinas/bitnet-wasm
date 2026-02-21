#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <limits.h>

/* ========== GPT-2 byte encoding ========== */

/*
 * GPT-2 maps each byte 0-255 to a printable Unicode character:
 *   33-126 ('!' to '~')          -> same code point (directly printable)
 *   161-172, 174-255             -> same code point (extended Latin)
 *   0-32, 127, 128-160, 173     -> code points 256+ (remapped to avoid control chars)
 *
 * This ensures all token strings are printable.
 */
static void build_byte_tables(tokenizer_t *tok) {
    /* Direct-mapped ranges */
    int direct[256] = {0};
    for (int b = 33; b <= 126; b++) direct[b] = 1;
    for (int b = 161; b <= 172; b++) direct[b] = 1;
    for (int b = 174; b <= 255; b++) direct[b] = 1;

    int next_cp = 256;
    for (int b = 0; b < 256; b++) {
        if (direct[b]) {
            tok->byte_to_unicode[b] = b;
        } else {
            tok->byte_to_unicode[b] = next_cp++;
        }
    }

    /* Build reverse table */
    memset(tok->unicode_to_byte, 0, sizeof(tok->unicode_to_byte));
    for (int b = 0; b < 256; b++) {
        int cp = tok->byte_to_unicode[b];
        if (cp < 324) {
            tok->unicode_to_byte[cp] = (uint8_t)b;
        }
    }

    /* Build UTF-8 encoding for each byte's GPT-2 character */
    for (int b = 0; b < 256; b++) {
        int cp = tok->byte_to_unicode[b];
        if (cp < 0x80) {
            tok->byte_to_utf8[b][0] = (uint8_t)cp;
            tok->byte_to_utf8_len[b] = 1;
        } else if (cp < 0x800) {
            tok->byte_to_utf8[b][0] = (uint8_t)(0xC0 | (cp >> 6));
            tok->byte_to_utf8[b][1] = (uint8_t)(0x80 | (cp & 0x3F));
            tok->byte_to_utf8_len[b] = 2;
        }
        /* All GPT-2 code points are < 0x800, no 3-byte needed */
    }
}

/* ========== FNV-1a hash ========== */

static uint32_t fnv1a(const char *str) {
    uint32_t h = 0x811c9dc5;
    while (*str) {
        h ^= (uint8_t)*str++;
        h *= 0x01000193;
    }
    return h;
}

static uint32_t fnv1a_bytes(const uint8_t *data, int32_t len) {
    uint32_t h = 0x811c9dc5;
    for (int32_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

/* ========== Vocab hash table ========== */

static int vocab_hash_init(vocab_hash_t *ht, int32_t n_entries) {
    /* Next power of 2 above 2*n_entries */
    int32_t n = 1;
    while (n < n_entries * 2) n <<= 1;
    ht->n_buckets = n;
    ht->buckets = (vocab_entry_t **)calloc(n, sizeof(vocab_entry_t *));
    return ht->buckets ? 0 : -1;
}

static void vocab_hash_free(vocab_hash_t *ht) {
    if (!ht->buckets) return;
    for (int32_t i = 0; i < ht->n_buckets; i++) {
        vocab_entry_t *e = ht->buckets[i];
        while (e) {
            vocab_entry_t *next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }
    free(ht->buckets);
    ht->buckets = NULL;
}

static int vocab_hash_insert(vocab_hash_t *ht, const char *key, int32_t id) {
    uint32_t h = fnv1a(key) & (ht->n_buckets - 1);
    vocab_entry_t *e = (vocab_entry_t *)malloc(sizeof(vocab_entry_t));
    if (!e) return -1;
    e->key = strdup(key);
    if (!e->key) { free(e); return -1; }
    e->id = id;
    e->next = ht->buckets[h];
    ht->buckets[h] = e;
    return 0;
}

static int32_t vocab_hash_lookup(const vocab_hash_t *ht, const char *key) {
    uint32_t h = fnv1a(key) & (ht->n_buckets - 1);
    for (vocab_entry_t *e = ht->buckets[h]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) return e->id;
    }
    return -1;
}

/* ========== Merge hash table ========== */

static int merge_hash_init(merge_hash_t *ht, int32_t n_entries) {
    int32_t n = 1;
    while (n < n_entries * 2) n <<= 1;
    ht->n_buckets = n;
    ht->buckets = (merge_entry_t **)calloc(n, sizeof(merge_entry_t *));
    return ht->buckets ? 0 : -1;
}

static void merge_hash_free(merge_hash_t *ht) {
    if (!ht->buckets) return;
    for (int32_t i = 0; i < ht->n_buckets; i++) {
        merge_entry_t *e = ht->buckets[i];
        while (e) {
            merge_entry_t *next = e->next;
            free(e->pair_key);
            free(e);
            e = next;
        }
    }
    free(ht->buckets);
    ht->buckets = NULL;
}

static int merge_hash_insert(merge_hash_t *ht, const char *a, const char *b, int32_t rank) {
    int32_t la = (int32_t)strlen(a);
    int32_t lb = (int32_t)strlen(b);
    int32_t key_len = la + 1 + lb;
    char *key = (char *)malloc(key_len + 1);
    if (!key) return -1;
    memcpy(key, a, la);
    key[la] = '\xff';
    memcpy(key + la + 1, b, lb);
    key[key_len] = '\0';

    uint32_t h = fnv1a_bytes((const uint8_t *)key, key_len) & (ht->n_buckets - 1);
    merge_entry_t *e = (merge_entry_t *)malloc(sizeof(merge_entry_t));
    if (!e) { free(key); return -1; }
    e->pair_key = key;
    e->key_len = key_len;
    e->rank = rank;
    e->next = ht->buckets[h];
    ht->buckets[h] = e;
    return 0;
}

static int32_t merge_hash_lookup(const merge_hash_t *ht, const char *a, const char *b) {
    int32_t la = (int32_t)strlen(a);
    int32_t lb = (int32_t)strlen(b);
    int32_t key_len = la + 1 + lb;
    char key_buf[512];
    if (key_len >= (int32_t)sizeof(key_buf)) return -1;
    memcpy(key_buf, a, la);
    key_buf[la] = '\xff';
    memcpy(key_buf + la + 1, b, lb);

    uint32_t h = fnv1a_bytes((const uint8_t *)key_buf, key_len) & (ht->n_buckets - 1);
    for (merge_entry_t *e = ht->buckets[h]; e; e = e->next) {
        if (e->key_len == key_len && memcmp(e->pair_key, key_buf, key_len) == 0)
            return e->rank;
    }
    return -1;
}

/* ========== Tokenizer init/free ========== */

int tokenizer_init(tokenizer_t *tok, const gguf_tokenizer_t *gguf_tok) {
    memset(tok, 0, sizeof(*tok));

    tok->vocab_size = gguf_tok->vocab_size;
    tok->n_merges = gguf_tok->n_merges;
    tok->bos_id = gguf_tok->bos_token_id;
    tok->eos_id = gguf_tok->eos_token_id;
    tok->add_bos = gguf_tok->add_bos;
    tok->add_eos = gguf_tok->add_eos;

    /* Build GPT-2 byte tables */
    build_byte_tables(tok);

    /* Copy vocab strings */
    tok->vocab = (char **)calloc(tok->vocab_size, sizeof(char *));
    if (!tok->vocab) return -1;
    for (int32_t i = 0; i < tok->vocab_size; i++) {
        tok->vocab[i] = strdup(gguf_tok->tokens[i]);
        if (!tok->vocab[i]) return -1;
    }

    /* Copy token types if available */
    if (gguf_tok->token_types) {
        tok->token_types = (int32_t *)malloc(tok->vocab_size * sizeof(int32_t));
        if (!tok->token_types) return -1;
        memcpy(tok->token_types, gguf_tok->token_types, tok->vocab_size * sizeof(int32_t));
    }

    /* Build vocab hash */
    if (vocab_hash_init(&tok->vocab_hash, tok->vocab_size) != 0) return -1;
    for (int32_t i = 0; i < tok->vocab_size; i++) {
        if (vocab_hash_insert(&tok->vocab_hash, tok->vocab[i], i) != 0) return -1;
    }

    /* Build merge hash */
    if (gguf_tok->n_merges > 0) {
        if (merge_hash_init(&tok->merge_hash, gguf_tok->n_merges) != 0) return -1;
        for (int32_t i = 0; i < gguf_tok->n_merges; i++) {
            /* Each merge is "tokenA tokenB" — split on first space */
            const char *m = gguf_tok->merges[i];
            const char *sp = strchr(m, ' ');
            if (!sp) continue;

            int32_t la = (int32_t)(sp - m);
            char a_buf[256], b_buf[256];
            if (la >= (int32_t)sizeof(a_buf)) continue;
            memcpy(a_buf, m, la);
            a_buf[la] = '\0';

            const char *b = sp + 1;
            int32_t lb = (int32_t)strlen(b);
            if (lb >= (int32_t)sizeof(b_buf)) continue;
            memcpy(b_buf, b, lb);
            b_buf[lb] = '\0';

            merge_hash_insert(&tok->merge_hash, a_buf, b_buf, i);
        }
    }

    printf("tokenizer: init ok (vocab=%d, merges=%d, bos=%d, eos=%d)\n",
           tok->vocab_size, tok->n_merges, tok->bos_id, tok->eos_id);
    return 0;
}

void tokenizer_free(tokenizer_t *tok) {
    if (tok->vocab) {
        for (int32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    free(tok->token_types);
    vocab_hash_free(&tok->vocab_hash);
    merge_hash_free(&tok->merge_hash);
    memset(tok, 0, sizeof(*tok));
}

/* ========== Pretokenizer ========== */

/*
 * Simplified Llama 3 / tiktoken pretokenizer.
 *
 * Splits text into chunks for independent BPE processing.
 * Rules (greedy, left-to-right):
 *   1. Contractions: 's 't 're 've 'm 'll 'd
 *   2. Letters: optional non-letter/digit + letters+
 *   3. Digits: 1-3 consecutive digits
 *   4. Punctuation: optional space + symbols + optional newlines
 *   5. Newlines: optional whitespace + newlines
 *   6. Whitespace: spaces/tabs
 *
 * Treats bytes >= 0x80 as letters (correct for UTF-8 multibyte).
 */

typedef struct { int32_t start; int32_t len; } chunk_t;

static int is_letter(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

static int is_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static int is_ws(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static int is_newline(uint8_t c) {
    return c == '\r' || c == '\n';
}

/* Check for contraction at position i, return length if matched */
static int match_contraction(const uint8_t *t, int32_t len, int32_t i) {
    if (i >= len || t[i] != '\'') return 0;
    int32_t rem = len - i;
    if (rem >= 3) {
        char c1 = tolower(t[i+1]);
        char c2 = tolower(t[i+2]);
        if (c1 == 'r' && c2 == 'e') return 3; /* 're */
        if (c1 == 'v' && c2 == 'e') return 3; /* 've */
        if (c1 == 'l' && c2 == 'l') return 3; /* 'll */
    }
    if (rem >= 2) {
        char c1 = tolower(t[i+1]);
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    }
    return 0;
}

static int32_t pretokenize(const uint8_t *text, int32_t text_len,
                            chunk_t *chunks, int32_t max_chunks) {
    int32_t n = 0;
    int32_t i = 0;

    while (i < text_len && n < max_chunks) {
        int32_t start = i;

        /* 1. Contractions */
        int clen = match_contraction(text, text_len, i);
        if (clen > 0) {
            chunks[n].start = start;
            chunks[n].len = clen;
            n++;
            i += clen;
            continue;
        }

        /* 2. Letters: [^\r\n\p{L}\p{N}]?\p{L}+ */
        if (is_letter(text[i]) ||
            (!is_letter(text[i]) && !is_digit(text[i]) && !is_newline(text[i]) &&
             i + 1 < text_len && is_letter(text[i+1]))) {
            /* Optional non-letter/digit/newline prefix */
            if (!is_letter(text[i])) i++;
            while (i < text_len && is_letter(text[i])) i++;
            chunks[n].start = start;
            chunks[n].len = i - start;
            n++;
            continue;
        }

        /* 3. Digits: \p{N}{1,3} */
        if (is_digit(text[i])) {
            int32_t count = 0;
            while (i < text_len && is_digit(text[i]) && count < 3) {
                i++;
                count++;
            }
            chunks[n].start = start;
            chunks[n].len = i - start;
            n++;
            continue;
        }

        /* 4. Newlines: \s*[\r\n]+ */
        if (is_newline(text[i])) {
            while (i < text_len && is_newline(text[i])) i++;
            chunks[n].start = start;
            chunks[n].len = i - start;
            n++;
            continue;
        }

        /* 5. Punctuation/symbols: ' ?[^\s\p{L}\p{N}]+[\r\n]*' */
        if (!is_ws(text[i]) && !is_letter(text[i]) && !is_digit(text[i])) {
            while (i < text_len && !is_ws(text[i]) && !is_letter(text[i]) && !is_digit(text[i])) {
                i++;
            }
            /* Consume trailing newlines */
            while (i < text_len && is_newline(text[i])) i++;
            chunks[n].start = start;
            chunks[n].len = i - start;
            n++;
            continue;
        }

        /* 6. Whitespace: \s+ */
        if (is_ws(text[i])) {
            while (i < text_len && is_ws(text[i]) && !is_newline(text[i])) i++;
            chunks[n].start = start;
            chunks[n].len = i - start;
            n++;
            continue;
        }

        /* Fallback: single character */
        i++;
        chunks[n].start = start;
        chunks[n].len = i - start;
        n++;
    }
    return n;
}

/* ========== BPE encoding ========== */

#define MAX_BPE_TOKENS 1024

typedef struct {
    char    str[128];
    int32_t len;
} bpe_tok_t;

static int32_t bpe_encode_chunk(const tokenizer_t *tok,
                                 const uint8_t *text, int32_t text_len,
                                 int32_t *ids_out, int32_t max_ids) {
    /* Convert each byte to its GPT-2 UTF-8 representation */
    bpe_tok_t tokens[MAX_BPE_TOKENS];
    int32_t n_tokens = 0;

    for (int32_t i = 0; i < text_len && n_tokens < MAX_BPE_TOKENS; i++) {
        uint8_t b = text[i];
        int32_t ulen = tok->byte_to_utf8_len[b];
        memcpy(tokens[n_tokens].str, tok->byte_to_utf8[b], ulen);
        tokens[n_tokens].str[ulen] = '\0';
        tokens[n_tokens].len = ulen;
        n_tokens++;
    }

    /* Iterative BPE merging */
    while (n_tokens > 1) {
        int32_t best_rank = INT32_MAX;
        int32_t best_pos = -1;

        for (int32_t i = 0; i < n_tokens - 1; i++) {
            int32_t rank = merge_hash_lookup(&tok->merge_hash,
                                              tokens[i].str, tokens[i+1].str);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_pos = i;
            }
        }

        if (best_pos < 0) break;

        /* Merge tokens[best_pos] and tokens[best_pos+1] */
        int32_t new_len = tokens[best_pos].len + tokens[best_pos+1].len;
        if (new_len >= (int32_t)sizeof(tokens[0].str)) break;

        memcpy(tokens[best_pos].str + tokens[best_pos].len,
               tokens[best_pos+1].str, tokens[best_pos+1].len);
        tokens[best_pos].str[new_len] = '\0';
        tokens[best_pos].len = new_len;

        /* Shift remaining tokens left */
        for (int32_t i = best_pos + 1; i < n_tokens - 1; i++) {
            tokens[i] = tokens[i+1];
        }
        n_tokens--;
    }

    /* Convert token strings to IDs */
    int32_t n_ids = 0;
    for (int32_t i = 0; i < n_tokens && n_ids < max_ids; i++) {
        int32_t id = vocab_hash_lookup(&tok->vocab_hash, tokens[i].str);
        if (id >= 0) {
            ids_out[n_ids++] = id;
        } else {
            /* Byte-level fallback: shouldn't happen with correct BPE vocab */
            fprintf(stderr, "tokenizer: unmappable token \"%s\"\n", tokens[i].str);
            return -1;
        }
    }
    return n_ids;
}

/* ========== Public encode/decode ========== */

int32_t tokenizer_encode(const tokenizer_t *tok, const char *text,
                          int32_t *tokens_out, int32_t max_tokens,
                          int32_t add_special) {
    int32_t n_out = 0;
    int32_t text_len = (int32_t)strlen(text);

    if (add_special && tok->add_bos && tok->bos_id >= 0) {
        if (n_out >= max_tokens) return -1;
        tokens_out[n_out++] = tok->bos_id;
    }

    if (text_len > 0) {
        chunk_t chunks[8192];
        int32_t n_chunks = pretokenize((const uint8_t *)text, text_len, chunks, 8192);

        for (int32_t c = 0; c < n_chunks; c++) {
            int32_t chunk_ids[MAX_BPE_TOKENS];
            int32_t n = bpe_encode_chunk(tok,
                (const uint8_t *)text + chunks[c].start, chunks[c].len,
                chunk_ids, MAX_BPE_TOKENS);
            if (n < 0) return -1;

            for (int32_t i = 0; i < n; i++) {
                if (n_out >= max_tokens) return -1;
                tokens_out[n_out++] = chunk_ids[i];
            }
        }
    }

    if (add_special && tok->add_eos && tok->eos_id >= 0) {
        if (n_out >= max_tokens) return -1;
        tokens_out[n_out++] = tok->eos_id;
    }

    return n_out;
}

const char *tokenizer_decode_token(const tokenizer_t *tok, int32_t token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return NULL;
    return tok->vocab[token_id];
}

int32_t tokenizer_decode(const tokenizer_t *tok, const int32_t *tokens,
                          int32_t n_tokens, char *buf, int32_t buf_size) {
    int32_t pos = 0;

    for (int32_t t = 0; t < n_tokens; t++) {
        const char *s = tokenizer_decode_token(tok, tokens[t]);
        if (!s) continue;

        /* Decode GPT-2 byte encoding: convert each UTF-8 character back to
         * its original byte using the unicode_to_byte table */
        const uint8_t *p = (const uint8_t *)s;
        while (*p) {
            int32_t cp;
            int32_t nbytes;

            if (*p < 0x80) {
                cp = *p;
                nbytes = 1;
            } else if ((*p & 0xE0) == 0xC0) {
                cp = (*p & 0x1F) << 6;
                cp |= (p[1] & 0x3F);
                nbytes = 2;
            } else if ((*p & 0xF0) == 0xE0) {
                cp = (*p & 0x0F) << 12;
                cp |= (p[1] & 0x3F) << 6;
                cp |= (p[2] & 0x3F);
                nbytes = 3;
            } else {
                cp = (*p & 0x07) << 18;
                cp |= (p[1] & 0x3F) << 12;
                cp |= (p[2] & 0x3F) << 6;
                cp |= (p[3] & 0x3F);
                nbytes = 4;
            }

            if (cp < 324) {
                /* Known GPT-2 byte encoding — convert back to original byte */
                if (pos >= buf_size - 1) return -1;
                buf[pos++] = (char)tok->unicode_to_byte[cp];
            } else {
                /* Outside GPT-2 range — pass through UTF-8 bytes as-is */
                if (pos + nbytes >= buf_size - 1) return -1;
                memcpy(buf + pos, p, nbytes);
                pos += nbytes;
            }
            p += nbytes;
        }
    }

    buf[pos] = '\0';
    return pos;
}
