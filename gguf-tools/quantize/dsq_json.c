#include "dsq_internal.h"

/* =====
 * Minimal JSON tokenizer
 *
 * Safetensors uses ordinary JSON for the model index and per-shard headers.
 * We only need objects, arrays, strings, and primitive numbers; escaped tensor
 * names do not occur in the files produced by Hugging Face, so strings are
 * copied as raw UTF-8 slices after locating the closing quote.
 */

static int json_add(json_doc *d, json_type type, int start, int end, int parent) {
    if (d->len == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 4096;
        d->v = xrealloc(d->v, (size_t)d->cap * sizeof(d->v[0]));
    }
    int id = d->len++;
    d->v[id] = (json_tok){ .type = type, .start = start, .end = end, .parent = parent, .size = 0 };
    if (parent >= 0) d->v[parent].size++;
    return id;
}

json_doc json_parse_text(const char *js, size_t len) {
    json_doc d = { .js = js, .js_len = (int)len };
    int parent = -1;
    for (int i = 0; i < (int)len; i++) {
        unsigned char c = (unsigned char)js[i];
        if (isspace(c) || c == ':' || c == ',') continue;
        if (c == '{' || c == '[') {
            parent = json_add(&d, c == '{' ? JT_OBJECT : JT_ARRAY, i, -1, parent);
            continue;
        }
        if (c == '}' || c == ']') {
            if (parent < 0) die("bad JSON: unmatched close");
            d.v[parent].end = i + 1;
            parent = d.v[parent].parent;
            continue;
        }
        if (c == '"') {
            int start = i + 1;
            i++;
            bool esc = false;
            for (; i < (int)len; i++) {
                if (esc) {
                    esc = false;
                } else if (js[i] == '\\') {
                    esc = true;
                } else if (js[i] == '"') {
                    break;
                }
            }
            if (i >= (int)len) die("bad JSON: unterminated string");
            json_add(&d, JT_STRING, start, i, parent);
            continue;
        }
        int start = i;
        while (i < (int)len && !isspace((unsigned char)js[i]) &&
               js[i] != ',' && js[i] != ']' && js[i] != '}') {
            i++;
        }
        json_add(&d, JT_PRIMITIVE, start, i, parent);
        i--;
    }
    if (parent != -1) die("bad JSON: unterminated object/array");
    return d;
}

void json_free(json_doc *d) {
    free(d->v);
    memset(d, 0, sizeof(*d));
}

bool json_tok_eq(const json_doc *d, int tok, const char *s) {
    const json_tok *t = &d->v[tok];
    const int n = t->end - t->start;
    return t->type == JT_STRING && (int)strlen(s) == n && memcmp(d->js + t->start, s, (size_t)n) == 0;
}

char *json_strdup_tok(const json_doc *d, int tok) {
    const json_tok *t = &d->v[tok];
    return xstrndup(d->js + t->start, (size_t)(t->end - t->start));
}

static bool json_is_descendant(const json_doc *d, int tok, int parent) {
    for (int p = d->v[tok].parent; p >= 0; p = d->v[p].parent) {
        if (p == parent) return true;
    }
    return false;
}

int json_skip(const json_doc *d, int tok) {
    int i = tok + 1;
    while (i < d->len && json_is_descendant(d, i, tok)) i++;
    return i;
}

int json_obj_get(const json_doc *d, int obj, const char *key) {
    if (obj < 0 || d->v[obj].type != JT_OBJECT) return -1;
    for (int i = obj + 1; i < d->len && d->v[i].parent == obj;) {
        int k = i;
        int v = i + 1;
        if (v >= d->len || d->v[v].parent != obj) return -1;
        if (json_tok_eq(d, k, key)) return v;
        i = json_skip(d, v);
    }
    return -1;
}

int64_t json_i64(const json_doc *d, int tok) {
    char tmp[64];
    const int n = d->v[tok].end - d->v[tok].start;
    if (n <= 0 || n >= (int)sizeof(tmp)) die("bad JSON integer");
    memcpy(tmp, d->js + d->v[tok].start, (size_t)n);
    tmp[n] = '\0';
    return strtoll(tmp, NULL, 10);
}

/* =====
 * Small string hash map
 */

static uint64_t fnv1a_str(const char *s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

void hmap_build(hmap *m, char **keys, int n) {
    int cap = 1;
    while (cap < n * 3) cap <<= 1;
    m->cap = cap ? cap : 2;
    m->slots = xcalloc((size_t)m->cap, sizeof(m->slots[0]));
    for (int i = 0; i < n; i++) {
        uint64_t h = fnv1a_str(keys[i]);
        int p = (int)(h & (uint64_t)(m->cap - 1));
        while (m->slots[p].key) p = (p + 1) & (m->cap - 1);
        m->slots[p].key = keys[i];
        m->slots[p].value = i;
    }
}

int hmap_get(const hmap *m, const char *key) {
    if (!m->slots) return -1;
    uint64_t h = fnv1a_str(key);
    int p = (int)(h & (uint64_t)(m->cap - 1));
    while (m->slots[p].key) {
        if (strcmp(m->slots[p].key, key) == 0) return m->slots[p].value;
        p = (p + 1) & (m->cap - 1);
    }
    return -1;
}

void hmap_free(hmap *m) {
    free(m->slots);
    memset(m, 0, sizeof(*m));
}

