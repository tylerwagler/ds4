/* mxfp8_lt_emit — synthesize a deterministic type-38 payload for a 128-aligned
 * workhorse shape [out, in], write the raw type-38 bytes and the native
 * ds4q_pack_mxfp8_lt() output, so tests/mxfp8_lt_ref.py can run the numpy
 * reference repack on the SAME raw bytes and the two LT files can be diffed
 * (make test-mxfp8-lt). Proves the native type-41 packer is byte-identical to
 * the retired repack_mxfp8_lt.py transform. CPU-only; no model, no GPU. */
#include "quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s OUT IN RAW38_FILE LT_FILE\n", argv[0]);
        return 2;
    }
    const int64_t out = atoll(argv[1]);
    const int64_t in = atoll(argv[2]);
    if (out <= 0 || in <= 0 || in % 32 != 0) {
        fprintf(stderr, "bad shape out=%lld in=%lld (in must be a positive multiple of 32)\n",
                (long long)out, (long long)in);
        return 2;
    }
    const int64_t kb = in / 32;
    const size_t raw_bytes = (size_t)out * (size_t)kb * 33;

    uint8_t *raw = malloc(raw_bytes);
    if (!raw) { perror("malloc"); return 1; }
    /* Deterministic PRNG fill (SplitMix-ish). Any byte works for the packer's
     * byte-move; only the block's E8M0 byte is kept in [1,254] to mirror the
     * type-38 writer's clamp. */
    uint64_t s = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < raw_bytes; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        raw[i] = (uint8_t)(s >> 40);
    }
    for (int64_t b = 0; b < out * kb; b++) {
        raw[(size_t)b * 33] = (uint8_t)(1 + (raw[(size_t)b * 33] % 253));
    }

    const size_t lt_bytes = ds4q_mxfp8_lt_bytes(out, in);
    uint8_t *lt = malloc(lt_bytes);
    if (!lt) { perror("malloc"); return 1; }
    ds4q_pack_mxfp8_lt(raw, lt, out, in);

    FILE *f = fopen(argv[3], "wb");
    if (!f) { perror(argv[3]); return 1; }
    if (fwrite(raw, 1, raw_bytes, f) != raw_bytes) { perror("fwrite raw"); return 1; }
    fclose(f);
    f = fopen(argv[4], "wb");
    if (!f) { perror(argv[4]); return 1; }
    if (fwrite(lt, 1, lt_bytes, f) != lt_bytes) { perror("fwrite lt"); return 1; }
    fclose(f);

    fprintf(stderr, "emit: out=%lld in=%lld raw=%zu lt=%zu\n",
            (long long)out, (long long)in, raw_bytes, lt_bytes);
    free(raw);
    free(lt);
    return 0;
}
