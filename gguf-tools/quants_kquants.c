#include "quants_internal.h"


static void ds4q_write_q2_k_block_ref(const float *x, uint8_t *y) {
    enum { scales_off = 0, qs_off = 16, d_off = 80, dmin_off = 82 };
    const float q4scale = 15.0f;
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float weights[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; j++) {
        for (int l = 0; l < 16; l++) weights[l] = fabsf(x[16 * j + l]);
        scales[j] = ds4q_make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j,
                                           &mins[j], Laux, -0.5f, 0.1f, 15, true);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    uint16_t hd, hmin;
    if (max_scale > 0) {
        float iscale = q4scale / max_scale;
        for (int j = 0; j < QK_K / 16; j++) scales_out[j] = ds4q_nearest_int(iscale * scales[j]);
        hd = ds4q_f32_to_f16(max_scale / q4scale);
    } else {
        memset(scales_out, 0, QK_K / 16);
        hd = ds4q_f32_to_f16(0.0f);
    }
    if (max_min > 0) {
        float iscale = q4scale / max_min;
        for (int j = 0; j < QK_K / 16; j++) scales_out[j] |= ds4q_nearest_int(iscale * mins[j]) << 4;
        hmin = ds4q_f32_to_f16(max_min / q4scale);
    } else {
        hmin = ds4q_f32_to_f16(0.0f);
    }
    memcpy(y + d_off, &hd, sizeof(hd));
    memcpy(y + dmin_off, &hmin, sizeof(hmin));

    for (int j = 0; j < QK_K / 16; j++) {
        const float d = ds4q_f16_to_f32(hd) * (scales_out[j] & 0xF);
        if (!d) continue;
        const float dm = ds4q_f16_to_f32(hmin) * (scales_out[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int((x[16 * j + ii] + dm) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(3, l));
            L[16 * j + ii] = l;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            qs_out[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static void ds4q_write_q2_k_block_weighted(const float *x, uint8_t *y, const float *quant_weights) {
    enum { scales_off = 0, qs_off = 16, d_off = 80, dmin_off = 82 };
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    float sw[QK_K / 16];
    float weight[16];
    uint8_t Ls[QK_K / 16], Lm[QK_K / 16];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    memset(sw, 0, sizeof(sw));
    float sumx2 = 0;
    for (int j = 0; j < QK_K; j++) sumx2 += x[j] * x[j];
    float sigma2 = sumx2 / QK_K;
    for (int j = 0; j < QK_K / 16; j++) {
        const float *qw = quant_weights + 16 * j;
        for (int l = 0; l < 16; l++) weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
        for (int l = 0; l < QK_K / 16; l++) sw[j] += weight[l];
        scales[j] = ds4q_make_qkx3_quants(16, 3, x + 16 * j, weight, L + 16 * j,
                                           &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float dm = ds4q_make_qp_quants(QK_K / 16, 15, scales, Ls, sw);
    float mm = ds4q_make_qp_quants(QK_K / 16, 15, mins, Lm, sw);
    uint16_t hd = ds4q_f32_to_f16(dm);
    uint16_t hmin = ds4q_f32_to_f16(mm);
    memcpy(y + d_off, &hd, sizeof(hd));
    memcpy(y + dmin_off, &hmin, sizeof(hmin));
    dm = ds4q_f16_to_f32(hd);
    mm = ds4q_f16_to_f32(hmin);

    for (int j = 0; j < QK_K / 16; j++) scales_out[j] = Ls[j] | (Lm[j] << 4);

    for (int j = 0; j < QK_K / 16; j++) {
        const float d = dm * (scales_out[j] & 0xF);
        if (!d) continue;
        const float m = mm * (scales_out[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int((x[16 * j + ii] + m) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(3, l));
            L[16 * j + ii] = l;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            qs_out[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

size_t ds4q_quantize_q2_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q2_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q2_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q2_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q2_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

typedef struct {
    uint64_t *grid;
    int *map;
    uint16_t *neighbours;
} ds4q_iq2_data;

static ds4q_iq2_data ds4q_iq2_xxs_data;

static int ds4q_iq2_compare_func(const void *left, const void *right) {
    const int *l = (const int *)left;
    const int *r = (const int *)right;
    return l[0] < r[0] ? -1 :
           l[0] > r[0] ?  1 :
           l[1] < r[1] ? -1 :
           l[1] > r[1] ?  1 : 0;
}

/*
 * IQ2_XXS quantizes a 256-value row block as eight 32-value groups.  Each
 * group stores four 8-value grid indices plus four 7-bit sign masks; the
 * single f16 block scale is refined by 4-bit per-group scale nibbles.
 *
 * The grid is tiny, but not every possible 2-bit 8-tuple is allowed.  During
 * initialization we build the direct map for allowed tuples and a nearest-grid
 * list for the missing ones, matching the GGML search exactly.
 */
void ds4q_iq2_xxs_init(void) {
    if (ds4q_iq2_xxs_data.grid) return;
    pthread_mutex_lock(&ds4q_init_mutex);
    if (ds4q_iq2_xxs_data.grid) {
        pthread_mutex_unlock(&ds4q_init_mutex);
        return;
    }

    enum { grid_size = 256, map_size = 43692, neighbour_shells = 2 };
    static const uint16_t kgrid[256] = {
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    };

    uint64_t *grid = malloc((size_t)grid_size * sizeof(grid[0]));
    int *map = malloc((size_t)map_size * sizeof(map[0]));
    int *dist2 = malloc((size_t)2 * grid_size * sizeof(dist2[0]));
    assert(grid && map && dist2);

    for (int k = 0; k < grid_size; k++) {
        int8_t *pos = (int8_t *)(grid + k);
        for (int i = 0; i < 8; i++) {
            int l = (kgrid[k] >> (2 * i)) & 3;
            pos[i] = 2 * l + 1;
        }
    }

    for (int i = 0; i < map_size; i++) map[i] = -1;
    for (int i = 0; i < grid_size; i++) map[kgrid[i]] = i;

    int8_t pos[8];
    int num_neighbors = 0;
    int num_not_in_map = 0;
    for (int i = 0; i < map_size; i++) {
        if (map[i] >= 0) continue;
        num_not_in_map++;
        for (int k = 0; k < 8; k++) pos[k] = 2 * ((i >> (2 * k)) & 3) + 1;
        for (int j = 0; j < grid_size; j++) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq2_compare_func);
        int d2 = dist2[0], have = 1;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                have++;
            }
            num_neighbors++;
        }
    }

    uint16_t *neighbours = malloc((size_t)(num_neighbors + num_not_in_map) * sizeof(neighbours[0]));
    assert(neighbours);
    int counter = 0;
    for (int i = 0; i < map_size; i++) {
        if (map[i] >= 0) continue;
        for (int k = 0; k < 8; k++) pos[k] = 2 * ((i >> (2 * k)) & 3) + 1;
        for (int j = 0; j < grid_size; j++) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq2_compare_func);
        map[i] = -(counter + 1);
        int d2 = dist2[0], have = 1;
        uint16_t *start = &neighbours[counter++];
        int n = 0;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                have++;
            }
            neighbours[counter++] = (uint16_t)dist2[2 * j + 1];
            n++;
        }
        *start = (uint16_t)n;
    }

    free(dist2);
    ds4q_iq2_xxs_data.map = map;
    ds4q_iq2_xxs_data.neighbours = neighbours;
    ds4q_iq2_xxs_data.grid = grid;
    pthread_mutex_unlock(&ds4q_init_mutex);
}

static int ds4q_iq2_find_best_neighbour(const uint16_t *neighbours, const uint64_t *grid,
                                        const float *xval, const float *weight,
                                        float scale, uint8_t *L) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; j++) {
        const int8_t *pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; i++) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t *pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; i++) L[i] = (uint8_t)((pg[i] - 1) / 2);
    return grid_index;
}

static void ds4q_write_iq2_xxs_block(const float *x, uint8_t *y, const float *quant_weights) {
    enum { d_off = 0, qs_off = 2, block_size = 32, k_max_q = 3 };
    assert(quant_weights);

    uint32_t q2[2 * (QK_K / block_size)];
    float scales[QK_K / block_size];
    float weight[block_size];
    float xval[block_size];
    uint8_t L[block_size];
    uint8_t Laux[block_size];
    float waux[block_size];
    uint8_t block_signs[4];

    uint16_t hd = ds4q_f32_to_f16(0.0f);
    memcpy(y + d_off, &hd, sizeof(hd));
    memset(q2, 0, sizeof(q2));

    const uint64_t *grid = ds4q_iq2_xxs_data.grid;
    const int *map = ds4q_iq2_xxs_data.map;
    const uint16_t *neighbours = ds4q_iq2_xxs_data.neighbours;
    assert(grid && map && neighbours);

    float sumx2 = 0;
    for (int i = 0; i < QK_K; i++) sumx2 += x[i] * x[i];
    float sigma2 = sumx2 / QK_K;
    float max_scale = 0;

    for (int ib = 0; ib < QK_K / block_size; ib++) {
        const float *xb = x + block_size * ib;
        const float *qw = quant_weights + block_size * ib;
        for (int i = 0; i < block_size; i++) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 4; k++) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; i++) {
                float v = xb[8 * k + i];
                if (v >= 0) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    nflip++;
                    s |= (uint8_t)(1u << i);
                }
            }
            if (nflip % 2) {
                int imin = 0;
                float min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                for (int i = 1; i < 8; i++) {
                    float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (ax < min) {
                        min = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (uint8_t)(1u << imin);
            }
            block_signs[k] = s & 127;
        }

        float max = xval[0];
        for (int i = 1; i < block_size; i++) max = DS4Q_MAX(max, xval[i]);
        if (max < DS4Q_GROUP_MAX_EPS) {
            scales[ib] = 0;
            memset(L, 0, sizeof(L));
            continue;
        }

        float scale = ds4q_make_qp_quants(block_size, k_max_q + 1, xval, L, weight);
        float eff_max = scale * k_max_q;
        if (eff_max <= 0) {
            scales[ib] = 0;
            memset(L, 0, sizeof(L));
            continue;
        }

        float best = 0;
        for (int is = -6; is <= 6; is++) {
            float id = (2 * k_max_q - 1 + is * 0.1f) / eff_max;
            float this_scale = 1 / id;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(k_max_q - 1, l));
                    Laux[8 * k + i] = (uint8_t)l;
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k, waux + 8 * k,
                                                 this_scale, Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; i++) {
                float w = weight[i];
                float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                memcpy(L, Laux, sizeof(L));
            }
        }

        if (scale > 0) {
            float id = 1 / scale;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(k_max_q - 1, l));
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    grid_index = ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k,
                                                              waux + 8 * k, scale, L + 8 * k);
                }
                const int8_t *pg = (const int8_t *)(grid + grid_index);
                for (int i = 0; i < 8; i++) L[8 * k + i] = (uint8_t)((pg[i] - 1) / 2);
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; i++) {
                float w = weight[i];
                float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; k++) block_signs[k] = (~block_signs[k]) & 127;
        }

        for (int k = 0; k < 4; k++) {
            uint16_t u = 0;
            for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
            int grid_index = map[u];
            assert(grid_index >= 0);
            q2[2 * ib + 0] |= (uint32_t)grid_index << (8 * k);
            q2[2 * ib + 1] |= (uint32_t)block_signs[k] << (7 * k);
        }
        assert(scale >= 0);
        scales[ib] = scale;
        max_scale = DS4Q_MAX(max_scale, scale);
    }

    if (!max_scale) {
        memset(y + qs_off, 0, QK_K / 4);
        return;
    }

    float d = max_scale / 31;
    hd = ds4q_f32_to_f16(d);
    memcpy(y + d_off, &hd, sizeof(hd));
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / block_size; ib++) {
        int l = ds4q_nearest_int(0.5f * (id * scales[ib] - 1));
        l = DS4Q_MAX(0, DS4Q_MIN(15, l));
        q2[2 * ib + 1] |= (uint32_t)l << 28;
    }
    memcpy(y + qs_off, q2, QK_K / 4);
}

size_t ds4q_quantize_iq2_xxs(const float *src, void *dst, int64_t start,
                                    int64_t nrows, int64_t ncols, const float *quant_weights) {
    assert(quant_weights);
    ds4q_iq2_xxs_init();
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_IQ2_XXS, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_IQ2_XXS].type_size;
            ds4q_write_iq2_xxs_block(xrow + (size_t)b * QK_K, block,
                                     quant_weights + (size_t)b * QK_K);
        }
    }
    return (size_t)nrows * row_size;
}
/* q2_K: scales[16]@0, qs[64]@16, d@80, dmin@82 (84B); 2-bit, 16 sub-blocks. */
void ds4q_dequantize_q2_k(const void *blocks, float *out, int64_t n) {
    const uint8_t *p = (const uint8_t *)blocks;
    for (int64_t i = 0; i < n / QK_K; i++) {
        const uint8_t *y = p + (size_t)i * 84, *scales = y, *q = y + 16;
        uint16_t hd, hm; memcpy(&hd, y + 80, 2); memcpy(&hm, y + 82, 2);
        float d = ds4q_f16_to_f32(hd), dmin = ds4q_f16_to_f32(hm);
        float *o = out + (size_t)i * QK_K; int is = 0;
        for (int g = 0; g < QK_K; g += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++]; float dl = d*(sc&0xF), ml = dmin*(sc>>4);
                for (int l = 0; l < 16; l++) *o++ = dl * ((q[l]    >> shift) & 3) - ml;
                sc = scales[is++]; dl = d*(sc&0xF); ml = dmin*(sc>>4);
                for (int l = 0; l < 16; l++) *o++ = dl * ((q[l+16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

/* iq2_xxs dequant (oracle floor tier). Inverts ds4q_quantize_iq2_xxs: block is
 * d(f16)@0 then 8 groups of [u32 grid-indices, u32 signs|scale] (66B total). The
 * codebook is the global grid built by ds4q_iq2_xxs_init (8 int8 {1,3,5,7} per
 * entry); signs use the canonical even-parity 7-bit ksigns table. */
static const uint8_t ds4q_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
void ds4q_dequantize_iq2_xxs(const void *blocks, float *out, int64_t n) {
    ds4q_iq2_xxs_init();
    const int8_t *grid = (const int8_t *)ds4q_iq2_xxs_data.grid;
    static const uint8_t kmask[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
    const uint8_t *p = (const uint8_t *)blocks;
    for (int64_t i = 0; i < n / QK_K; i++) {
        const uint8_t *blk = p + (size_t)i * 66;
        uint16_t hd; memcpy(&hd, blk, 2);
        const float d = ds4q_f16_to_f32(hd);
        const uint8_t *q2 = blk + 2;
        float *y = out + (size_t)i * QK_K;
        for (int ib = 0; ib < QK_K/32; ib++) {
            uint32_t aux32[2]; memcpy(aux32, q2 + (size_t)ib * 8, 8);
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const float db = d * (float)(2u * (aux32[1] >> 28) + 1u);
            for (int l = 0; l < 4; l++) {
                const int8_t *g = grid + (size_t)aux8[l] * 8;
                const uint8_t signs = ds4q_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++)
                    *y++ = db * (float)g[j] * ((signs & kmask[j]) ? -1.f : 1.f);
            }
        }
    }
}

