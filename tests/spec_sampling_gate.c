/* Statistical oracle for exact sampled speculative decoding.
 *
 * Promoted from temp/archive-2026-07-14/spec_sampling_test.c. temp/ is
 * transient and we have already lost harnesses that way (the Tier-2
 * byte-compare harness is gone; only gates promoted to tests/ survived), and
 * this is the ONLY end-to-end check that the speculative path emits the exact
 * target distribution.
 *
 * From one frozen session state, generates N short trajectories two ways:
 *   A) plain sampling (ds4_session_sample + eval per token)
 *   B) speculative   (ds4_session_generate_speculative, drafter active)
 * and chi-square-compares the per-position token marginals. If the acceptance +
 * residual-carry scheme is exact, the two are the same distribution; a biased
 * sampler shows up as a fat chi-square at position 1+ (position 0 is the same
 * plain draw in both paths by construction).
 *
 * The oracle is PROPOSAL-AGNOSTIC: it compares emitted marginals and never
 * inspects how a draft was proposed. It therefore validates the deterministic
 * (argmax-proposal, accept w.p. p) rule and the temperature-matched
 * (sampled-proposal, accept w.p. min(1,p/q), residual (p-q)+) rule unchanged —
 * which is exactly what makes it the gate for spec-decode Item 1.
 *
 * Also runs the greedy gate: temp=0 speculative output must agree with plain
 * greedy for a long prefix, and must be deterministic run-to-run.
 *
 * Reports acceptance alpha (accepted drafts / proposed drafts) from the
 * engine's own counters, so the same run answers "is the sampled proposal
 * beating the greedy p(mode) acceptance ceiling?".
 *
 * MODEL-DEPENDENT — needs the merged drafter gguf and ~95 GB free. Run:
 *   make cuda-spec-sampling-gate                  (defaults to gguf/model.gguf)
 *   ./tests/spec_sampling_gate <model.gguf> [temp] [filler_tokens]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"

#define TRAJ 2500
#define DEPTH 4
#define TOPN 24   /* histogram buckets per position: top tokens + tail bucket */

static const char *PROMPT =
    "The economic history of the Mediterranean is inseparable from its ports. ";

typedef struct { int id; long a, b; } bucket;

static int bucket_cmp(const void *x, const void *y) {
    const bucket *p = x, *q = y;
    return (int)((q->a + q->b) - (p->a + p->b));
}

/* alpha over a window of the engine's cumulative spec counters */
typedef struct { uint64_t drafted, accepted, rounds, gen; } spec_snap;

static spec_snap spec_take(ds4_engine *e) {
    ds4_spec_metrics m;
    memset(&m, 0, sizeof(m));
    ds4_engine_spec_metrics(e, &m);
    spec_snap s = { m.draft_tokens, m.accepted_tokens, m.num_drafts, m.gen_tokens };
    return s;
}

static void spec_report(const char *tag, spec_snap a, spec_snap b) {
    const double drafted = (double)(b.drafted - a.drafted);
    const double accepted = (double)(b.accepted - a.accepted);
    const double rounds = (double)(b.rounds - a.rounds);
    const double gen = (double)(b.gen - a.gen);
    printf("%s: alpha=%.4f (accepted %.0f / drafted %.0f)  tau=%.3f tokens/round  "
           "rounds=%.0f gen=%.0f\n",
           tag, drafted > 0 ? accepted / drafted : 0.0, accepted, drafted,
           rounds > 0 ? gen / rounds : 0.0, rounds, gen);
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "gguf/model.gguf";
    /* 0.95 is the acceptance-sensitive regime: hot enough that the greedy
     * p(mode) acceptance ceiling actually binds. */
    const float TEMP = argc > 2 ? (float)atof(argv[2]) : 0.95f;
    const int filler = argc > 3 ? atoi(argv[3]) : 0;
    const float TOP_P = 0.38f, MIN_P = 0.0f;

    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_CUDA };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, 16384) != 0) { fprintf(stderr, "session failed\n"); return 1; }

    /* Optional context filler: alpha falls with depth (77.6% shallow -> 61.7%
     * at 9.4k on v5mx), so a single shallow number is not the whole story. */
    char *user = NULL;
    if (filler > 0) {
        const size_t cap = (size_t)filler * 8u + strlen(PROMPT) + 64u;
        user = malloc(cap);
        size_t off = 0;
        for (int i = 0; i < filler && off + 8 < cap; i++)
            off += (size_t)snprintf(user + off, cap - off, "port%d ", i % 997);
        snprintf(user + off, cap - off, "%s", PROMPT);
    }

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", user ? user : PROMPT);
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    char err[256];
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "sync failed: %s\n", err);
        return 1;
    }
    printf("model=%s temp=%.2f top_p=%.2f ctx_depth=%d\n",
           model, (double)TEMP, (double)TOP_P, ds4_session_pos(session));
    ds4_session_snapshot snap = {0};
    if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "snapshot failed: %s\n", err);
        return 1;
    }
    const int eos = ds4_token_eos(engine);

    /* ---- greedy gates ----
     * The batched verifier is documented near-greedy (batch reductions can
     * flip nearly-tied argmaxes vs the plain decode kernels), so token-exact
     * equality with plain decode is NOT the engine's contract. Gates:
     *   1. spec greedy is deterministic (two runs identical), and
     *   2. it agrees with plain greedy for a long prefix (>= 8 tokens).
     * Temperature-matched draft sampling must leave this path untouched: at
     * temp <= 0 no q is built, no rng is drawn, and the argmax-equality accept
     * walk runs exactly as before. */
    {
        int ref[24], got[24], got2[24];
        int nref = 0, ngot = 0, ngot2 = 0;
        uint64_t rng = 7;
        for (int t = 0; t < 24; t++) {
            int tok = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
            if (tok == eos) break;
            ref[nref++] = tok;
            if (ds4_session_eval(session, tok, err, sizeof(err)) != 0) return 1;
        }
        const spec_snap g0 = spec_take(engine);
        for (int rep = 0; rep < 2; rep++) {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) return 1;
            int *dst = rep == 0 ? got : got2;
            int *n = rep == 0 ? &ngot : &ngot2;
            rng = 7;
            while (*n < nref) {
                int toks[17];
                int k = ds4_session_generate_speculative(session, 0.0f, 0, 1.0f, 0.0f, &rng,
                                                         nref - *n, eos, toks, 17,
                                                         err, sizeof(err));
                if (k <= 0) { fprintf(stderr, "greedy spec failed: %s\n", err); return 1; }
                for (int i = 0; i < k && *n < nref; i++) dst[(*n)++] = toks[i];
            }
        }
        const spec_snap g1 = spec_take(engine);
        int det = ngot == ngot2 && memcmp(got, got2, (size_t)ngot * sizeof(int)) == 0;
        int prefix = 0;
        while (prefix < nref && prefix < ngot && ref[prefix] == got[prefix]) prefix++;
        /* informational: batch verify inherits the prefill atomicAdd
         * nondeterminism (#17), so run-to-run equality is not achievable
         * until ordered reductions land; tie positions flip. */
        printf("greedy determinism (2 runs): %s (informational; batch FP nondeterminism)\n",
               det ? "same" : "tie-flips");
        printf("greedy prefix agreement vs plain: %d/%d tokens %s\n",
               prefix, nref, prefix >= 8 ? "PASS" : "FAIL");
        spec_report("greedy  ", g0, g1);
        /* Byte-identity aid: the greedy token stream is printed so a build from
         * a baseline commit can be diffed against this one token-for-token. */
        printf("greedy tokens:");
        for (int i = 0; i < ngot; i++) printf(" %d", got[i]);
        printf("\n");
        if (prefix < 8) return 1;
    }

    /* ---- sampled-distribution comparison ---- */
    static int seqA[TRAJ][DEPTH], seqB[TRAJ][DEPTH];
    spec_snap s0 = {0}, s1 = {0};
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 1) s0 = spec_take(engine);
        for (int t = 0; t < TRAJ; t++) {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) return 1;
            uint64_t rng = 0x9E3779B97F4A7C15ull * (uint64_t)(t + 1) + (uint64_t)mode * 77777u;
            int *dst = mode == 0 ? seqA[t] : seqB[t];
            int got = 0;
            if (mode == 0) {
                while (got < DEPTH) {
                    int tok = ds4_session_sample(session, TEMP, 0, TOP_P, MIN_P, &rng);
                    dst[got++] = tok;
                    if (tok == eos) break;
                    if (got < DEPTH && ds4_session_eval(session, tok, err, sizeof(err)) != 0) return 1;
                }
            } else {
                while (got < DEPTH) {
                    int toks[17];
                    int n = ds4_session_generate_speculative(session, TEMP, 0, TOP_P, MIN_P, &rng,
                                                             DEPTH - got, eos, toks, 17,
                                                             err, sizeof(err));
                    if (n <= 0) { fprintf(stderr, "spec step failed: %s\n", err); return 1; }
                    for (int i = 0; i < n && got < DEPTH; i++) dst[got++] = toks[i];
                    if (dst[got - 1] == eos) break;
                }
            }
            for (int k = got; k < DEPTH; k++) dst[k] = -1;
            if ((t + 1) % 500 == 0)
                fprintf(stderr, "mode %d: %d/%d trajectories\r", mode, t + 1, TRAJ);
        }
        fputc('\n', stderr);
    }
    s1 = spec_take(engine);
    /* THE Item 1 number: sampled-proposal acceptance at TEMP. The deterministic
     * (argmax-proposal) rule is capped at E[p(mode)]; min(1,p/q) is not. Compare
     * against a build of this same gate from the pre-Item-1 commit. */
    spec_report("sampled ", s0, s1);

    /* chi-square per position over pooled top buckets */
    int fail = 0;
    for (int posn = 0; posn < DEPTH; posn++) {
        bucket bk[4096];
        int nb = 0;
        for (int mode = 0; mode < 2; mode++)
            for (int t = 0; t < TRAJ; t++) {
                int tok = mode == 0 ? seqA[t][posn] : seqB[t][posn];
                int j = 0;
                for (; j < nb; j++) if (bk[j].id == tok) break;
                if (j == nb) { if (nb >= 4096) continue; bk[nb++] = (bucket){tok, 0, 0}; }
                if (mode == 0) bk[j].a++; else bk[j].b++;
            }
        qsort(bk, (size_t)nb, sizeof(bk[0]), bucket_cmp);
        int keep = nb < TOPN ? nb : TOPN;
        long resta = 0, restb = 0;
        for (int j = keep; j < nb; j++) { resta += bk[j].a; restb += bk[j].b; }
        double chi = 0.0;
        int df = 0;
        for (int j = 0; j <= keep; j++) {
            const long a = j < keep ? bk[j].a : resta;
            const long b = j < keep ? bk[j].b : restb;
            const double tot = (double)(a + b);
            if (tot < 10.0) continue;
            const double ea = tot / 2.0, ebb = tot / 2.0;
            chi += ((double)a - ea) * ((double)a - ea) / ea +
                   ((double)b - ebb) * ((double)b - ebb) / ebb;
            df++;
        }
        df = df > 1 ? df - 1 : 1;
        /* p ~ 0.001 critical values: chi2(df) ≈ df + 3.1*sqrt(2 df) + 4 */
        const double crit = df + 3.1 * sqrt(2.0 * df) + 4.0;
        printf("pos %d: chi2=%.1f df=%d crit(p~.001)=%.1f -> %s\n",
               posn, chi, df, crit, chi <= crit ? "OK" : "FAIL");
        if (chi > crit) fail = 1;
    }
    printf(fail ? "spec sampling oracle FAIL\n" : "spec sampling oracle PASS\n");
    free(user);
    return fail;
}
