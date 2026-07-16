#!/usr/bin/env python3
"""Compare two engines' trajectory dumps from tests/spec_sampling_gate.

THE DISCRIMINATOR: mode1(fixed) vs mode1(baseline).

The gate's own chi-square pits mode0 (plain, sampled from single-token DECODE
logits) against mode1 (speculative, p read from BATCHED verify rows). Those are
two different numeric paths and they are known to diverge on near-ties, so at
depth that test has a systematic confound present in BOTH engines -- the shipped
baseline sits at 94% of crit.

mode1-vs-mode1 removes it: both arms are speculative, both source p from batched
rows. Both accept rules (deterministic accept-w.p.-p, and min(1,p/q)+residual)
are exact for an ARBITRARY proposal, so if p/q is implemented correctly the two
engines must reproduce the SAME filtered target distribution and agree.

CONTROL: mode0(fixed) vs mode0(baseline) must be IDENTICAL -- the plain sampling
path is untouched by Item 1 and the seeds match. If that control does not come
out clean, the comparison means nothing.
"""
import struct
import sys
import math


def load(path):
    with open(path, "rb") as f:
        traj, depth = struct.unpack("<ii", f.read(8))
        n = traj * depth
        a = list(struct.unpack("<%di" % n, f.read(4 * n)))
        b = list(struct.unpack("<%di" % n, f.read(4 * n)))
    A = [a[i * depth:(i + 1) * depth] for i in range(traj)]
    B = [b[i * depth:(i + 1) * depth] for i in range(traj)]
    return traj, depth, A, B


def chi2_compare(X, Y, depth, label, topn=24):
    """2-sample homogeneity chi-square per position, same pooling as the gate."""
    print("=== %s ===" % label)
    worst = 0
    for pos in range(depth):
        cnt = {}
        for row in X:
            cnt.setdefault(row[pos], [0, 0])[0] += 1
        for row in Y:
            cnt.setdefault(row[pos], [0, 0])[1] += 1
        items = sorted(cnt.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
        keep = items[:topn]
        rest_a = sum(v[0] for _, v in items[topn:])
        rest_b = sum(v[1] for _, v in items[topn:])
        buckets = [v for _, v in keep] + ([[rest_a, rest_b]] if (rest_a + rest_b) else [])
        chi, df = 0.0, 0
        for a, b in buckets:
            tot = a + b
            if tot < 10:
                continue
            ea = eb = tot / 2.0
            chi += (a - ea) ** 2 / ea + (b - eb) ** 2 / eb
            df += 1
        df = df - 1 if df > 1 else 1
        crit = df + 3.1 * math.sqrt(2.0 * df) + 4.0
        # N-invariant effect size: total-variation distance between the arms
        NA = sum(v[0] for _, v in items) or 1
        NB = sum(v[1] for _, v in items) or 1
        tvd = 0.5 * sum(abs(v[0] / NA - v[1] / NB) for _, v in items)
        ok = chi <= crit
        worst = max(worst, chi / crit)
        print("  pos %d: chi2=%6.1f df=%2d crit=%5.1f  TVD=%.4f  N=%d/%d  distinct=%3d -> %s"
              % (pos, chi, df, crit, tvd, NA, NB, len(items), "OK" if ok else "FAIL"))
    print("  -> %s (worst chi2/crit = %.2f)\n" % ("PASS" if worst <= 1.0 else "FAIL", worst))
    return worst <= 1.0


def identical(X, Y):
    return X == Y


f_traj, f_depth, fA, fB = load(sys.argv[1])   # fixed
b_traj, b_depth, bA, bB = load(sys.argv[2])   # baseline
assert (f_traj, f_depth) == (b_traj, b_depth)
print("traj=%d depth=%d\n" % (f_traj, f_depth))

# CONTROL: the untouched plain path must be bit-identical across engines.
same = identical(fA, bA)
print("CONTROL mode0(fixed) vs mode0(baseline) identical: %s%s\n"
      % (same, "" if same else "   <-- control BROKEN, comparison below is meaningless"))
if not same:
    chi2_compare(fA, bA, f_depth, "mode0 vs mode0 (control, expected identical)")

# THE DISCRIMINATOR
ok = chi2_compare(fB, bB, f_depth, "DISCRIMINATOR: mode1(fixed p/q) vs mode1(baseline deterministic)")

# For reference, each engine's own gate test (mode0 vs mode1), confound included.
chi2_compare(fA, fB, f_depth, "reference: fixed mode0 vs fixed mode1 (has batch-vs-decode confound)")
chi2_compare(bA, bB, f_depth, "reference: baseline mode0 vs baseline mode1 (same confound)")

print("VERDICT: p/q %s" % ("agrees with the deterministic rule -> exact" if ok
                           else "DISAGREES with the deterministic rule -> real bias"))
sys.exit(0 if ok else 1)
