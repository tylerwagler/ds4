#!/usr/bin/env python3
"""prisma_alloc.py — Fisher x MSE knapsack allocator for ds4 mixed quants.

Rob's method (PrismaQuant), applied to the ds4 GGUF quantizer:
  cost  dKL(tensor, fmt) ~= 0.5 * sensitivity(tensor) * mse(tensor, fmt)
  solve multiple-choice knapsack: each tensor picks ONE format from its
  candidate set, minimizing sum(dKL) subject to sum(size) <= budget.
Solved by Lagrangian relaxation: for a price lambda (per byte), each tensor
independently picks argmin_fmt (dKL + lambda*size); binary-search lambda to hit
the budget. Emits one per-tensor format manifest per Pareto budget target.

Inputs (JSON):
  --tensors  [{name, shape:[...], family, type}]            (from gguf inventory)
  --sens     {name: sensitivity}     optional; from imatrix (Fisher proxy)
  --mse      {name: {FMT: mse}}       optional; round-trip MSE per candidate
If --sens/--mse are absent, a documented SYNTHETIC fallback is used so the
machinery is testable before the source/imatrix land; results are then
size-driven only (NOT a real allocation).
"""
import argparse, json, math, os

# ggml block geometry: (elems_per_block, bytes_per_block). bpw = bytes*8/elems.
BLOCK = {
    "IQ2_XXS": (256, 66),   # 2.0625 bpw
    "Q2_K":    (256, 84),   # 2.625
    "Q3_K":    (256, 110),  # 3.4375  (pending encoder)
    "Q4_K":    (256, 144),  # 4.5
    "Q5_K":    (256, 176),  # 5.5
    "Q6_K":    (256, 210),  # 6.5625
    "Q8_0":    (32, 34),    # 8.5
    "F16":     (1, 2),
    "F32":     (1, 4),
}
# Candidate menus. ds4-spark widened the fork's routed-expert kernels to the
# full k-quant ladder (mmq dispatchers in routed_moe_launch + q2_K/q3_K/q5_K/q6_K
# pair kernels). Remaining hard constraint: gate & up MUST share a type (one
# fused SwiGLU pair kernel covers both). down is independent. Both roles now
# span the whole ladder, so the knapsack has fine granularity end to end.
LADDER       = ["IQ2_XXS", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
GATEUP_CANDS = LADDER                          # gate+up: full ladder (q2_K pair added)
DOWN_CANDS   = LADDER                          # down: full ladder
ROUTED_CANDS = LADDER                          # union (display only), bpw-ascending

def nelems(shape):
    n = 1
    for s in shape: n *= int(s)
    return n

def size_bytes(n_el, fmt):
    eb, bb = BLOCK[fmt]
    return math.ceil(n_el / eb) * bb

def bpw(fmt):
    eb, bb = BLOCK[fmt]
    return bb * 8.0 / eb

def candidates(t):
    """Format menu for a tensor, honoring ds4's gate/up vs down constraints.
    Non-experts are pinned at their current type (small + already chosen)."""
    if t["family"] == "routed_expert":
        return DOWN_CANDS if "down_exps" in t["name"] else GATEUP_CANDS
    return [t["type"]] if t["type"] in BLOCK else ["F16"]

def synth_sens(t):
    """Synthetic sensitivity fallback (NOT real). Heuristic: down-proj and later
    layers matter more — purely to exercise the knapsack until imatrix lands."""
    n = t["name"]; s = 1.0
    if "ffn_down" in n: s *= 1.6           # down-proj is MoE-sensitive
    try:
        layer = int(n.split("blk.")[1].split(".")[0]); s *= 1.0 + layer / 60.0
    except Exception:
        pass
    return s

def synth_mse(n_el, fmt):
    """Synthetic round-trip MSE fallback (NOT real): ~ exp falloff with bpw."""
    return math.exp(-1.4 * (bpw(fmt) - 2.0))

def load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None

def allocate(tensors, sens, mse, budget_bytes):
    """Greedy-upgrade multiple-choice knapsack. Each routed tensor starts at the
    cheapest format; we repeatedly apply the upgrade (next format up) with the
    best dKL-saved-per-extra-byte, while it fits the budget. Optimal for convex
    per-tensor cost curves; near-optimal otherwise and always budget-respecting."""
    import heapq, collections
    # CONSTRAINT: gate+up experts in a layer must share a quant type (fused
    # moe_pair kernel). So the allocation UNIT is (layer, gate+up) or (layer,
    # down); down is independent. Each unit emits its format to all its tensors.
    fixed_bytes = 0
    units = collections.OrderedDict()   # unit_key -> {tensors:[(name,n_el,sv,cands)]}
    for t in tensors:
        n_el = nelems(t["shape"]); cands = candidates(t)
        if t["family"] != "routed_expert" or len(cands) == 1:
            fixed_bytes += size_bytes(n_el, cands[0]); continue
        n = t["name"]
        try: layer = n.split("blk.")[1].split(".")[0]
        except Exception: layer = "?"
        role = "gate_up" if ("gate_exps" in n or "up_exps" in n) else "down"
        key = (layer, role)
        sv = (sens or {}).get(n, synth_sens(t))
        units.setdefault(key, {"tensors": []})["tensors"].append((n, n_el, sv, cands))

    routed = []   # [names, opts(sorted by size), lvl]
    for key, u in units.items():
        ts = u["tensors"]
        cands = ts[0][3]                       # same menu for gate & up
        opts = []
        for f in cands:
            sz = sum(size_bytes(n_el, f) for (_, n_el, _, _) in ts)
            dkl = 0.0
            for (nm, n_el, sv, _) in ts:
                m = (mse or {}).get(nm, {}).get(f)
                if m is None: m = synth_mse(n_el, f)
                dkl += 0.5 * sv * m
            opts.append((f, sz, dkl))
        opts.sort(key=lambda o: o[1])
        routed.append([[nm for (nm, _, _, _) in ts], opts, 0])

    rsize = sum(o[1][0][1] for o in routed)       # all at cheapest
    rkl   = sum(o[1][0][2] for o in routed)
    budget = budget_bytes - fixed_bytes

    # max-heap (via negation) of upgrade efficiency = dKL_saved / extra_bytes
    heap = []
    for i, (name, opts, lvl) in enumerate(routed):
        if lvl + 1 < len(opts):
            ds = opts[lvl+1][1] - opts[lvl][1]
            dk = opts[lvl][2]  - opts[lvl+1][2]      # KL reduction (>0 if higher fmt better)
            if ds > 0: heapq.heappush(heap, (-(dk/ds), i))
    while heap:
        eff, i = heapq.heappop(heap)
        name, opts, lvl = routed[i]
        ds = opts[lvl+1][1] - opts[lvl][1]
        if rsize + ds > budget:
            continue                                # this upgrade doesn't fit; skip
        # apply upgrade
        rkl -= (opts[lvl][2] - opts[lvl+1][2]); rsize += ds
        routed[i][2] = lvl + 1; lvl += 1
        if lvl + 1 < len(opts):
            ds2 = opts[lvl+1][1] - opts[lvl][1]
            dk2 = opts[lvl][2]  - opts[lvl+1][2]
            if ds2 > 0: heapq.heappush(heap, (-(dk2/ds2), i))

    chosen = {}
    for names, opts, lvl in routed:
        fmt = opts[lvl][0]
        for nm in names: chosen[nm] = fmt
    return chosen, fixed_bytes, rsize, rkl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", required=True)
    ap.add_argument("--sens"); ap.add_argument("--mse")
    ap.add_argument("--pareto", nargs="+", type=float, default=[90, 99, 105])
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    tensors = json.load(open(a.tensors)); sens = load(a.sens); mse = load(a.mse)
    using = ("imatrix sens" if sens else "SYNTHETIC sens") + " + " + ("real MSE" if mse else "SYNTHETIC MSE")
    print(f"# {len(tensors)} tensors; cost model: {using}")
    print(f"# routed-expert candidate menu: {ROUTED_CANDS}\n")
    for gb in a.pareto:
        chosen, fixed, rtot, kl = allocate(tensors, sens, mse, gb * 1e9)
        total = (fixed + rtot) / 1e9
        # full manifest (experts allocated, others pinned at current type)
        man = {}
        for t in tensors:
            man[t["name"]] = chosen.get(t["name"], (candidates(t)[0]))
        outp = os.path.join(a.out_dir, f"manifest.{int(gb)}gb.json")
        json.dump(man, open(outp, "w"), indent=0)
        # per-format routed counts
        from collections import Counter
        c = Counter(chosen.values())
        cnt = "  ".join(f"{k}:{c.get(k,0)}" for k in ROUTED_CANDS)
        print(f"target {gb:>5.0f} GB -> actual {total:5.1f} GB  fixed {fixed/1e9:4.1f}  "
              f"routed {rtot/1e9:5.1f}  sum_dKL {kl:8.3f}   [{cnt}]  -> {os.path.basename(outp)}")

if __name__ == "__main__":
    main()
