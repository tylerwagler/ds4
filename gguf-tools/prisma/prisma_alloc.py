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
# Candidate menu — antirez two-pair law (ds4_cuda.cu routed_moe_launch:12251).
# The clean antirez runtime's ONLY routed-MoE entry point hard-accepts exactly
# two per-layer presets and returns 0 ("ffn batch encode failed") for anything
# else:
#   q4k_path : gate==Q4_K(12) AND down==Q4_K(12)
#   else     : gate==IQ2_XXS(16) AND down==Q2_K(10)
# So a whole MoE layer is the allocation UNIT and it picks ONE of two presets.
# (The Entrpi fork widened this to the full ladder via the mmq graft; that work
# is staged for a surgical, oracle-triggered port — see ROADMAP. Until then the
# allocator MUST emit only these presets or it produces unservable GGUFs, which
# is exactly how smart-99gb got IQ2_XXS down-projections.)
PRESETS = {
    "cheap": {"gate": "IQ2_XXS", "up": "IQ2_XXS", "down": "Q2_K"},   # ~2.06 / 2.6 bpw
    "rich":  {"gate": "Q4_K",    "up": "Q4_K",    "down": "Q4_K"},   # 4.5 bpw
}
PRESET_ORDER = ["cheap", "rich"]               # ascending size; promote cheap->rich
ROUTED_CANDS = ["IQ2_XXS", "Q2_K", "Q4_K"]     # display only, bpw-ascending

def expert_role(name):
    if "down_exps" in name: return "down"
    if "up_exps"   in name: return "up"
    return "gate"                               # gate_exps

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
    """Per-role format options under the two presets (cheap-first), for routed
    experts; non-experts pinned at their current type. candidates(t)[0] is the
    cheap-preset format for the tensor's role (used as the pinned default)."""
    if t["family"] == "routed_expert":
        role = expert_role(t["name"])
        return [PRESETS[p][role] for p in PRESET_ORDER]
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
    """Per-layer two-preset knapsack for the antirez two-pair runtime. The unit
    is a whole MoE layer (its gate_exps+up_exps+down_exps); it chooses 'cheap'
    (IQ2_XXS gate/up, Q2_K down) or 'rich' (Q4_K all). Start every layer cheap,
    then greedily promote whole layers to rich by best dKL-saved-per-extra-byte
    while the budget holds. Two presets => 0/1 per layer; greedy-by-efficiency is
    exact for the LP relaxation and always budget-respecting."""
    import heapq, collections
    fixed_bytes = 0
    layers = collections.OrderedDict()   # layer -> [(name, n_el, sv, role), ...]
    for t in tensors:
        n_el = nelems(t["shape"])
        if t["family"] != "routed_expert":
            fixed_bytes += size_bytes(n_el, candidates(t)[0]); continue
        n = t["name"]
        try: layer = n.split("blk.")[1].split(".")[0]
        except Exception: layer = "?"
        sv = (sens or {}).get(n, synth_sens(t))
        layers.setdefault(layer, []).append((n, n_el, sv, expert_role(n)))

    def preset_cost(ts, preset):
        sz = 0; dkl = 0.0
        for (nm, n_el, sv, role) in ts:
            fmt = PRESETS[preset][role]
            sz += size_bytes(n_el, fmt)
            m = (mse or {}).get(nm, {}).get(fmt)
            if m is None: m = synth_mse(n_el, fmt)
            dkl += 0.5 * sv * m
        return sz, dkl

    units = []   # [layer, tensors, {preset:(sz,dkl)}, is_rich]
    for layer, ts in layers.items():
        costs = {p: preset_cost(ts, p) for p in PRESET_ORDER}
        units.append([layer, ts, costs, False])

    rsize = sum(u[2]["cheap"][0] for u in units)   # all cheap
    rkl   = sum(u[2]["cheap"][1] for u in units)
    budget = budget_bytes - fixed_bytes

    # max-heap (via negation) of promotion efficiency = dKL_saved / extra_bytes
    heap = []
    for i, u in enumerate(units):
        ds = u[2]["rich"][0] - u[2]["cheap"][0]
        dk = u[2]["cheap"][1] - u[2]["rich"][1]     # KL reduction from cheap->rich
        if ds > 0: heapq.heappush(heap, (-(dk/ds), i))
    while heap:
        eff, i = heapq.heappop(heap)
        u = units[i]
        ds = u[2]["rich"][0] - u[2]["cheap"][0]
        if rsize + ds > budget:
            continue                                # doesn't fit; try next-best
        rkl -= (u[2]["cheap"][1] - u[2]["rich"][1]); rsize += ds; u[3] = True

    chosen = {}
    for layer, ts, costs, is_rich in units:
        preset = "rich" if is_rich else "cheap"
        for (nm, _, _, role) in ts:
            chosen[nm] = PRESETS[preset][role]
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
