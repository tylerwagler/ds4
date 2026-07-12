#!/usr/bin/env python3
"""prisma_alloc.py — per-layer expert-format allocator for ds4 mixed quants.

Cost model, in order of preference:
  1. --kl kl.json          MEASURED per-layer promotion benefit: real end-to-end
                           KL reduction from promoting one MoE layer's routed
                           experts cheap->rich, measured by measure_layer_kl.py
                           on this engine (AURA-era methodology: for routed
                           experts, trust measurement, not proxies — gradient/
                           imatrix proxies can't see router flips).
  2. --sens/--mse          PROXY: dKL(tensor,fmt) ~= 0.5 * sensitivity * mse
                           (imatrix Fisher proxy x round-trip MSE from
                           deepseek4-quantize --mse-probe).
  3. neither               documented SYNTHETIC fallback so the machinery is
                           testable; results are size-driven only (NOT real).

Optimization: each MoE layer is a 0/1 unit (cheap or rich preset — the GGUF
stacks all 256 experts per tensor, so per-expert formats are structurally
unavailable). Solved EXACTLY by dynamic programming over the byte budget at
1 MiB granularity (all units happen to weigh the same for this model, which
makes top-K-by-benefit exact too; the DP stays correct if that ever changes).

--ucb-z Z charges `Z * stderr` against each measured benefit before the solve
(uncertainty-aware allocation; default 0 = raw means).

Emits one per-tensor format manifest per Pareto budget target; feed it to
deepseek4-quantize --format-map manifest.json.
"""
import argparse, json, math, os

# ggml block geometry: (elems_per_block, bytes_per_block). bpw = bytes*8/elems.
BLOCK = {
    "IQ2_XXS": (256, 66),   # 2.0625 bpw
    "Q2_K":    (256, 84),   # 2.625
    "MXFP4":   (32,  17),   # 4.25 — lossless vs source
    "FP8_E4M3":(32,  33),   # 8.25
    "F16":     ( 1,   2),
    "F32":     ( 1,   4),
}
# Candidate presets. Each MoE layer picks one: cheap (IQ2_XXS/Q2_K) or rich
# (MXFP4 all three, lossless vs the HF source).  No Q4_K/Q3_K/Q5_K/Q6_K/Q8_0
# — those are no longer producible or servable.
# NOTE: rich layers served on the prefill tensor-core path additionally need
# the CUTLASS type-40 splice (converter/splice_cutlass_mxfp4.py) as a
# post-step; plain type-39 MXFP4 decodes fine but skips prefill batching.
PRESETS = {
    "cheap": {"gate": "IQ2_XXS", "up": "IQ2_XXS", "down": "Q2_K"},   # ~2.06 / 2.6 bpw
    "mid":   {"gate": "Q2_K",    "up": "Q2_K",    "down": "Q2_K"},   # 2.625 bpw everywhere
    "rich":  {"gate": "MXFP4",   "up": "MXFP4",   "down": "MXFP4"},  # 4.25 bpw, lossless
}
PRESET_ORDER = ["cheap", "mid", "rich"]        # ascending size
ROUTED_CANDS = ["IQ2_XXS", "Q2_K", "MXFP4"]    # display only, bpw-ascending

MIB = 1 << 20   # DP budget granularity

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
    if t["family"] == "routed_expert":
        role = expert_role(t["name"])
        return [PRESETS[p][role] for p in PRESET_ORDER]
    return [t["type"]] if t["type"] in BLOCK else ["F16"]

def synth_sens(t):
    n = t["name"]; s = 1.0
    if "ffn_down" in n: s *= 1.6
    try:
        layer = int(n.split("blk.")[1].split(".")[0]); s *= 1.0 + layer / 60.0
    except Exception:
        pass
    return s

def synth_mse(n_el, fmt):
    """Synthetic round-trip MSE fallback (NOT real). MXFP4 is lossless vs the
    FP4 source, so its MSE is effectively zero.  IQ2_XXS/Q2_K get the old
    exponential falloff with bpw gap from MXFP4."""
    if fmt == "MXFP4":
        return 1e-12   # lossless
    return math.exp(-1.4 * (bpw(fmt) - 2.0))

def load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None

def solve_01_knapsack(weights_mib, values, budget_mib):
    """Exact 0/1 knapsack: maximize sum(values) under sum(weights) <= budget.
    DP over MiB-discretized budget; returns the chosen index set."""
    n = len(weights_mib)
    W = max(0, int(budget_mib))
    NEG = float("-inf")
    best = [0.0] + [NEG] * W
    keep = [[False] * (W + 1) for _ in range(n)]
    for i in range(n):
        w, v = weights_mib[i], values[i]
        if v <= 0.0:
            continue        # promoting can't help; never take
        for c in range(W, w - 1, -1):
            cand = best[c - w] + v
            if best[c - w] > NEG and cand > best[c]:
                best[c] = cand
                keep[i][c] = True
    # backtrack from the best-value cell
    c = max(range(W + 1), key=lambda x: best[x])
    chosen = set()
    for i in range(n - 1, -1, -1):
        if keep[i][c]:
            chosen.add(i)
            c -= weights_mib[i]
    return chosen

def allocate(tensors, sens, mse, kl, kl_mid, budget_bytes, ucb_z):
    """Per-layer preset allocation. Unit = whole MoE layer (gate+up+down);
    picks 'cheap', 'mid' (only when --kl-mid measurements exist), or 'rich'.
    Benefits come from measured KL when available (--kl / --kl-mid), else the
    0.5*sens*mse proxy (rich only). Solved exactly by grouped DP."""
    import collections
    fixed_bytes = 0
    layers = collections.OrderedDict()
    for t in tensors:
        n_el = nelems(t["shape"])
        if t["family"] != "routed_expert":
            fixed_bytes += size_bytes(n_el, candidates(t)[0]); continue
        n = t["name"]
        try: layer = n.split("blk.")[1].split(".")[0]
        except Exception: layer = "?"
        sv = (sens or {}).get(n, synth_sens(t))
        layers.setdefault(layer, []).append((n, n_el, sv, expert_role(n)))

    def preset_size(ts, preset):
        return sum(size_bytes(n_el, PRESETS[preset][role]) for (_, n_el, _, role) in ts)

    def proxy_dkl(ts, preset):
        dkl = 0.0
        for (nm, n_el, sv, role) in ts:
            fmt = PRESETS[preset][role]
            m = (mse or {}).get(nm, {}).get(fmt)
            if m is None: m = synth_mse(n_el, fmt)
            dkl += 0.5 * sv * m
        return dkl

    def measured(store, layer):
        if store is None or layer not in store:
            return None
        return float(store[layer]["dkl_promote"]) - \
               ucb_z * float(store[layer].get("stderr", 0.0))

    units = []   # [layer, ts, sizes{preset}, benefits{preset}]
    for layer, ts in layers.items():
        sizes = {p: preset_size(ts, p) for p in PRESET_ORDER}
        b_rich = measured(kl, layer)
        if b_rich is None:
            b_rich = proxy_dkl(ts, "cheap") - proxy_dkl(ts, "rich")
        benefits = {"cheap": 0.0, "rich": b_rich}
        b_mid = measured(kl_mid, layer)
        if b_mid is not None:
            # mid is a strict subset of the rich improvement; cap noise
            benefits["mid"] = min(b_mid, b_rich)
        units.append([layer, ts, sizes, benefits])

    base_routed = sum(u[2]["cheap"] for u in units)
    budget_left = (budget_bytes - fixed_bytes - base_routed) // MIB
    # grouped DP: each layer picks one preset (cheap weight 0, value 0)
    NEG = float("-inf")
    W = max(0, int(budget_left))
    best = [0.0] + [NEG] * W
    pick = []
    for layer, ts, sizes, benefits in units:
        opts = [("cheap", 0, 0.0)]
        for p in ("mid", "rich"):
            if p in benefits and benefits[p] > 0.0:
                w = max(1, math.ceil((sizes[p] - sizes["cheap"]) / MIB))
                opts.append((p, w, benefits[p]))
        row = [None] * (W + 1)
        nxt = [NEG] * (W + 1)
        for c in range(W + 1):
            for (pname, w, v) in opts:
                if c >= w and best[c - w] > NEG:
                    cand = best[c - w] + v
                    if cand > nxt[c]:
                        nxt[c] = cand
                        row[c] = (pname, w)
        best = nxt
        pick.append(row)

    c = max(range(W + 1), key=lambda x: best[x])
    choice = {}
    for i in range(len(units) - 1, -1, -1):
        pname, w = pick[i][c] if pick[i][c] else ("cheap", 0)
        choice[i] = pname
        c -= w

    chosen = {}
    rsize = 0
    saved_kl = 0.0
    for i, (layer, ts, sizes, benefits) in enumerate(units):
        preset = choice.get(i, "cheap")
        rsize += sizes[preset]
        saved_kl += benefits.get(preset, 0.0)
        for (nm, _, _, role) in ts:
            chosen[nm] = PRESETS[preset][role]
    return chosen, fixed_bytes, rsize, saved_kl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", required=True)
    ap.add_argument("--sens"); ap.add_argument("--mse")
    ap.add_argument("--kl", help="measured per-layer KL costs from measure_layer_kl.py")
    ap.add_argument("--kl-mid", help="measured per-layer KL for the mid (Q2_K gate/up) preset")
    ap.add_argument("--ucb-z", type=float, default=0.0,
                    help="charge Z*stderr against measured benefits (default 0)")
    ap.add_argument("--pareto", nargs="+", type=float, default=[90, 99, 105])
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    tensors = json.load(open(a.tensors))
    sens = load(a.sens); mse = load(a.mse); kl = load(a.kl); kl_mid = load(a.kl_mid)
    if kl:
        using = f"MEASURED KL ({len(kl)} layers" + \
                (f" + mid {len(kl_mid)}" if kl_mid else "") + f", ucb_z={a.ucb_z})"
    else:
        using = ("imatrix sens" if sens else "SYNTHETIC sens") + " + " + \
                ("real MSE" if mse else "SYNTHETIC MSE")
    print(f"# {len(tensors)} tensors; cost model: {using}")
    print(f"# routed-expert candidate menu: {ROUTED_CANDS}\n")
    for gb in a.pareto:
        chosen, fixed, rtot, saved = allocate(tensors, sens, mse, kl, kl_mid, gb * 1e9, a.ucb_z)
        total = (fixed + rtot) / 1e9
        man = {}
        for t in tensors:
            man[t["name"]] = chosen.get(t["name"], (candidates(t)[0]))
        outp = os.path.join(a.out_dir, f"manifest.{int(gb)}gb.json")
        json.dump(man, open(outp, "w"), indent=0)
        from collections import Counter
        c = Counter(chosen.values())
        cnt = "  ".join(f"{k}:{c.get(k,0)}" for k in ROUTED_CANDS)
        print(f"target {gb:>5.0f} GB -> actual {total:5.1f} GB  fixed {fixed/1e9:4.1f}  "
              f"routed {rtot/1e9:5.1f}  dKL_saved {saved:10.6f}   [{cnt}]  -> {os.path.basename(outp)}")
        if c.get("MXFP4", 0):
            print(f"#   note: {c['MXFP4']//3} rich layers -> for TC prefill, splice to CUTLASS "
                  f"type-40 via converter/splice_cutlass_mxfp4.py after quantizing")

if __name__ == "__main__":
    main()
