"""DSpark drafter converter: HF DeepSeek-V4-Flash-DSpark (mtp.0/1/2.*) -> ds4 dspark sidecar GGUF.

The engine's dspark_weights_bind (src/engine/weights.c:1648) loads `dspark.*` tensors; the HF
checkpoint ships them as `mtp.N.*`. Every required tensor has an exact source (verified), so this
is a name-map + byte-lossless repack, NOT a re-quantize:
  routed experts (w1/w3/w2, MXFP4 e2m1+ue8m0)      -> CUTLASS type-40 (pack on Sparky)
  shared experts + all attention incl. wo (MXFP8)  -> FP8_E4M3 type-38 (repack_mxfp8, byte-lossless)
  main_proj (MXFP8)                                -> FP8_E4M3
  hc_*, norms, sinks, markov, confidence, gate     -> F32 (upcast tiny tensors; lossless enough)
attn_output is MXFP8 now (P2e made the fused fp8_hc_expand / attention_output_low kernels FP8-aware;
oracle-zeroq8 ships attn_output_a/b as FP8_E4M3 and runs) -- no Q8_0.

Phase 1 (this file, runnable anywhere): build + VALIDATE the manifest against the loader's required
set. Phase 2 (Sparky, GB10+CUTLASS): stream the bytes through the p0conv codecs + mxfp4_pack CLI.

  usage: python3 dspark_convert.py manifest [SNAPSHOT_DIR]
"""
import json, os, re, sys, struct, collections

DEF_SNAP = ("/mnt/pve1-fast/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/"
            "snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7")

# gguf type ids used by the ds4 fork (see splice_cutlass_mxfp4.py BLK / gguf.c)
T_F32, T_F16, T_FP8, T_CUTLASS_MXFP4 = 0, 1, 38, 40

# ---------------------------------------------------------------------------
# ds4 loader required tensors (mirror of dspark_weights_bind, weights.c:1652-1689)
# ---------------------------------------------------------------------------
def required_dspark_tensors():
    req = {"dspark.main_proj.weight", "dspark.main_norm.weight"}
    per = ["hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm",
           "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv", "attn_kv_a_norm",
           "attn_sinks", "attn_output_a", "attn_output_b",
           "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm", "ffn_gate_inp",
           "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
           "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]
    for li in range(3):
        for t in per:
            req.add(f"dspark.{li}.{t}.weight")
    req |= {"dspark.2.markov_head.markov_w1.weight", "dspark.2.markov_head.markov_w2.weight",
            "dspark.2.confidence_head.proj.weight",
            "dspark.2.hc_head_base.weight", "dspark.2.hc_head_fn.weight",
            "dspark.2.hc_head_scale.weight", "dspark.2.norm.weight"}
    return req

# ---------------------------------------------------------------------------
# HF mtp.N.* -> ds4 dspark.* name map. Returns (dspark_name) or None to DROP.
# .scale tensors are folded into their .weight (returned as CONSUMED, not a
# separate output) -- handled by the caller keying on the .weight name.
# expert-index tensors fold into the stacked ffn_*_exps target.
# ---------------------------------------------------------------------------
# HF ffn expert matrix -> ds4 routed/shared target: gate=w1, up=w3, down=w2
_WMAP = {"w1": "gate", "w3": "up", "w2": "down"}

def mtp_to_dspark(name):
    if not name.startswith("mtp."):
        return None
    m = re.match(r"mtp\.(\d+)\.(.*)$", name)
    if not m:
        return None
    li, rest = int(m.group(1)), m.group(2)

    # ---- drafter-global pieces (live on specific blocks in HF, global in ds4) ----
    if rest == "main_proj.weight":  return "dspark.main_proj.weight"
    if rest == "main_proj.scale":   return "@scale:dspark.main_proj.weight"
    if rest == "main_norm.weight":  return "dspark.main_norm.weight"

    # ---- block-2 heads ----
    if rest.startswith("markov_head."):       return f"dspark.2.{rest}"           # markov_w1/w2.weight
    if rest == "confidence_head.proj.weight":  return "dspark.2.confidence_head.proj.weight"
    if rest in ("hc_head_base", "hc_head_fn", "hc_head_scale"): return f"dspark.2.{rest}.weight"
    if rest == "norm.weight":                  return "dspark.2.norm.weight"

    # ---- per-block HC mix tensors (HF has no .weight suffix) ----
    if rest in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
        return f"dspark.{li}.{rest}.weight"
    if rest in ("attn_norm.weight", "ffn_norm.weight"):
        return f"dspark.{li}.{rest}"

    # ---- attention block ----
    A = {"attn.wq_a.weight": "attn_q_a.weight", "attn.wq_b.weight": "attn_q_b.weight",
         "attn.wkv.weight": "attn_kv.weight",
         "attn.wo_a.weight": "attn_output_a.weight", "attn.wo_b.weight": "attn_output_b.weight",
         "attn.q_norm.weight": "attn_q_a_norm.weight", "attn.kv_norm.weight": "attn_kv_a_norm.weight",
         "attn.attn_sink": "attn_sinks.weight"}
    if rest in A:                       return f"dspark.{li}.{A[rest]}"
    if rest.endswith(".scale") and rest[:-6] + ".weight" in A:
        return f"@scale:dspark.{li}.{A[rest[:-6] + '.weight']}"

    # ---- FFN gate (router) ----
    if rest == "ffn.gate.weight":  return f"dspark.{li}.ffn_gate_inp.weight"
    if rest == "ffn.gate.bias":    return None                      # ds4 loader has no gate bias

    # ---- shared experts ----
    ms = re.match(r"ffn\.shared_experts\.(w[123])\.(weight|scale)$", rest)
    if ms:
        tgt = _WMAP[ms.group(1)]; kind = ms.group(2)
        wname = f"dspark.{li}.ffn_{tgt}_shexp.weight"
        return wname if kind == "weight" else f"@scale:{wname}"

    # ---- routed experts (fold all 256 into the stacked tensor) ----
    me = re.match(r"ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$", rest)
    if me:
        tgt = _WMAP[me.group(2)]
        wname = f"dspark.{li}.ffn_{tgt}_exps.weight"
        return wname if me.group(3) == "weight" else f"@scale:{wname}"

    return f"@UNMAPPED:{name}"

# target gguf type for a produced dspark tensor
def target_type(dspark_name):
    base = dspark_name.split(".")[-2] if dspark_name.endswith(".weight") else dspark_name
    if re.search(r"ffn_(gate|up|down)_exps$", dspark_name.replace(".weight", "")):
        return T_CUTLASS_MXFP4
    if re.search(r"(ffn_(gate|up|down)_shexp|attn_q_a|attn_q_b|attn_kv|attn_output_a|attn_output_b)$",
                 dspark_name.replace(".weight", "")) or dspark_name == "dspark.main_proj.weight":
        return T_FP8
    return T_F32   # hc_*, norms, sinks, gate_inp, markov, confidence, hc_head

# ---------------------------------------------------------------------------
# safetensors dtype map + header reader (metadata only; no bulk data read)
# ---------------------------------------------------------------------------
def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def build_manifest(snap):
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    mtp = {n: sh for n, sh in wm.items() if n.startswith("mtp.")}
    # cache shard headers we touch
    hdrs, shape, dtype = {}, {}, {}
    for n, sh in mtp.items():
        if sh not in hdrs:
            hdrs[sh] = st_header(os.path.join(snap, sh))
        meta = hdrs[sh][n]
        shape[n], dtype[n] = meta["shape"], meta["dtype"]

    produced = collections.defaultdict(lambda: {"srcs": [], "scales": [], "experts": 0})
    dropped, unmapped = [], []
    for n in sorted(mtp):
        d = mtp_to_dspark(n)
        if d is None:
            dropped.append(n); continue
        if d.startswith("@UNMAPPED:"):
            unmapped.append(n); continue
        if d.startswith("@scale:"):
            produced[d[len("@scale:"):]]["scales"].append(n); continue
        produced[d]["srcs"].append(n)
        if ".ffn_" in d and d.endswith("_exps.weight"):
            produced[d]["experts"] += 1
    return produced, dropped, unmapped, shape, dtype

def cmd_manifest(snap):
    produced, dropped, unmapped, shape, dtype = build_manifest(snap)
    req = required_dspark_tensors()
    got = set(produced)
    missing = sorted(req - got)
    extra = sorted(got - req)

    TN = {T_F32: "F32", T_F16: "F16", T_FP8: "MXFP8", T_CUTLASS_MXFP4: "CUTLASS_MXFP4"}
    by_fmt = collections.Counter()
    print(f"{'dspark tensor':<40}{'type':<15}{'src dtype':<10}{'shape / experts'}")
    print("-" * 92)
    for d in sorted(produced):
        info = produced[d]; t = target_type(d)
        by_fmt[TN[t]] += 1
        s0 = info["srcs"][0] if info["srcs"] else "?"
        if info["experts"]:
            desc = f"{info['experts']} experts x {dtype.get(s0,'?')}"
        else:
            desc = f"{dtype.get(s0,'?'):<9} {shape.get(s0,'?')}"
        sc = f"  (+{len(info['scales'])} scale)" if info["scales"] else ""
        print(f"{d:<40}{TN[t]:<15}{'':<10}{desc}{sc}")
    print("-" * 92)
    print(f"produced tensors: {len(produced)}   by target type: {dict(by_fmt)}")
    print(f"dropped (intentional): {len(dropped)}  e.g. {dropped[:3]}")
    print(f"\nVALIDATION vs ds4 loader required set ({len(req)} tensors):")
    print(f"  MISSING (loader needs, not produced): {len(missing)}")
    for m in missing: print(f"     - {m}")
    print(f"  EXTRA (produced, loader ignores):     {len(extra)}")
    for e in extra: print(f"     + {e}")
    print(f"  UNMAPPED HF tensors:                  {len(unmapped)}")
    for u in unmapped[:10]: print(f"     ? {u}")
    ok = not missing and not unmapped
    print(f"\n{'OK — mapping complete and covers the loader' if ok else 'INCOMPLETE — see above'}")
    return 0 if ok else 1

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "manifest"
    snap = sys.argv[2] if len(sys.argv) > 2 else DEF_SNAP
    if cmd == "manifest":
        sys.exit(cmd_manifest(snap))
    print(f"unknown command: {cmd}"); sys.exit(2)
