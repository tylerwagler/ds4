#!/usr/bin/env python3
"""measure_layer_kl.py — measured per-layer expert-promotion KL for prisma_alloc.

For each MoE layer L, measures the real end-to-end KL divergence between the
all-cheap base model and the same model with layer L's routed experts swapped
(via ds4 --expert-overlay) to the all-rich donor's lossless MXFP4 tensors.
That KL is the distortion layer L's cheap quantization contributes to the
output distribution == the benefit of promoting it, which is exactly the
objective prisma_alloc.py's knapsack trades against bytes.

Method (AURA-era, measured-not-proxied): one --kl-ref-dump run on the base
(the anchor), then one --kl-score run per layer with the overlay applied.
Runs are strictly serial (ds4 holds an instance lock). Resumable: layers
already present in --out are skipped.

Usage:
  measure_layer_kl.py --ds4 ./ds4 --base cheap.gguf --donor rich.gguf \
      --calib gguf-tools/imatrix/dataset/rendered_prompts.txt \
      --out kl.json [--layers 0-42] [--tokens 2048] [--ctx 4096] [--stride 4]

Output (--out): {"17": {"dkl_promote": 1.23e-3, "stderr": 4.5e-5}, ...}
Feed to: prisma_alloc.py --kl kl.json
"""
import argparse, json, os, re, subprocess, sys, tempfile, time


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_ds4(args, cmd, log_name):
    log_path = os.path.join(args.log_dir, log_name)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
    if proc.returncode != 0:
        sys.exit(f"error: ds4 failed (exit {proc.returncode}); see {log_path}")
    return proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--base", required=True, help="all-cheap base GGUF (the anchor)")
    ap.add_argument("--donor", required=True, help="all-rich donor GGUF for overlays")
    ap.add_argument("--calib", required=True, help="calibration text file")
    ap.add_argument("--out", required=True, help="kl.json output (resumable)")
    ap.add_argument("--ref-dump", help="reference logit dump path (default: <out>.ref.bin)")
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--tokens", type=int, default=2048, help="scored tokens per run")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--log-dir", default=None)
    a = ap.parse_args()

    ref_dump = a.ref_dump or (a.out + ".ref.bin")
    a.log_dir = a.log_dir or tempfile.mkdtemp(prefix="measure_layer_kl_")
    layers = parse_layers(a.layers)
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}

    common = [a.ds4, "-m", a.base, "--cuda", "--kl-file", a.calib,
              "-n", str(a.tokens), "-c", str(a.ctx), "--kl-stride", str(a.stride)]

    if not os.path.exists(ref_dump):
        print(f"[ref] dumping base logits -> {ref_dump}", flush=True)
        t0 = time.time()
        out = run_ds4(a, common + ["--kl-ref-dump", ref_dump], "ref_dump.log")
        print(f"[ref] {out.strip()}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"[ref] reusing {ref_dump}", flush=True)

    kl_re = re.compile(r"kl_mean=([0-9.eE+-]+) kl_stderr=([0-9.eE+-]+)")
    for L in layers:
        key = str(L)
        if key in results:
            print(f"[layer {L}] cached: {results[key]}", flush=True)
            continue
        t0 = time.time()
        cmd = common + ["--kl-score", ref_dump,
                        "--expert-overlay", f"{a.donor}:blk.{L}."]
        out = run_ds4(a, cmd, f"layer_{L}.log")
        m = kl_re.search(out)
        if not m:
            sys.exit(f"error: could not parse kl output for layer {L}: {out!r}")
        results[key] = {"dkl_promote": float(m.group(1)), "stderr": float(m.group(2))}
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"[layer {L}] dkl_promote={m.group(1)} stderr={m.group(2)} "
              f"({time.time()-t0:.0f}s, {len(results)}/{len(layers)} done)", flush=True)

    print(f"done: {a.out} ({len(results)} layers); logs in {a.log_dir}")


if __name__ == "__main__":
    main()
