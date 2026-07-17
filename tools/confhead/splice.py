#!/usr/bin/env python3
"""Confidence-head retrain, stage 3: splice a retrained head into a gguf.

Copies SRC to OUT and overwrites dspark.2.confidence_head.proj.weight (F32
[4352], same size, no header changes) with the retrained weights. Works on
both the sidecar drafter ggufs and the merged model artifacts (same tensor
name; alignment read from general.alignment).

NOTE (load precedence, verified in session.c engine-open logic): when the
main gguf contains a merged drafter (dspark.main_proj.weight present),
--dspark PATH is IGNORED — the merged branch wins. So to measure a new head
against the shipped merged v5mx you must splice into a COPY of the merged
artifact; a sidecar-only splice can never override a merged head without a
loader change (Tyler's call, not this tool's).

Usage: splice.py --src <in.gguf> --out <out.gguf> --proj <proj_new.npy>
"""
import argparse, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_io import parse, patch_f32

NAME = "dspark.2.confidence_head.proj.weight"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proj", required=True)
    args = ap.parse_args()

    w = np.load(args.proj).astype(np.float32)
    assert w.shape == (4352,), w.shape
    tens, _ = parse(args.src)
    assert NAME in tens, f"{NAME} not in {args.src}"
    typ, dims, _ = tens[NAME]
    assert typ == 0 and int(np.prod(dims)) == 4352, (typ, dims)

    if os.path.abspath(args.src) == os.path.abspath(args.out):
        raise SystemExit("refusing in-place splice of a versioned artifact; pick a new --out")
    print(f"copying {args.src} -> {args.out} ({os.path.getsize(args.src)/2**30:.1f} GiB)")
    shutil.copyfile(args.src, args.out)
    abs_off, ne = patch_f32(args.out, NAME, w)
    print(f"spliced {NAME} ({ne} f32) at byte {abs_off} in {args.out}")

if __name__ == "__main__":
    main()
