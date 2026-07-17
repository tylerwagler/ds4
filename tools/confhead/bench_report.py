#!/usr/bin/env python3
"""Join a bench leg's manifest with its fused-stats log and report per-cell
eff_tps / alpha / tok-per-step (medians across runs).

Requests are segmented at n_batch==1 in log order; the manifest's generating
entries map 1:1 onto segments (bench_driver is strictly sequential).

Usage: bench_report.py <out_dir> [<out_dir2> ...]   (each dir from bench.sh)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_stats import parse_requests, agg

def leg(d):
    manifest = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
    gen = [e for e in manifest if e["status"] == "ok" and e["completion_tokens"] >= 1]
    segs = [s for s in parse_requests(os.path.join(d, "server.log")) if s]
    # hard-fail: zipping past a divergence would misattribute every later
    # segment to the wrong cell (e.g. a client-side timeout whose request the
    # server kept serving). retrain.py asserts the same join.
    assert len(segs) == len(gen), \
        f"{d}: {len(segs)} log segments vs {len(gen)} ok requests -- cannot attribute cells"
    cells = {}
    for e, s in zip(gen, segs):
        key = (e["workload"], e["depth"], e["temperature"])
        cells.setdefault(key, []).append(agg(s))
    return cells

def main():
    for d in sys.argv[1:]:
        print(f"\n=== {d} ===")
        cells = leg(d)
        print(f"{'cell':32s} {'runs':4s} {'eff_tps(med)':12s} {'alpha':7s} {'tok/step':8s} {'ms/step':8s}")
        for key in sorted(cells, key=str):
            rs = cells[key]
            med = lambda k: float(np.median([r[k] for r in rs]))
            name = f"{key[0]}/{key[1]}/t{key[2]}"
            print(f"{name:32s} {len(rs):4d} {med('eff_tps'):12.2f} {med('alpha'):7.3f} "
                  f"{med('tok_step'):8.2f} {med('ms_step'):8.1f}")

if __name__ == "__main__":
    main()
