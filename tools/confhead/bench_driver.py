#!/usr/bin/env python3
"""Fixed-seed request suites for the tau re-sweep and old-vs-new-head A/B.

Suites (prompts reuse collect_driver's builders; seeds fixed per cell so legs
are comparable; spec t/s is stochastic across seeds -- never compare runs
with different seeds):
  sweep: sampled-only (t=0.7), {prose,structured} x {2k,8k}, 1 run
  ab:    {prose,structured} x {2k,8k,28k} x {t=0.7 sampled, t=0 greedy},
         --runs N repeats (same seeds; medians measure batch-FP noise)

Greedy (t=0) completions are saved to <out>/texts/ for the byte-stream A/B
(trimming changes verify batch size, so small divergence is the documented
batch-shape near-tie caveat, not a failure).
"""
import argparse, json, os, sys, time, urllib.request, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_driver import BUILDERS

GEN_TOKENS = 512

def cells(suite):
    if suite == "sweep":
        return [(w, d, 0.7) for d in (2000, 8000) for w in ("prose", "structured")]
    if suite == "ab":
        return [(w, d, t) for d in (2000, 8000, 28000)
                for w in ("prose", "structured") for t in (0.7, 0.0)]
    raise SystemExit(f"unknown suite {suite}")

def request(port, prompt, temperature, seed, timeout):
    body = {"model": "d", "prompt": prompt, "max_tokens": GEN_TOKENS,
            "temperature": temperature, "seed": seed}
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}),
        timeout=timeout)
    return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--suite", choices=("sweep", "ab"), required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--texts", default=None, help="dir to save greedy completions")
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()
    if args.texts:
        os.makedirs(args.texts, exist_ok=True)

    entries = []
    idx = 0
    for run in range(args.runs):
        for (w, d, t) in cells(args.suite):
            prompt = BUILDERS[w](d, 0)
            # fixed per cell, same every leg/run/process (python hash() is
            # per-process randomized -- never use it for seeds)
            seed = 7000 + zlib.crc32(f"{w}/{d}/{t}".encode()) % 1000
            tag = f"{w}/{d}/t{t}/r{run}"
            ent = {"idx": idx, "tag": tag, "workload": w, "depth": d,
                   "temperature": t, "run": run, "seed": seed,
                   "status": "pending", "prompt_tokens": 0, "completion_tokens": 0}
            t0 = time.time()
            try:
                resp = request(args.port, prompt, t, seed, args.timeout)
                u = resp["usage"]
                ent.update(status="ok", prompt_tokens=u["prompt_tokens"],
                           completion_tokens=u["completion_tokens"])
                if args.texts and t == 0.0:
                    with open(os.path.join(args.texts, f"{idx:03d}_{w}_{d}_r{run}.txt"), "w") as f:
                        f.write(resp["choices"][0]["text"])
                print(f"[{idx:2d}] {tag}: prompt={u['prompt_tokens']} "
                      f"gen={u['completion_tokens']} in {time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                ent.update(status=f"fail: {e}")
                print(f"[{idx:2d}] {tag}: FAILED {e}", flush=True)
            entries.append(ent)
            idx += 1
            with open(args.manifest, "w") as f:
                for e2 in entries:
                    f.write(json.dumps(e2) + "\n")
    ok = sum(1 for e in entries if e["status"] == "ok")
    print(f"DONE: {ok}/{len(entries)} requests ok", flush=True)

if __name__ == "__main__":
    main()
