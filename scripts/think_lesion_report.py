
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis_out/think_leak/lesion")
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()
    C = {}
    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    npid = max((max(json.loads(x)["pid"] for x in open(f) if x.strip()) for f in files), default=-1) + 1
    for f in files:
        R, seen = [], set()
        for x in open(f):
            try:
                r = json.loads(x)
            except json.JSONDecodeError:
                continue
            if (r["pid"], r["sid"]) in seen:
                continue
            seen.add((r["pid"], r["sid"]))
            R.append(r)
        k, n, ln, mk, st = np.zeros(npid), np.zeros(npid), np.zeros(npid), np.zeros(npid), np.zeros(npid)
        for r in R:
            k[r["pid"]] += r["correct"]; n[r["pid"]] += 1; ln[r["pid"]] += r["n_tok"]; mk[r["pid"]] += r["n_markers"]; st[r["pid"]] += r["stopped"]
        C[os.path.basename(f)[:-6]] = dict(k=k, n=n, ln=ln, mk=mk, st=st)
    if "rl_none" not in C or "base_none" not in C:
        print("need rl_none and base_none; have", list(C)); return
    rng = np.random.default_rng(0)
    full = C["rl_none"]["n"].sum()
    P = npid
    idx = rng.integers(0, P, (args.n_boot, P))
    acc = lambda c, i=None: (c["k"].sum() / max(c["n"].sum(), 1)) if i is None else (c["k"][i].sum(-1) / np.maximum(c["n"][i].sum(-1), 1))
    G, Gb = acc(C["rl_none"]) - acc(C["base_none"]), acc(C["rl_none"], idx) - acc(C["base_none"], idx)
    print(f"base_none acc={acc(C['base_none']):.3f}  rl_none acc={acc(C['rl_none']):.3f}  G={G:+.3f} [{np.quantile(Gb, .025):+.3f},{np.quantile(Gb, .975):+.3f}]")
    print(f"{'cond':14s} {'n':>4s} {'acc':>6s} {'len':>6s} {'stop':>5s} {'mk/1k':>6s} | effect (abs, 95% CI) | fraction of G (95% CI)")
    for name, c in sorted(C.items(), key=lambda kv: -kv[1]["n"].sum()):
        tgt = name.split("_")[0]
        partial = " PARTIAL" if c["n"].sum() < full else ""
        ref = C["rl_none"] if tgt == "rl" else C["base_none"]
        d, db = acc(c) - acc(ref), acc(c, idx) - acc(ref, idx)
        fr, frb = (-d / G, -db / Gb) if tgt == "rl" else (d / G, db / Gb)
        lab = "removed" if tgt == "rl" else "recovered"
        print(f"{name:14s} {int(c['n'].sum()):4d} {acc(c):6.3f} {c['ln'].sum() / c['n'].sum():6.0f} {c['st'].sum() / c['n'].sum():5.2f} "
              f"{1000 * c['mk'].sum() / c['ln'].sum():6.2f} | {d:+.3f} [{np.quantile(db, .025):+.3f},{np.quantile(db, .975):+.3f}] | "
              f"{lab} {fr:+.2f} [{np.quantile(frb, .025):+.2f},{np.quantile(frb, .975):+.2f}]{partial}")


if __name__ == "__main__":
    main()
