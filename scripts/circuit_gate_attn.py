
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import circuit_gate as cg
from think_leak_gen import render
from think_vector import load_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--gate_json", required=True)
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--max_resp", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pcts", default="1,5")
    ap.add_argument("--n_rand", type=int, default=5)
    ap.add_argument("--with_last_prefix", action="store_true", help="also knock out the query at the last prefix position (the "
                    "last token of the block itself), as circuit_gate.py does by default; default here: response positions only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import transformers.models.qwen3.modeling_qwen3 as mq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mq.eager_attention_forward = cg.patched_eager
    cg.CTX["qoff"] = 1 if args.with_last_prefix else 0
    tok = AutoTokenizer.from_pretrained(args.base)
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    df = pd.read_parquet(args.data)
    S_T = [render(tok, p, None, enable_thinking=True) for p in df["prompt"]]
    S_N = [render(tok, p, None, enable_thinking=False) for p in df["prompt"]]
    rows = sorted([r for r in load_rows(args.gen_dir, "base_nothink") if r["sid"] == 0], key=lambda r: r["pid"])[: args.limit or None]
    blk_ids = E("<think>\n\n</think>\n\n")
    items = []
    for r in rows:
        pn = E(S_N[r["pid"]])
        assert pn[-len(blk_ids):] == blk_ids
        items.append({"pid": r["pid"], "PN": pn, "PT": E(S_T[r["pid"]]) + E("<think>\n"), "PD": pn[: len(pn) - len(blk_ids)],
                      "blk": (len(pn) - len(blk_ids), len(pn)),
                      "y": r["token_ids"][: args.max_resp]})

    g = json.load(open(args.gate_json))
    M = np.array(g["mass"]["base_N"])
    L, H = M.shape
    order = np.argsort(-M.ravel())
    rng = np.random.default_rng(args.seed)
    sets = {}
    for p in [float(x) for x in args.pcts.split(",")]:
        n = int(round(p / 100 * L * H))
        sets[f"attn{p:g}"] = [(int(i) // H, int(i) % H) for i in order[:n]]
        for r in range(args.n_rand):
            sets[f"rand{p:g}_{r}"] = [(int(i) // H, int(i) % H) for i in rng.choice(L * H, n, replace=False)]
    sets["all"] = [(l, h) for l in range(L) for h in range(H)]

    def knock_of(hl):
        d = {}
        for l, h in hl:
            d.setdefault(l, []).append(h)
        return d

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()
    assert model.config.num_hidden_layers == L and model.config.num_attention_heads == H
    mid = L // 2
    acc = {}

    def add(name, hs, n, k):
        a = acc.setdefault(name, {"hs": 0.0, "n": 0, "kl": 0.0})
        a["hs"] = a["hs"] + hs
        a["n"] += n
        a["kl"] += k * n

    s0 = 0 if args.with_last_prefix else 1
    first = []
    for i, it in enumerate(items):
        n = len(it["y"])
        hsT, lpT, _ = cg.run(model, it["PT"], it["y"], None, False, it["blk"])
        hsN, lpN, _ = cg.run(model, it["PN"], it["y"], None, False, it["blk"])
        first.append((cg.kl(lpT[:1], lpN[:1]), cg.kl(lpT, lpN), n))
        add("base_T", hsT, n, 0.0)
        add("base_N", hsN, n, cg.kl(lpT[s0:], lpN[s0:]))
        del lpN
        hs, lp, _ = cg.run(model, it["PD"], it["y"], None, False, it["blk"])
        add("delete", hs, n, cg.kl(lpT[s0:], lp[s0:]))
        del lp
        for name, hl in sets.items():
            hs, lp, _ = cg.run(model, it["PN"], it["y"], knock_of(hl), False, it["blk"])
            add(name, hs, n, cg.kl(lpT[s0:], lp[s0:]))
            del lp
        del lpT
        if i % 10 == 0:
            print(f"text {i + 1}/{len(items)}", flush=True)

    mN = acc["base_N"]["hs"] / acc["base_N"]["n"]
    vth = acc["base_T"]["hs"] / acc["base_T"]["n"] - mN
    nv = (vth * vth).sum(-1)
    klN = acc["base_N"]["kl"] / acc["base_N"]["n"]
    res = {}
    for name, a in acc.items():
        al = (((a["hs"] / a["n"] - mN) * vth).sum(-1) / nv).tolist()
        res[name] = {"alpha_mid": al[mid], "kl_rec": 1 - (a["kl"] / a["n"]) / klN, "n_heads": len(sets[name]) if name in sets else None}
    kf = float(np.mean([f[0] for f in first])); ka = float(np.mean([f[1] for f in first]))
    print(f"KL(think||no-think): first response token {kf:.3f} nats; mean over all tokens {ka:.4f}; from the 2nd token on {klN:.4f}", flush=True)
    json.dump({"args": vars(args), "L": L, "H": H, "mid": mid, "n_texts": len(items), "cond": res, "kl_first_token": kf, "kl_all_tokens": ka,
               "kl_from_second_token": klN,
               "sets": {k: [f"{l}.{h}" for l, h in v] for k, v in sets.items() if k != "all"}}, open(args.out, "w"))
    for name in ["delete"] + list(sets):
        print(f"{name:10s} n_heads {len(sets.get(name, [])):4d} alpha_mid {res[name]['alpha_mid']:+.3f} kl_rec {res[name]['kl_rec']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
