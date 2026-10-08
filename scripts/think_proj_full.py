
import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render
from think_vector import find_layers


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rollouts_dir", default="analysis_out/think_dyn/q8s2/rollouts")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--vectors", default="analysis_out/think_leak/qwen_s2/think_vector_fulllen_vectors.pt")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    P = pd.read_json(os.path.join(args.rollouts_dir, "prompts.jsonl"), lines=True)
    P_N = [E(render(tok, p, None, enable_thinking=False)) for p in P["prompt"]]
    rows = [json.loads(line) for line in open(os.path.join(args.rollouts_dir, f"{args.tag}.jsonl"))]
    v = torch.load(args.vectors)["v_Fn"][args.layer].double()
    vnorm = float(v.norm())
    vhat = (v / vnorm).cuda()

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    cap = {}
    layer = find_layers(model)[args.layer - 1]
    layer.register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    out = []
    for k, r in enumerate(rows):
        p, y = P_N[r["pid"]], r["token_ids"]
        if len(y) <= args.skip:
            continue
        model.model(input_ids=torch.tensor([p + y], device="cuda"), use_cache=False)
        h = cap["h"][0, len(p) + args.skip:len(p) + len(y)].double().mean(0)
        out.append({"pid": r["pid"], "sid": r["sid"], "proj": float(h @ vhat), "n_tok": len(y)})
        if k % 500 == 0:
            print(f"{k}/{len(rows)}", flush=True)
    json.dump({"vnorm": vnorm, "layer": args.layer, "skip": args.skip, "rows": out}, open(args.out, "w"))
    print("done", len(out), flush=True)


if __name__ == "__main__":
    main()
