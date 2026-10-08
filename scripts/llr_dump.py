import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render
from think_vector import load_rows, split_opener


@torch.no_grad()
def lp_sum(model, pad_id, prefix, y, max_tok):
    dev = next(model.parameters()).device
    y = y[:max_tok]
    ids = torch.tensor([prefix + y], dtype=torch.long, device=dev)
    logits = model(input_ids=ids, use_cache=False).logits
    lg = logits[0, len(prefix) - 1:len(prefix) + len(y) - 1].float().log_softmax(-1)
    return float(lg.gather(-1, torch.tensor(y, device=dev)[:, None]).sum()), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--tags", nargs="*", default=[], help="default: every *.jsonl in gen_dir")
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--template_kwarg", action="store_true")
    ap.add_argument("--think_opener", default="<think>\\n")
    ap.add_argument("--max_tok", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    opener = args.think_opener.encode().decode("unicode_escape")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    df = pd.read_parquet(args.data)
    df = df.head(args.limit) if args.limit else df
    kw_t = {"enable_thinking": True} if args.template_kwarg else {}
    kw_n = {"enable_thinking": False} if args.template_kwarg else {}
    op_ids = E(opener) if opener else []
    openers = sorted([o for o in [op_ids, E("<think>\n"), E("<think>")] if o], key=len, reverse=True)
    P_T = [E(render(tok, p, None, **kw_t)) + op_ids for p in df["prompt"]]
    P_N = [E(render(tok, p, None, **kw_n)) for p in df["prompt"]]

    tags = args.tags or sorted(os.path.splitext(f)[0] for f in os.listdir(args.gen_dir) if f.endswith(".jsonl"))
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for tag in tags:
        if tag in out:
            print("skip", tag, flush=True)
            continue
        rows = load_rows(args.gen_dir, tag)
        recs = []
        for r in sorted(rows, key=lambda r: (r["pid"], r["sid"])):
            y = split_opener(r["token_ids"], openers)[1]
            if len(y) == 0:
                continue
            a, n = lp_sum(model, pad_id, P_T[r["pid"]], y, args.max_tok)
            b, _ = lp_sum(model, pad_id, P_N[r["pid"]], y, args.max_tok)
            recs.append({"pid": r["pid"], "sid": r["sid"], "llr": (a - b) / n, "n_tok": n})
        out[tag] = recs
        json.dump(out, open(args.out, "w"))
        m = sum(x["llr"] for x in recs) / len(recs)
        print(f"{tag}: n={len(recs)} mean per-text llr {m:+.4f}", flush=True)


if __name__ == "__main__":
    main()
