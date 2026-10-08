
import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render, text_stats
from think_vector import find_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--inp", required=True, help="think_vector_steer_L<l>_a<a>.jsonl")
    ap.add_argument("--vectors", required=True, help="think_vector_vectors.pt with v_own")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--max_total", type=int, default=32768)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1, help=">1: continue only every nparts-th truncated row; write <out>.part<i>.jsonl and stop")
    ap.add_argument("--merge", action="store_true", help="combine <out>.part*.jsonl over the input rows and write the final outputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from verl.utils.reward_score import default_compute_score

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gc = GenerationConfig.from_pretrained(args.base)
    eos_ids = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    eos_ids = sorted({e for e in eos_ids + [tok.eos_token_id] if e is not None})
    df = pd.read_parquet(args.data)
    S_N = [render(tok, p, None, enable_thinking=False) for p in df["prompt"]]
    rows = [json.loads(x) for x in open(args.inp)]
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = next(model.parameters()).device
    vec = (args.alpha * torch.load(args.vectors)["v_own"][args.layer]).to(dev, torch.bfloat16)
    layer = find_layers(model)[args.layer - 1]
    st = {"mask": None}

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1:
            h = h + vec
        elif st["mask"] is not None:
            h = h + st["mask"][:, :, None].to(h.dtype) * vec
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handle = layer.register_forward_hook(hook) if args.alpha != 0 else None
    todo = [i for i, r in enumerate(rows) if not r["stopped"]]
    print(f"{len(rows)} rows, {len(todo)} truncated, alpha {args.alpha}", flush=True)
    if args.merge:
        todo = []
        for i in range(args.nparts):
            for x in open(f"{args.out}.part{i}.jsonl"):
                r = json.loads(x)
                if r.get("continued"):
                    rows[r["row"]] = r
        assert all(r["stopped"] or r.get("continued") for r in rows), "a truncated row was not continued by any part"
    elif args.nparts > 1:
        todo = todo[args.part::args.nparts]
    torch.manual_seed(args.seed)
    for b in range(0, len(todo), args.bs):
        idx = todo[b:b + args.bs]
        pr = [tok(S_N[rows[i]["pid"]], add_special_tokens=False)["input_ids"] for i in idx]
        rs = [tok(rows[i]["text"], add_special_tokens=False)["input_ids"] for i in idx]
        T = max(len(p) + len(y) for p, y in zip(pr, rs))
        ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        mask = torch.zeros((len(idx), T))
        for r, (p, y) in enumerate(zip(pr, rs)):
            off = T - len(p) - len(y)
            ids[r, off:] = torch.tensor(p + y)
            att[r, off:] = 1
            mask[r, off + len(p):] = 1
        st["mask"] = mask.to(dev)
        room = args.max_total - min(len(y) for y in rs)
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=room, use_cache=True, eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        st["mask"] = None
        for r, i in enumerate(idx):
            g = out[r, T:].tolist()[:args.max_total - len(rs[r])]
            cut = next((k for k, t in enumerate(g) if t in eos_ids), None)
            keep = g if cut is None else g[:cut + 1]
            rows[i] = dict(rows[i], text=rows[i]["text"] + tok.decode(keep, skip_special_tokens=False), n_tok=len(rs[r]) + len(keep),
                           stopped=cut is not None, continued=True, prev_len=len(rs[r]), row=i)
        print(f"batch {b // args.bs + 1}/{(len(todo) + args.bs - 1) // args.bs} done", flush=True)
    if handle is not None:
        handle.remove()
    if args.nparts > 1 and not args.merge:
        with open(f"{args.out}.part{args.part}.jsonl", "w") as f:
            for r in rows:
                if r.get("continued"):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote part", args.part, flush=True)
        return
    corr = 0
    for r in rows:
        j = r["pid"]
        try:
            sc = default_compute_score(df.iloc[j].get("data_source", "math_dapo"), r["text"], df.iloc[j]["reward_model"]["ground_truth"])
            r["correct"] = bool((sc.get("score", 0) if isinstance(sc, dict) else sc) > 0.5)
        except Exception:
            r["correct"] = False
        corr += r["correct"]
    stt = [text_stats(r["text"], r["n_tok"]) for r in rows]
    tt = sum(r["n_tok"] for r in rows)
    agg = {"alpha": args.alpha, "n": len(rows), "acc": corr / len(rows), "mean_len": tt / len(rows),
           "stop_frac": sum(r["stopped"] for r in rows) / len(rows),
           "markers_per_1k": 1000.0 * sum(s["n_markers"] for s in stt) / max(1, tt)}
    with open(args.out + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(agg, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
