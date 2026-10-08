
import argparse
import json
import os
import sys
from contextlib import nullcontext

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render, text_stats
from think_vector import find_layers


def read_rows(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--subspace", required=True)
    ap.add_argument("--target", choices=["rl", "base"], required=True)
    ap.add_argument("--family", default="none", choices=["none", "think", "pca", "ctrl", "rand", "think_x_ctrl", "ctrl_x_think", "head"])
    ap.add_argument("--heads", default="", help="family head: comma list layer.head (0-based decoder layer); the head's attention output "
                    "(its slice of the o_proj input) is replaced by the source's, instead of a residual-subspace edit")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layers", default="1-35")
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--prev", required=True, help="jsonl written by think_lesion.py at the smaller cap")
    ap.add_argument("--max_total", type=int, default=32768, help="new cap on prefix + continuation tokens")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--greedy", action="store_true", help="argmax decoding (for testing)")
    ap.add_argument("--no_compact", action="store_true", help="keep finished rows in the batch (for testing)")
    ap.add_argument("--limit", type=int, default=0, help="first N truncated rows only (smoke tests)")
    ap.add_argument("--test_budgets", default="", help="comma list: force per-row continuation budgets (testing compaction)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig
    from verl.utils.reward_score import default_compute_score

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = GenerationConfig.from_pretrained(args.base)
    eos = set((gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]) + [tok.eos_token_id])
    df = pd.read_parquet(args.data)
    prompts = [render(tok, p, None, enable_thinking=False) for p in df["prompt"]]

    prev = {}
    for r in read_rows(args.prev):
        prev.setdefault((r["pid"], r["sid"]), r)
    jobs = sorted(k for k, r in prev.items() if not r["stopped"])
    if args.limit:
        jobs = jobs[:args.limit]

    lock = args.out + ".lock"
    jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    try:
        os.makedirs(lock)
    except FileExistsError:
        owner = open(os.path.join(lock, "jobid")).read().strip() if os.path.exists(os.path.join(lock, "jobid")) else ""
        import subprocess
        live = owner and owner != jid and subprocess.run(["squeue", "-h", "-j", owner, "-t", "R"], capture_output=True, text=True).stdout.strip()
        if live:
            print(f"job {owner} is already writing {args.out}; exiting", flush=True)
            return
    open(os.path.join(lock, "jobid"), "w").write(jid)

    done = set()
    if os.path.exists(args.out + ".jsonl"):
        done = {(r["pid"], r["sid"]) for r in read_rows(args.out + ".jsonl")}
    jobs = [x for x in jobs if x not in done]
    print(f"{len(prev)} previous rows, {sum(not r['stopped'] for r in prev.values())} truncated, {len(jobs)} to continue", flush=True)
    if not jobs:
        return

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    pm = PeftModel.from_pretrained(model, args.adapter, adapter_name="rl")
    pm.eval()
    dev = next(pm.parameters()).device
    layers = find_layers(pm)
    L, d = len(layers), model.config.hidden_size
    a, b = (int(x) for x in args.layers.split("-"))
    patched = [l for l in range(a, b + 1) if 1 <= l < L] if args.family not in ("none", "head") else []
    heads = {}
    if args.family == "head":
        for x in args.heads.replace("+", ",").split(","):
            li, h = (int(y) for y in x.split("."))
            heads.setdefault(li, []).append(h)
    Q = {}
    if args.family == "rand":
        g_ = torch.Generator().manual_seed(1234)
        for l in patched:
            Q[l] = torch.linalg.qr(torch.randn(d, args.k, generator=g_))[0].to(dev, torch.float32)
    elif "_x_" in args.family:
        sub = torch.load(args.subspace)
        fa, fb = args.family.split("_x_")
        for l in patched:
            A_, B_ = sub["Q"][fa][l][:, :args.k].float(), sub["Q"][fb][l][:, :args.k].float()
            Q[l] = torch.linalg.qr(A_ - B_ @ (B_.T @ A_))[0].to(dev, torch.float32)
    elif args.family != "none":
        sub = torch.load(args.subspace)
        for l in patched:
            Q[l] = sub["Q"][args.family][l][:, :args.k].to(dev, torch.float32)

    st = {"mode": None, "src": {}, "num": 0.0, "den": 0.0}

    def mk(l):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "source":
                st["src"][l] = h.detach()
                return None
            diff = (st["src"][l] - h).float()
            edit = args.g * (diff @ Q[l]) @ Q[l].T
            st["num"] += float(edit.norm(dim=-1).sum())
            st["den"] += float(diff.norm(dim=-1).sum())
            h = h + edit.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    for l in patched:
        layers[l - 1].register_forward_hook(mk(l))

    hd = model.config.hidden_size // model.config.num_attention_heads
    hd = getattr(model.config, "head_dim", None) or hd

    def mk_head(li):
        def pre(_m, args_):
            x = args_[0]
            if st["mode"] == "source":
                st["src"][("h", li)] = x.detach()
                return None
            src = st["src"][("h", li)]
            x = x.clone()
            for h in heads[li]:
                sl = slice(h * hd, (h + 1) * hd)
                st["num"] += float((src[..., sl] - x[..., sl]).float().norm(dim=-1).sum())
                st["den"] += float(x[..., sl].float().norm(dim=-1).sum())
                x[..., sl] = src[..., sl]
            return (x,)
        return pre

    for li in heads:
        layers[li].self_attn.o_proj.register_forward_pre_hook(mk_head(li))
    if heads:
        patched = ["heads"]

    def ctx(which):
        return nullcontext() if which == "rl" else pm.disable_adapter()

    source = "base" if args.target == "rl" else "rl"

    @torch.no_grad()
    def step(ids, att, cache_t, cache_s):
        if patched:
            st["mode"] = "source"
            with ctx(source):
                pm(input_ids=ids, attention_mask=att, past_key_values=cache_s, use_cache=True)
        st["mode"] = "target"
        with ctx(args.target):
            return pm(input_ids=ids, attention_mask=att, past_key_values=cache_t, use_cache=True).logits[:, -1].float()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    eos_t = torch.tensor(sorted(eos), device=dev)
    for b0 in range(0, len(jobs), args.bs):
        chunk = jobs[b0:b0 + args.bs]
        P = [tok(prompts[j], add_special_tokens=False)["input_ids"] for j, _ in chunk]
        X = [tok(prev[(j, s)]["text"], add_special_tokens=False)["input_ids"] for j, s in chunk]
        full = [p + x for p, x in zip(P, X)]
        T = max(len(f) for f in full)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        for r, f in enumerate(full):
            ids[r, T - len(f):] = torch.tensor(f)
            att[r, T - len(f):] = 1
        ids, att = ids.to(dev), att.to(dev)
        budget = [max(0, args.max_total - len(x)) for x in X]
        if args.test_budgets:
            tb = [int(v) for v in args.test_budgets.split(",")]
            budget = [tb[i % len(tb)] for i in range(len(chunk))]
        cache_t, cache_s = DynamicCache(), DynamicCache()
        gen = [[] for _ in chunk]
        rows = list(range(len(chunk)))
        alive = torch.tensor([bdg > 0 for bdg in budget], device=dev)
        cur = ids
        while alive.any():
            logits = step(cur, att, cache_t, cache_s)
            nxt = logits.argmax(-1) if args.greedy else torch.multinomial(torch.softmax(logits, -1), 1)[:, 0]
            nxt = torch.where(alive, nxt, torch.full_like(nxt, pad))
            for bpos in alive.nonzero()[:, 0].tolist():
                gen[rows[bpos]].append(int(nxt[bpos]))
            stop = torch.isin(nxt, eos_t) | torch.tensor([len(gen[rows[bpos]]) >= budget[rows[bpos]] for bpos in range(len(rows))], device=dev)
            alive = alive & ~stop
            if not alive.any():
                break
            att = torch.cat([att, torch.ones(len(rows), 1, dtype=att.dtype, device=dev)], 1)
            cur = nxt[:, None]
            if not args.no_compact and not alive.all():
                keep = alive.nonzero()[:, 0]
                cache_t.batch_select_indices(keep)
                if patched:
                    cache_s.batch_select_indices(keep)
                att, cur = att[keep], cur[keep]
                rows = [rows[i] for i in keep.tolist()]
                alive = alive[keep]
        with open(args.out + ".jsonl", "a") as f:
            for (j, s), g, x in zip(chunk, gen, X):
                new_text = tok.decode(g, skip_special_tokens=False)
                text = prev[(j, s)]["text"] + new_text
                try:
                    r = default_compute_score(df.iloc[j].get("data_source", "math_dapo"), text, df.iloc[j]["reward_model"]["ground_truth"])
                    ok = bool((r.get("score", 0) if isinstance(r, dict) else r) > 0.5)
                except Exception:
                    ok = False
                row = {"pid": j, "sid": s, "text": text, "correct": ok, "stopped": bool(g and g[-1] in eos), "continued": True,
                       "prefix_tok": len(x), "new_tok": len(g), "new_token_ids": g}
                row.update(text_stats(text, len(x) + len(g)))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        del cache_t, cache_s
        torch.cuda.empty_cache()
        print(f"batch {b0 // args.bs + 1}/{(len(jobs) + args.bs - 1) // args.bs} done; delivered={st['num'] / max(st['den'], 1e-9):.3f}", flush=True)

    R = read_rows(args.out + ".jsonl")
    res = {"args": vars(args), "n_continued": len(R), "stop_frac": sum(r["stopped"] for r in R) / len(R),
           "mean_new_tok": sum(r["new_tok"] for r in R) / len(R), "acc_continued": sum(r["correct"] for r in R) / len(R),
           "delivered": st["num"] / max(st["den"], 1e-9) if st["den"] else None}
    json.dump(res, open(args.out + ".json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "args"})


if __name__ == "__main__":
    main()
