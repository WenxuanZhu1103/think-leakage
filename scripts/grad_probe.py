
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render
from think_vector import load_rows

MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def grpo_adv(r):
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return (r - r.mean()) / (sd + 1e-6)


def build_conditions(rows, scores, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["pid"], r["sid"] // 5)].append(i)
    W = {c: np.zeros(len(rows)) for c in ("full", "pos", "neg", "pos_plain", "pos_thinky", "shuffled", "sft")}
    info = defaultdict(list)
    for key, idx in groups.items():
        r = np.array([rows[i]["r_train"] for i in idx])
        if len(set(r.tolist())) < 2 or len(idx) != 5:
            continue
        A = grpo_adv(r)
        s = np.array([scores[(rows[i]["pid"], rows[i]["sid"])] for i in idx])
        wrong_mean = s[r <= 0].mean()
        perm = rng.permutation(len(idx))
        for j, i in enumerate(idx):
            W["full"][i] = A[j]
            W["sft"][i] = 1.0
            W["shuffled"][i] = A[perm[j]]
            if A[j] > 0:
                W["pos"][i] = A[j]
                if s[j] < wrong_mean:
                    W["pos_plain"][i] = A[j]
                    info["pos_plain_gap"].append(s[j] - wrong_mean)
                else:
                    W["pos_thinky"][i] = A[j]
                    info["pos_thinky_gap"].append(s[j] - wrong_mean)
            else:
                W["neg"][i] = A[j]
    return W, {k: (float(np.mean(v)), len(v)) for k, v in info.items()}


@torch.no_grad()
def mean_hidden(model, items, layer, dev):
    tot = None
    n = 0
    for p, y in items:
        ids = torch.tensor([p + y], device=dev)
        hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[layer][0]
        h = hs[len(p):len(p) + len(y)].double().sum(0)
        tot = h if tot is None else tot + h
        n += len(y)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rollouts_dir", default="analysis_out/think_dyn/q8s2/rollouts")
    ap.add_argument("--scores", default="analysis_out/think_dyn/q8s2/scores/base.jsonl")
    ap.add_argument("--aime_gen", default="analysis_out/think_leak/qwen_s2/gen")
    ap.add_argument("--aime_data", default="data/aime/test.parquet")
    ap.add_argument("--vectors", default="analysis_out/think_leak/qwen_s2/think_vector_fulllen_vectors.pt")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--scales", type=float, nargs="*", default=[1.0, 10.0])
    ap.add_argument("--max_tok", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rollouts only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    import pandas as pd
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    P = pd.read_json(os.path.join(args.rollouts_dir, "prompts.jsonl"), lines=True)
    P_N = [E(render(tok, p, None, enable_thinking=False)) for p in P["prompt"]]
    rows = [json.loads(line) for line in open(os.path.join(args.rollouts_dir, "base.jsonl"))]
    rows = sorted(rows, key=lambda r: (r["pid"], r["sid"]))
    sc = {(s["pid"], s["sid"]): s["llr"] for s in map(json.loads, open(args.scores))}
    W, info = build_conditions(rows, sc, args.seed)
    conds = list(W)
    used = np.nonzero(np.any(np.stack([W[c] != 0 for c in conds]), 0))[0]
    if args.limit:
        used = used[:args.limit]
    ntok = np.array([min(len(rows[i]["token_ids"]), args.max_tok) for i in range(len(rows))])
    norm = {c: float(ntok[used][W[c][used] != 0].sum()) for c in conds}
    print("conditions:", {c: int((W[c][used] != 0).sum()) for c in conds}, "info:", info, flush=True)

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map=dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, target_modules=MODS))
    Bs = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    for n, p in model.named_parameters():
        p.requires_grad_("lora_B" in n)

    os.makedirs(args.grad_dir, exist_ok=True)
    gpath = os.path.join(args.grad_dir, "grads.pt")
    if os.path.exists(gpath):
        G = torch.load(gpath)
        print("loaded", gpath, flush=True)
    else:
        G = {c: [torch.zeros_like(p, dtype=torch.float32) for _, p in Bs] for c in conds}
        model.train()
        for k, i in enumerate(used):
            r = rows[i]
            p, y = P_N[r["pid"]], r["token_ids"][:args.max_tok]
            ids = torch.tensor([p + y], device=dev)
            logits = model(input_ids=ids, use_cache=False).logits[0, len(p) - 1:len(p) + len(y) - 1].float()
            lp = logits.log_softmax(-1).gather(-1, torch.tensor(y, device=dev)[:, None]).sum()
            grads = torch.autograd.grad(lp, [q for _, q in Bs])
            for c in conds:
                w = W[c][i]
                if w != 0:
                    for acc, g in zip(G[c], grads):
                        acc.add_(g.float(), alpha=w / norm[c])
            if k % 200 == 0:
                print(f"grad {k}/{len(used)}", flush=True)
        torch.save(G, gpath)

    model.eval()
    model = model.float()
    df = pd.read_parquet(args.aime_data)
    PA = [E(render(tok, p, None, enable_thinking=False)) for p in df["prompt"]]
    Fn = [(PA[r["pid"]], r["token_ids"]) for r in load_rows(args.aime_gen, "base_nothink") if r["sid"] == 0]
    v = torch.load(args.vectors)["v_Fn"][args.layer].double().to(dev)
    v2 = float(v @ v)
    for _, p in Bs:
        p.data.zero_()
    mu0 = mean_hidden(model, Fn, args.layer, dev)
    gnorm = {c: float(torch.sqrt(sum((g ** 2).sum() for g in G[c]))) for c in conds}
    res ={"info": info, "n": {c: int((W[c][used] != 0).sum()) for c in conds}, "gnorm": gnorm,
           "cos_with_full": {}, "alpha": {}}
    flat_full = torch.cat([g.flatten() for g in G["full"]])
    for c in conds:
        flat = torch.cat([g.flatten() for g in G[c]])
        res["cos_with_full"][c] = float(flat @ flat_full / (flat.norm() * flat_full.norm() + 1e-30))
        res["alpha"][c] = {}
        for rule in ("adam", "sgd"):
            for s in args.scales:
                sgd_norm = args.lr * s * float(torch.sqrt(sum(torch.sign(g).abs().sum() for g in G["full"])))
                for (_, p), g in zip(Bs, G[c]):
                    if rule == "adam":
                        upd = args.lr * s * torch.sign(g)
                    else:
                        upd = g * (sgd_norm / (gnorm[c] + 1e-30))
                    p.data.copy_(upd.to(p.dtype))
                mu = mean_hidden(model, Fn, args.layer, dev)
                a = float((mu - mu0) @ v / v2)
                res["alpha"][c][f"{rule}_x{s:g}"] = a
                print(f"{c:11s} {rule} x{s:g}: alpha {a:+.5f}", flush=True)
                json.dump(res, open(args.out, "w"), indent=1)
        for _, p in Bs:
            p.data.zero_()
    json.dump(res, open(args.out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
