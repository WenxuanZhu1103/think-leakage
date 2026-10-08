
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_vector import load_rows, split_opener


@torch.no_grad()
def binned(model, pad_id, items, n_bins, bin_len, bs):
    dev = next(model.parameters()).device
    K = n_bins + 1
    enc = [(p, y[:n_bins * bin_len]) for _, p, y in items if len(y) > 0]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]))
    S, C = None, torch.zeros(K, dtype=torch.float64)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        T = max(len(enc[i][0]) + len(enc[i][1]) for i in idx)
        ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        msk = torch.zeros((len(idx), K, T))
        for r, i in enumerate(idx):
            p, y = enc[i]
            n = len(p) + len(y)
            ids[r, :n] = torch.tensor(p + y)
            att[r, :n] = 1
            msk[r, 0, len(p) - 1] = 1
            for k in range(n_bins):
                a, e = len(p) + k * bin_len, min(n, len(p) + (k + 1) * bin_len)
                if a < e:
                    msk[r, k + 1, a:e] = 1
        hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        m = msk.to(dev)
        per = torch.stack([torch.einsum("bkt,btd->kd", m, h.float()) for h in hs], 1).double().cpu()
        S = per if S is None else S + per
        C += msk.sum((0, 2)).double()
    return S, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--gen_dir", required=True, help="dir with base_nothink.jsonl, base_think.jsonl, <rl_tag>.jsonl")
    ap.add_argument("--rl_tag", default="s67")
    ap.add_argument("--think_opener", default="<think>\n")
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--bin", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    rows_n, rows_t, rows_r = (load_rows(args.gen_dir, t) for t in ("base_nothink", "base_think", args.rl_tag))
    op_ids = E(args.think_opener)
    openers = sorted([o for o in (op_ids, E("<think>\n"), E("<think>")) if o], key=len, reverse=True)
    P_N = {r["pid"]: E(r["prompt"]) for r in rows_n}
    P_T = {r["pid"]: E(r["prompt"]) + op_ids for r in rows_t}

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    pm = PeftModel.from_pretrained(model, args.adapter, adapter_name="rl")
    pm.eval()
    L = model.config.num_hidden_layers
    layers = sorted({round(L * q) for q in (0.25, 0.5, 0.75)})
    nb, bl, bs = args.n_bins, args.bin, args.bs

    own_t_items = []
    for r in rows_t:
        op, rest = split_opener(r["token_ids"], openers)
        own_t_items.append((r["pid"], E(r["prompt"]) + op, rest))
    fixed = [(r["pid"], r["token_ids"]) for r in rows_n]
    res = {}
    with pm.disable_adapter():
        res["own_base_nothink"] = binned(pm, pad_id, [(r["pid"], P_N[r["pid"]], r["token_ids"]) for r in rows_n], nb, bl, bs)
        res["own_base_think"] = binned(pm, pad_id, own_t_items, nb, bl, bs)
        res["fix_base_N"] = res["own_base_nothink"]
        res["fix_base_T"] = binned(pm, pad_id, [(p, P_T[p], y) for p, y in fixed], nb, bl, bs)
    res["own_rl_nothink"] = binned(pm, pad_id, [(r["pid"], P_N[r["pid"]], r["token_ids"]) for r in rows_r], nb, bl, bs)
    res["fix_rl_N"] = binned(pm, pad_id, [(p, P_N[p], y) for p, y in fixed], nb, bl, bs)

    mean = {k: (S / C.clamp(min=1)[:, None, None]) for k, (S, C) in res.items()}
    cnt = {k: C.tolist() for k, (S, C) in res.items()}

    def all_mean(k):
        S, C = res[k]
        return S[1:].sum(0) / C[1:].sum()

    summary = {"layers": layers, "bins": ["boundary"] + [f"{k * bl}-{(k + 1) * bl}" for k in range(nb)], "counts": cnt, "own": {}, "fixed_alpha": {}}
    mu0, mu1 = all_mean("own_base_nothink"), all_mean("own_base_think")
    vfix = all_mean("fix_base_T") - all_mean("fix_base_N")
    for l in layers:
        v = vfix[l]
        den = float((mu1[l] - mu0[l]) @ v)
        summary["own"][str(l)] = {c: [float((mean[f"own_{c}"][k, l] - mu0[l]) @ v) / den for k in range(nb + 1)]
                                  for c in ("base_nothink", "base_think", "rl_nothink")}
        al = []
        for k in range(nb + 1):
            vb = mean["fix_base_T"][k, l] - mean["fix_base_N"][k, l]
            db = mean["fix_rl_N"][k, l] - mean["fix_base_N"][k, l]
            al.append(float(db @ vb) / float(vb @ vb))
        summary["fixed_alpha"][str(l)] = al
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    torch.save({k: v.float() for k, v in mean.items()}, os.path.splitext(args.out)[0] + "_means.pt")
    mid = str(layers[1])
    print("bins:", summary["bins"])
    for c, xs in summary["own"][mid].items():
        print(f"own {c:13s}", " ".join(f"{x:+.2f}" for x in xs))
    print("fixed alpha RL ", " ".join(f"{x:+.2f}" for x in summary["fixed_alpha"][mid]))


if __name__ == "__main__":
    main()
