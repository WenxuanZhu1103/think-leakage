
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", default="data/dapo17k/train.parquet")
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--resp_tok", type=int, default=512)
    ap.add_argument("--think_template", default=None)
    ap.add_argument("--nothink_template", default=None)
    ap.add_argument("--template_kwarg", action="store_true")
    ap.add_argument("--think_opener", default="")
    ap.add_argument("--ctrl_system", default="Respond in a formal, polite tone.")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    opener = args.think_opener.encode().decode("unicode_escape")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    df = pd.read_parquet(args.data).sample(n=args.n_prompts, random_state=args.seed)
    t_think = open(args.think_template).read() if args.think_template else None
    t_noth = open(args.nothink_template).read() if args.nothink_template else None
    kw_t = {"enable_thinking": True} if args.template_kwarg else {}
    kw_n = {"enable_thinking": False} if args.template_kwarg else {}
    S_N = [render(tok, p, t_noth, **kw_n) for p in df["prompt"]]
    P = {"T": [E(render(tok, p, t_think, **kw_t)) + (E(opener) if opener else []) for p in df["prompt"]],
         "N": [E(s) for s in S_N],
         "C": [E(render(tok, [{"role": "system", "content": args.ctrl_system}] + [m for m in p if m["role"] != "system"], t_noth, **kw_n))
               for p in df["prompt"]]}
    assert P["T"][0] != P["N"][0] != P["C"][0]

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = next(model.parameters()).device
    L1, d = model.config.num_hidden_layers + 1, model.config.hidden_size

    tok.padding_side = "left"
    torch.manual_seed(args.seed)
    Y = []
    with torch.no_grad():
        for b in range(0, len(S_N), 32):
            enc = tok(S_N[b:b + 32], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.resp_tok,
                                 pad_token_id=tok.pad_token_id)
            for g in out[:, enc["input_ids"].shape[1]:].tolist():
                while g and g[-1] == tok.pad_token_id:
                    g.pop()
                Y.append(g)
    keep = [i for i, y in enumerate(Y) if len(y) >= 16]
    print(f"responses: {len(keep)}/{len(Y)} kept, mean len {sum(len(Y[i]) for i in keep) / len(keep):.0f}", flush=True)

    G = {k: torch.zeros(L1, d, d, device=dev, dtype=torch.float32) for k in ("think", "ctrl", "pca")}
    s_n = torch.zeros(L1, d, device=dev, dtype=torch.float64)
    s_v = torch.zeros(L1, d, device=dev, dtype=torch.float64)
    s_c = torch.zeros(L1, d, device=dev, dtype=torch.float64)
    s_nu = torch.zeros(L1, d, device=dev, dtype=torch.float64)
    n_tok = 0
    unit = lambda x: torch.nn.functional.normalize(x, dim=-1)

    @torch.no_grad()
    def resp_states(prefixes, idx):
        T = max(len(prefixes[i]) + len(Y[i]) for i in idx)
        ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            n = len(prefixes[i]) + len(Y[i])
            ids[r, :n] = torch.tensor(prefixes[i] + Y[i])
            att[r, :n] = 1
        hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        return [torch.cat([h[r, len(prefixes[i]) + args.skip:len(prefixes[i]) + len(Y[i])] for r, i in enumerate(idx)]).float() for h in hs]

    for b in range(0, len(keep), args.bs):
        idx = keep[b:b + args.bs]
        hN, hT, hC = resp_states(P["N"], idx), resp_states(P["T"], idx), resp_states(P["C"], idx)
        for l in range(L1):
            dt, dc = hT[l] - hN[l], hC[l] - hN[l]
            s_v[l] += dt.double().sum(0)
            s_c[l] += dc.double().sum(0)
            dt, dc, hn = unit(dt), unit(dc), unit(hN[l])
            G["think"][l] += dt.T @ dt
            G["ctrl"][l] += dc.T @ dc
            G["pca"][l] += hn.T @ hn
            s_nu[l] += hn.double().sum(0)
        n_tok += hN[0].shape[0]
    mu = (s_nu / n_tok).float()
    means = {"think": (s_v / n_tok).float(), "ctrl": (s_c / n_tok).float()}
    out = {"n_tok": n_tok, "k": args.k, "v_mean": means["think"].cpu(), "c_mean": means["ctrl"].cpu(), "Q": {}, "share": {}}
    for name in G:
        Qs, sh = [], []
        for l in range(L1):
            M = G[name][l] / n_tok
            lead = None
            if name == "pca":
                M = M - torch.outer(mu[l], mu[l])
            elif means[name][l].norm() > 1e-8:
                lead = torch.nn.functional.normalize(means[name][l], dim=0)
                Pm = torch.eye(d, device=dev) - torch.outer(lead, lead)
                M = Pm @ M @ Pm
            w, U = torch.linalg.eigh(M.double())
            w, U = w.flip(0).clamp(min=0), U.flip(1)
            Qk = U[:, :args.k].float() if lead is None else torch.linalg.qr(torch.cat([lead[:, None], U[:, :args.k - 1].float()], 1))[0]
            Qs.append(Qk.cpu())
            sh.append((w[:args.k] / w.sum().clamp(min=1e-30)).float().cpu())
        out["Q"][name], out["share"][name] = torch.stack(Qs), torch.stack(sh)
    torch.save(out, args.out)
    mid = L1 // 2
    vm = torch.nn.functional.normalize(out["v_mean"][mid], dim=0)
    print(f"n_tok={n_tok}; layer {mid}: think top-1/4/8 share = {out['share']['think'][mid][:1].sum():.3f}/"
          f"{out['share']['think'][mid][:4].sum():.3f}/{out['share']['think'][mid][:8].sum():.3f}; "
          f"|cos(Q[:,0], mean v)| = {abs(float(out['Q']['think'][mid][:, 0] @ vm)):.3f}; "
          f"think-vs-ctrl subspace overlap (k=8) = {float((out['Q']['think'][mid][:, :8].T @ out['Q']['ctrl'][mid][:, :8]).pow(2).sum() / 8):.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
