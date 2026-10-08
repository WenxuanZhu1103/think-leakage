
import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render
from think_vector import find_layers

WIN = {"early": (4, 256), "late": (256, 1024), "resp": (4, 1024)}


def batches(items, max_tok):
    order = sorted(range(len(items)), key=lambda i: items[i][2])
    cur, tot = [], 0
    for i in order:
        n = items[i][2]
        if cur and tot + n > max_tok:
            yield cur
            cur, tot = [], 0
        cur.append(i)
        tot += n
    if cur:
        yield cur


@torch.no_grad()
def run_pass(model, pad, items, layer_mod, vhat, max_tok, want_proj):
    dev = next(model.parameters()).device
    cap = {}
    h = layer_mod.register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
    out = [None] * len(items)
    try:
        for idx in batches(items, max_tok):
            T = max(items[i][2] for i in idx)
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            for r, i in enumerate(idx):
                p, y, n = items[i]
                ids[r, :n] = torch.tensor(p + y)
                att[r, :n] = 1
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).logits
            H = cap["h"]
            for r, i in enumerate(idx):
                p, y, n = items[i]
                lp = 0.0
                for c0 in range(len(p) - 1, n - 1, 1024):
                    c1 = min(c0 + 1024, n - 1)
                    lg = logits[r, c0:c1].float().log_softmax(-1)
                    tgt = ids[r, c0 + 1:c1 + 1].to(dev)
                    lp += float(lg.gather(-1, tgt[:, None]).sum())
                d = {"lp": lp}
                if want_proj:
                    hr = H[r].float()
                    pr = hr @ vhat
                    L0 = len(p)
                    d["bnd"] = float(pr[L0 - 1])
                    for k, (a, b) in WIN.items():
                        seg = pr[L0 + a:min(L0 + b, n)]
                        d[k] = float(seg.mean()) if seg.numel() else float("nan")
                    seg = pr[L0 + 4:min(L0 + 1024, max(L0 + 4, n - 64))]
                    d["noans"] = float(seg.mean()) if seg.numel() else float("nan")
                out[i] = d
            del logits, H
            cap.clear()
    finally:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rollouts_dir", required=True)
    ap.add_argument("--vectors", required=True, help="think_vector_vectors.pt (v_Fn)")
    ap.add_argument("--layer", type=int, default=None, help="hidden_states index; default L/2")
    ap.add_argument("--traj", nargs="+", required=True, help="tag=adapter_dir in training order; 'base=none' first")
    ap.add_argument("--think_opener", default="<think>\\n")
    ap.add_argument("--max_tok", type=int, default=20000, help="tokens per forward batch")
    ap.add_argument("--limit_rows", type=int, default=0, help="smoke tests: first N rollouts per file")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    opener = args.think_opener.encode().decode("unicode_escape")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    P = pd.read_json(os.path.join(args.rollouts_dir, "prompts.jsonl"), lines=True)
    P_N = [E(render(tok, p, None, enable_thinking=False)) for p in P["prompt"]]
    P_T = [E(render(tok, p, None, enable_thinking=True)) + E(opener) for p in P["prompt"]]

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    L = args.layer or model.config.num_hidden_layers // 2
    layers = find_layers(model)
    layer_mod = layers[L - 1]
    vec = torch.load(args.vectors)
    v = vec["v_Fn"][L].float()
    vhat = (v / v.norm()).to(next(model.parameters()).device)
    traj = [s.split("=", 1) for s in args.traj]
    pm, loaded = None, []

    def use(tag, path):
        nonlocal pm
        if tag not in loaded:
            if pm is None:
                pm = PeftModel.from_pretrained(model, path, adapter_name=tag)
            else:
                pm.load_adapter(path, adapter_name=tag)
            loaded.append(tag)
        pm.set_adapter(tag)
        pm.eval()
        return pm

    for ti, (tag, path) in enumerate(traj):
        fn = os.path.join(args.out_dir, f"{tag}.jsonl")
        if os.path.exists(fn):
            continue
        rows = [json.loads(x) for x in open(os.path.join(args.rollouts_dir, f"{tag}.jsonl"))]
        rows = rows[:args.limit_rows] if args.limit_rows else rows
        it_n = [(P_N[r["pid"]], r["token_ids"], len(P_N[r["pid"]]) + len(r["token_ids"])) for r in rows]
        it_t = [(P_T[r["pid"]], r["token_ids"], len(P_T[r["pid"]]) + len(r["token_ids"])) for r in rows]
        base_model = pm.base_model.model if pm is not None else model
        if pm is not None:
            with pm.disable_adapter():
                b_nt = run_pass(pm, pad, it_n, layer_mod, vhat, args.max_tok, True)
                b_t = run_pass(pm, pad, it_t, layer_mod, vhat, args.max_tok, False)
        else:
            b_nt = run_pass(base_model, pad, it_n, layer_mod, vhat, args.max_tok, True)
            b_t = run_pass(base_model, pad, it_t, layer_mod, vhat, args.max_tok, False)
        k_self = k_next = None
        if tag != "base":
            k_self = run_pass(use(tag, path), pad, it_n, layer_mod, vhat, args.max_tok, True)
        if ti + 1 < len(traj):
            nt, npth = traj[ti + 1]
            k_next = run_pass(use(nt, npth), pad, it_n, layer_mod, vhat, args.max_tok, True)
        for old in [t for t in loaded if t not in (tag, traj[ti + 1][0] if ti + 1 < len(traj) else None)]:
            pm.delete_adapter(old)
            loaded.remove(old)
        with open(fn + ".tmp", "w") as f:
            for i, r in enumerate(rows):
                d = {k: r[k] for k in ("pid", "sid", "r_train", "r_boxed", "n_tok", "stopped", "n_markers")}
                d.update({"lp_nt": b_nt[i]["lp"], "lp_t": b_t[i]["lp"], "llr": (b_t[i]["lp"] - b_nt[i]["lp"]) / max(1, r["n_tok"]),
                          "pb_bnd": b_nt[i]["bnd"], "pb_early": b_nt[i]["early"], "pb_late": b_nt[i]["late"], "pb_resp": b_nt[i]["resp"], "pb_noans": b_nt[i]["noans"]})
                if k_self:
                    d.update({"lp_self": k_self[i]["lp"], "pk_resp": k_self[i]["resp"], "pk_bnd": k_self[i]["bnd"]})
                else:
                    d.update({"lp_self": b_nt[i]["lp"], "pk_resp": b_nt[i]["resp"], "pk_bnd": b_nt[i]["bnd"]})
                if k_next:
                    d.update({"lp_next": k_next[i]["lp"], "pn_resp": k_next[i]["resp"], "pn_bnd": k_next[i]["bnd"], "next_tag": traj[ti + 1][0]})
                f.write(json.dumps(d) + "\n")
        os.replace(fn + ".tmp", fn)
        import numpy as np
        llr = np.mean([(b_t[i]["lp"] - b_nt[i]["lp"]) / max(1, rows[i]["n_tok"]) for i in range(len(rows))])
        pb = np.nanmean([b_nt[i]["resp"] for i in range(len(rows))])
        msg = f"{tag}: n={len(rows)} llr={llr:+.4f} pb_resp={pb:+.3f}"
        if k_self and k_next:
            u = np.array([k_next[i]["lp"] - k_self[i]["lp"] for i in range(len(rows))])
            s = np.array([b_nt[i]["resp"] for i in range(len(rows))])
            msg += f"  Cov(pb_resp, u)={np.cov(s, u)[0, 1]:+.4f}  mean u={u.mean():+.2f}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
