
import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render, text_stats

EARLY = 256
BINS = {"bnd": [0], "early": [1], "late": [2], "resp": [1, 2]}


def load_rows(gen_dir, tag):
    rows = [json.loads(x) for x in open(os.path.join(gen_dir, f"{tag}.jsonl"))]
    keys = [(r["pid"], r["sid"]) for r in rows]
    assert len(set(keys)) == len(keys), f"{tag}: duplicate (pid, sid)"
    assert all(len(r["token_ids"]) > 0 for r in rows), f"{tag}: empty response"
    return rows


def split_opener(ids, openers):
    for op in openers:
        if op and ids[:len(op)] == op:
            return op, ids[len(op):]
    return [], ids


@torch.no_grad()
def acts(model, pad_id, items, max_resp, bs, want_sq=False):
    dev = next(model.parameters()).device
    enc = [(p, y[:max_resp]) for _, p, y in items]
    assert all(len(p) > 0 and len(y) > 0 for p, y in enc)
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]))
    out_s, out_c, sq = [None] * len(enc), [None] * len(enc), None
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        T = max(len(enc[i][0]) + len(enc[i][1]) for i in idx)
        ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        msk = torch.zeros((len(idx), 3, T), dtype=torch.bool)
        for r, i in enumerate(idx):
            p, y = enc[i]
            n = len(p) + len(y)
            ids[r, :n] = torch.tensor(p + y)
            att[r, :n] = 1
            msk[r, 0, len(p) - 1] = True
            msk[r, 1, len(p):min(n, len(p) + EARLY)] = True
            msk[r, 2, len(p) + EARLY:n] = True
        hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        m = msk.to(dev).float()
        per = torch.stack([torch.einsum("bkt,btd->bkd", m, h.float()) for h in hs], 2)
        if want_sq:
            mr = (msk[:, 1] | msk[:, 2]).to(dev)[:, :, None]
            s2 = torch.stack([((h.double() ** 2) * mr).sum((0, 1)) for h in hs], 0)
            sq = s2 if sq is None else sq + s2
        for r, i in enumerate(idx):
            out_s[i] = per[r].cpu()
            out_c[i] = msk[r].sum(-1).float()
    cond = {"sums": torch.stack(out_s), "cnts": torch.stack(out_c), "pid": torch.tensor([it[0] for it in items])}
    if want_sq:
        cond["sq"] = sq.cpu()
    return cond


def sub(cond, keep):
    keep = torch.as_tensor(keep)
    return {"sums": cond["sums"][keep], "cnts": cond["cnts"][keep], "pid": cond["pid"][keep]}


def mean_of(cond, bins, pid_w=None, textw=False, layer=None):
    s = (cond["sums"][:, bins] if layer is None else cond["sums"][:, bins, layer]).sum(1).double()
    c = cond["cnts"][:, bins].sum(1).double()
    w = torch.ones_like(c) if pid_w is None else pid_w[cond["pid"]].double()
    w = w * (c > 0)
    shape = [-1] + [1] * (s.dim() - 1)
    if textw:
        return ((s / c.clamp(min=1).view(shape)) * w.view(shape)).sum(0) / w.sum().clamp(min=1e-9)
    return (s * w.view(shape)).sum(0) / (w * c).sum().clamp(min=1e-9)


def cosl(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def nan_where(x, bad):
    x = x.clone()
    x[bad] = float("nan")
    return x.tolist()


def metrics(d, v, std, ctrls, alphas):
    vn = v.norm(dim=-1)
    bad = vn < 1e-9
    vn2 = (vn ** 2).clamp(min=1e-30)
    out = {"cos": nan_where(cosl(d, v), bad), "cos_std": nan_where(cosl(d / std, v / std), bad),
           "alpha": nan_where((d * v).sum(-1) / vn2, bad), "norm_ratio": nan_where(d.norm(dim=-1) / vn2.sqrt(), bad)}
    for i, c in enumerate(ctrls):
        out[f"cos_ctrl{i}"] = nan_where(cosl(d, c), c.norm(dim=-1) < 1e-9)
    for a in alphas:
        out[f"dist_a{a:g}"] = nan_where((d - a * v).norm(dim=-1) / vn2.sqrt(), bad)
    return out


def estimate(K, Zd, T, Zv, C, std, alphas, n_pid, blayers, n_boot, gen):
    even = (torch.arange(n_pid) % 2 == 0).double()
    odd = 1.0 - even
    out = {}
    for bname, bins in BINS.items():
        for wname, tw in (("tok", False), ("text", True)):
            if bname == "bnd" and tw:
                continue
            m = lambda cond, w=None: mean_of(cond, bins, w, tw)
            d, v = m(K) - m(Zd), m(T) - m(Zv)
            r = metrics(d, v, std, [m(c) - m(Zv) for c in C], alphas)
            xa = metrics(m(K, odd) - m(Zd, odd), m(T, even) - m(Zv, even), std, [], [])
            xb = metrics(m(K, even) - m(Zd, even), m(T, odd) - m(Zv, odd), std, [], [])
            for k in ("cos", "alpha"):
                r[f"{k}_xfit"] = [(a + b) / 2 for a, b in zip(xa[k], xb[k])]
            out[f"{bname}_{wname}"] = r
    boot = {}
    for bname in ("resp", "bnd"):
        bins = BINS[bname]
        for l in blayers:
            cs, al = [], []
            for _ in range(n_boot):
                w = torch.bincount(torch.randint(0, n_pid, (n_pid,), generator=gen), minlength=n_pid).double()
                d = mean_of(K, bins, w, layer=l) - mean_of(Zd, bins, w, layer=l)
                v = mean_of(T, bins, w, layer=l) - mean_of(Zv, bins, w, layer=l)
                cs.append(float(cosl(d, v)))
                al.append(float((d * v).sum() / (v * v).sum().clamp(min=1e-30)))
            q = lambda x: [float(torch.tensor(x).quantile(p)) for p in (0.025, 0.5, 0.975)]
            boot[f"{bname}_L{l}"] = {"cos": q(cs), "alpha": q(al)}
    out["boot"] = boot
    return out


@torch.no_grad()
def llr(model, pad_id, triples, max_resp, bs, n_boot, gen):
    dev = next(model.parameters()).device

    def lp(pairs):
        tot = []
        for b in range(0, len(pairs), bs):
            enc = [(p, y[:max_resp]) for p, y in pairs[b:b + bs]]
            T = max(len(p) + len(y) for p, y in enc)
            ids = torch.full((len(enc), T), pad_id, dtype=torch.long)
            att = torch.zeros((len(enc), T), dtype=torch.long)
            for r, (p, y) in enumerate(enc):
                ids[r, :len(p) + len(y)] = torch.tensor(p + y)
                att[r, :len(p) + len(y)] = 1
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).logits
            for r, (p, y) in enumerate(enc):
                lg = logits[r, len(p) - 1:len(p) + len(y) - 1].float().log_softmax(-1)
                tot.append(float(lg.gather(-1, torch.tensor(y, device=dev)[:, None]).sum()))
        return torch.tensor(tot, dtype=torch.float64)

    n = torch.tensor([min(len(y), max_resp) for _, _, y in triples], dtype=torch.float64)
    a = lp([(t, y) for t, _, y in triples])
    b = lp([(q, y) for _, q, y in triples])
    diff = a - b
    bs_ = []
    for _ in range(n_boot):
        i = torch.randint(0, len(n), (len(n),), generator=gen)
        bs_.append(float(diff[i].sum() / n[i].sum()))
    t = torch.tensor(bs_)
    return {"llr_per_tok": float(diff.sum() / n.sum()), "ci95": [float(t.quantile(0.025)), float(t.quantile(0.975))],
            "lp_think": float(a.sum() / n.sum()), "lp_nothink": float(b.sum() / n.sum()),
            "frac_texts_pos": float((diff > 0).double().mean()), "n_tok": int(n.sum())}


def find_layers(model):
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == model.config.num_hidden_layers:
            return mod
    raise RuntimeError("decoder layer list not found")


@torch.no_grad()
def steer_generate(model, tok, prompts, v, alpha, l0, max_new, bs, eos_ids, seed):
    layers = find_layers(model)
    assert 1 <= l0 < len(layers), "hidden_states[L] is post-final-norm: not a block output, cannot be injected here"
    dev = next(model.parameters()).device
    vec = (alpha * v[l0]).to(dev, torch.bfloat16)

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] != 1:
            return None
        h = h + vec
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handle = layers[l0 - 1].register_forward_hook(hook) if alpha != 0 else None
    side = tok.padding_side
    texts, ntok, stopped = [], [], []
    try:
        tok.padding_side = "left"
        torch.manual_seed(seed)
        for b in range(0, len(prompts), bs):
            enc = tok(prompts[b:b + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new,
                                 use_cache=True, eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
            for g in out[:, enc["input_ids"].shape[1]:].tolist():
                cut = next((i for i, t in enumerate(g) if t in eos_ids), None)
                keep = g if cut is None else g[:cut + 1]
                texts.append(tok.decode(keep, skip_special_tokens=False))
                ntok.append(len(keep))
                stopped.append(cut is not None)
    finally:
        tok.padding_side = side
        if handle is not None:
            handle.remove()
    return texts, ntok, stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--traj", nargs="*", default=[], help="tag=adapter_dir; <gen_dir>/<tag>.jsonl must exist")
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--think_template", default=None)
    ap.add_argument("--nothink_template", default=None)
    ap.add_argument("--template_kwarg", action="store_true")
    ap.add_argument("--think_opener", default="", help="text the MODEL emits to open thinking when the template does not "
                    "(Qwen3: '<think>\\n'; OLMo-3-Think: '' because its template already ends with <think>)")
    ap.add_argument("--ctrl_system", nargs="*", default=["Respond in a formal, polite tone.", "You are a helpful assistant."])
    ap.add_argument("--max_resp_tok", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--steer", nargs="*", default=[], help="single:<l> with 1 <= l < L")
    ap.add_argument("--steer_max_new", type=int, default=4096)
    ap.add_argument("--steer_bs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (must match think_leak_gen.py)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    opener = args.think_opener.encode().decode("unicode_escape")

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    df = pd.read_parquet(args.data)
    df = df.head(args.limit) if args.limit else df
    n_pid = len(df)
    t_think = open(args.think_template).read() if args.think_template else None
    t_noth = open(args.nothink_template).read() if args.nothink_template else None
    kw_t = {"enable_thinking": True} if args.template_kwarg else {}
    kw_n = {"enable_thinking": False} if args.template_kwarg else {}
    S_T = [render(tok, p, t_think, **kw_t) for p in df["prompt"]]
    S_N = [render(tok, p, t_noth, **kw_n) for p in df["prompt"]]
    op_ids = E(opener) if opener else []
    openers = [op_ids, E("<think>\n"), E("<think>")]
    openers = sorted([o for o in openers if o], key=len, reverse=True)
    P_T = [E(s) + op_ids for s in S_T]
    P_N = [E(s) for s in S_N]
    P_C = [[E(render(tok, [{"role": "system", "content": c}] + [m for m in p if m["role"] != "system"], t_noth, **kw_n))
            for p in df["prompt"]] for c in args.ctrl_system]

    rows_t, rows_n = load_rows(args.gen_dir, "base_think"), load_rows(args.gen_dir, "base_nothink")
    for r in rows_t:
        assert 0 <= r["pid"] < n_pid and r["prompt"] == S_T[r["pid"]], "base_think prompts do not match --think_template"
    for r in rows_n:
        assert 0 <= r["pid"] < n_pid and r["prompt"] == S_N[r["pid"]], "base_nothink prompts do not match --nothink_template"
    emitted = sum(bool(split_opener(r["token_ids"], openers)[0]) for r in rows_t) / len(rows_t)
    print(f"base_think samples that start by emitting an opener: {emitted:.2f}; --think_opener={opener!r}", flush=True)
    assert (emitted > 0.5) == bool(op_ids), "set --think_opener iff the model (not the template) opens the think block"
    for p in range(n_pid):
        sids = {r["sid"] for r in rows_n if r["pid"] == p}
        assert any(s % 2 == 0 for s in sids) and any(s % 2 == 1 for s in sids), f"pid {p}: need even and odd base_nothink samples"
        assert any(r["pid"] == p for r in rows_t), f"pid {p}: no base_think sample"

    Fn = [(r["pid"], r["token_ids"]) for r in rows_n if r["sid"] == 0]
    Ft = [(r["pid"], split_opener(r["token_ids"], openers)[1]) for r in rows_t if r["sid"] == 0]

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    L = model.config.num_hidden_layers
    L1 = L + 1
    blayers = sorted({round(L * q) for q in (0.25, 0.5, 0.75)})
    gen = torch.Generator().manual_seed(0)
    A, M, B = args.alphas, args.max_resp_tok, args.bs

    base = {}
    for name, F in (("Fn", Fn), ("Ft", Ft)):
        base[name] = {"think": acts(model, pad_id, [(p, P_T[p], y) for p, y in F], M, B),
                      "noth": acts(model, pad_id, [(p, P_N[p], y) for p, y in F], M, B),
                      "ctrl": [acts(model, pad_id, [(p, PC[p], y) for p, y in F], M, B) for PC in P_C]}

    def own_items(rows, think):
        items = []
        for r in rows:
            if think:
                op, rest = split_opener(r["token_ids"], openers)
                items.append((r["pid"], E(r["prompt"]) + op, rest))
            else:
                items.append((r["pid"], E(r["prompt"]), r["token_ids"]))
        return [it for it in items if len(it[2]) > 0]

    own_t = acts(model, pad_id, own_items(rows_t, True), M, B)
    own_n = acts(model, pad_id, own_items(rows_n, False), M, B, want_sq=True)
    sid_n = torch.tensor([r["sid"] for r in rows_n])
    own_n_ev, own_n_od = sub(own_n, (sid_n % 2 == 0).nonzero()[:, 0]), sub(own_n, (sid_n % 2 == 1).nonzero()[:, 0])
    R = BINS["resp"]
    ntok_n = own_n["cnts"][:, R].sum().double()
    mu_n = mean_of(own_n, R)
    std = (own_n["sq"] / ntok_n - mu_n ** 2).clamp(min=0).sqrt()
    std = torch.maximum(std, 0.05 * std.median(dim=-1, keepdim=True).values)

    v_fn = mean_of(base["Fn"]["think"], R) - mean_of(base["Fn"]["noth"], R)
    v_ft = mean_of(base["Ft"]["think"], R) - mean_of(base["Ft"]["noth"], R)
    v_own = mean_of(own_t, R) - mean_of(own_n, R)
    B0 = BINS["bnd"]
    v_bnd = mean_of(base["Fn"]["think"], B0) - mean_of(base["Fn"]["noth"], B0)
    even = (torch.arange(n_pid) % 2 == 0).double()
    res = {"n_layers_plus1": L1, "boot_layers": blayers, "alphas": A, "bins": list(BINS), "ctrl_system": args.ctrl_system,
           "direction": {
               "cos_v_Fn_Ft": cosl(v_fn, v_ft).tolist(), "cos_v_Fn_own": cosl(v_fn, v_own).tolist(),
               "cos_v_resp_bnd": cosl(v_fn, v_bnd).tolist(),
               "cos_v_ctrl_Fn": [cosl(v_fn, mean_of(c, R) - mean_of(base["Fn"]["noth"], R)).tolist() for c in base["Fn"]["ctrl"]],
               "norm_v_Fn": v_fn.norm(dim=-1).tolist(), "norm_v_own": v_own.norm(dim=-1).tolist(), "norm_v_bnd": v_bnd.norm(dim=-1).tolist(),
               "act_norm_rms": (own_n["sq"].sum(-1) / ntok_n).sqrt().tolist(),
               "own_split_half_cos": cosl(mean_of(own_t, R, even) - mean_of(own_n, R, even),
                                          mean_of(own_t, R, 1 - even) - mean_of(own_n, R, 1 - even)).tolist()},
           "ckpt": {}, "llr": {}}
    torch.save({"v_Fn": v_fn, "v_Ft": v_ft, "v_own": v_own, "v_bnd": v_bnd, "mu_nothink": mu_n, "std": std},
               os.path.splitext(args.out)[0] + "_vectors.pt")

    def trip(rows):
        first = {}
        for r in sorted(rows, key=lambda r: r["sid"]):
            first.setdefault(r["pid"], r)
        out = [(P_T[p], P_N[p], split_opener(r["token_ids"], openers)[1]) for p, r in sorted(first.items())]
        return [t for t in out if len(t[2]) > 0]

    traj = [s.split("=", 1) for s in args.traj]
    assert len({t for t, _ in traj}) == len(traj), "duplicate tags"
    for tag, rows in [("base_think", rows_t), ("base_nothink", rows_n)] + [(t, load_rows(args.gen_dir, t)) for t, _ in traj]:
        res["llr"][tag] = llr(model, pad_id, trip(rows), M, max(1, B // 2), args.n_boot, gen)
        print("llr", tag, res["llr"][tag], flush=True)
    json.dump(res, open(args.out, "w"))

    if args.steer:
        from verl.utils.reward_score import default_compute_score
        gc = GenerationConfig.from_pretrained(args.base)
        eos_ids = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
        eos_ids = sorted({e for e in eos_ids + [tok.eos_token_id] if e is not None})
        res["steer"] = {}
        for mode in args.steer:
            l0 = int(mode.split(":")[1])
            for a in A:
                texts, ntok, stopped = steer_generate(model, tok, S_N, v_own, a, l0, args.steer_max_new, args.steer_bs, eos_ids, seed=1234)
                st = [text_stats(t, n) for t, n in zip(texts, ntok)]
                corr = 0
                for j, t in enumerate(texts):
                    try:
                        r = default_compute_score(df.iloc[j].get("data_source", "math_dapo"), t, df.iloc[j]["reward_model"]["ground_truth"])
                        corr += int((r.get("score", 0) if isinstance(r, dict) else r) > 0.5)
                    except Exception:
                        pass
                tt = sum(ntok)
                key = f"L{l0}_a{a:g}"
                res["steer"][key] = {"mean_len": tt / len(texts), "acc": corr / len(texts), "stop_frac": sum(stopped) / len(texts),
                                     "markers_per_1k": 1000.0 * sum(s["n_markers"] for s in st) / max(1, tt),
                                     "frac_with_marker": sum(s["n_markers"] > 0 for s in st) / len(st),
                                     "frac_think_open": sum(s["has_think_open"] for s in st) / len(st),
                                     "frac_think_close": sum(s["has_think_close"] for s in st) / len(st)}
                print("steer", key, res["steer"][key], flush=True)
                with open(os.path.splitext(args.out)[0] + f"_steer_{key}.jsonl", "w") as f:
                    for j, t in enumerate(texts):
                        f.write(json.dumps({"pid": j, "text": t, "n_tok": ntok[j], "stopped": stopped[j]}, ensure_ascii=False) + "\n")
                json.dump(res, open(args.out, "w"))

    from peft import PeftModel
    pm, prev = None, None
    for tag, path in traj:
        an = tag.replace(".", "_")
        if pm is None:
            pm = PeftModel.from_pretrained(model, path, adapter_name=an)
        else:
            pm.load_adapter(path, adapter_name=an)
        pm.set_adapter(an)
        if prev is not None:
            pm.delete_adapter(prev)
        prev = an
        pm.eval()
        assert list(pm.peft_config) == [an] and pm.active_adapter == an
        out = {}
        for name, F in (("Fn", Fn), ("Ft", Ft)):
            K = acts(pm, pad_id, [(p, P_N[p], y) for p, y in F], M, B)
            b = base[name]
            out[name] = estimate(K, b["noth"], b["think"], b["noth"], b["ctrl"], std, A, n_pid, blayers, args.n_boot, gen)
        K = acts(pm, pad_id, own_items(load_rows(args.gen_dir, tag), False), M, B)
        out["own"] = estimate(K, own_n_od, own_t, own_n_ev, [], std, A, n_pid, blayers, args.n_boot, gen)
        res["ckpt"][tag] = out
        mid = blayers[1]
        f, o = out["Fn"], out["own"]
        print(f"{tag} L{mid} | Fn resp cos={f['resp_tok']['cos'][mid]:+.3f} xfit={f['resp_tok']['cos_xfit'][mid]:+.3f} "
              f"a={f['resp_tok']['alpha'][mid]:+.3f} ctrl={f['resp_tok']['cos_ctrl0'][mid]:+.3f} | bnd cos={f['bnd_tok']['cos'][mid]:+.3f} "
              f"a={f['bnd_tok']['alpha'][mid]:+.3f} | own cos={o['resp_tok']['cos'][mid]:+.3f} xfit={o['resp_tok']['cos_xfit'][mid]:+.3f} "
              f"a={o['resp_tok']['alpha'][mid]:+.3f}", flush=True)
        json.dump(res, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
