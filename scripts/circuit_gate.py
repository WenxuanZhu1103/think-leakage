
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render
from think_vector import load_rows

CTX = {"on": False}


def patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    k = repeat_kv(key, module.num_key_value_groups)
    v = repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : k.shape[-2]]
    if CTX["on"]:
        l = module.layer_idx
        q0, blk = CTX["q0"], CTX["blk"]
        heads = CTX["knock"].get(l)
        if heads is not None:
            qs = q0 - CTX.get("qoff", 1)
            sel = w[:, heads, qs:, :]
            sel[..., blk[0]:blk[1]] = torch.finfo(w.dtype).min
            w[:, heads, qs:, :] = sel
    p = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32)
    if CTX["on"] and CTX["rec"] is not None:
        q0, blk = CTX["q0"], CTX["blk"]
        CTX["rec"][module.layer_idx] = p[0, :, q0:, blk[0]:blk[1]].sum(-1).mean(-1).double()
    out = torch.matmul(p.to(query.dtype), v).transpose(1, 2).contiguous()
    return out, None


@torch.no_grad()
def run(model, prefix, y, knock=None, rec=False, blk=None):
    dev = next(model.parameters()).device
    ids = torch.tensor([prefix + y], device=dev)
    CTX.update(on=True, q0=len(prefix), blk=blk, knock=knock or {}, rec={} if rec else None)
    try:
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    finally:
        CTX["on"] = False
    p = len(prefix)
    hs = torch.stack([h[0, p:].double().sum(0) for h in out.hidden_states])
    lp = torch.log_softmax(out.logits[0, p - 1:p - 1 + len(y)].float(), -1)
    mass = torch.stack([CTX["rec"][l] for l in sorted(CTX["rec"])]) if rec else None
    return hs, lp, mass


def kl(lp_ref, lp):
    return float((lp_ref.exp() * (lp_ref - lp)).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gen_dir", required=True, help="dir with base_nothink.jsonl (the fixed text)")
    ap.add_argument("--traj", nargs="*", default=[], help="tag=adapter_dir")
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--max_resp", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="first N texts only (smoke)")
    ap.add_argument("--scan_layers", type=int, default=4, help="per-head scan within this many most effective layers")
    ap.add_argument("--scan_texts", type=int, default=30)
    ap.add_argument("--topk", default="1,2,4,8,16,32")
    ap.add_argument("--n_rand", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import transformers.models.qwen3.modeling_qwen3 as mq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mq.eager_attention_forward = patched_eager
    tok = AutoTokenizer.from_pretrained(args.base)
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    df = pd.read_parquet(args.data)
    S_T = [render(tok, p, None, enable_thinking=True) for p in df["prompt"]]
    S_N = [render(tok, p, None, enable_thinking=False) for p in df["prompt"]]
    rows = [r for r in load_rows(args.gen_dir, "base_nothink") if r["sid"] == 0]
    rows = sorted(rows, key=lambda r: r["pid"])[: args.limit or None]
    for r in rows:
        assert r["prompt"] == S_N[r["pid"]]
    blk_ids = E("<think>\n\n</think>\n\n")
    items = []
    for r in rows:
        pn = E(S_N[r["pid"]])
        assert pn[-len(blk_ids):] == blk_ids, "empty-think prefix does not end with the block"
        b0 = len(pn) - len(blk_ids)
        items.append({"pid": r["pid"], "PN": pn, "PT": E(S_T[r["pid"]]) + E("<think>\n"), "PD": pn[:b0],
                      "blk": (b0, len(pn)), "y": r["token_ids"][: args.max_resp]})
    print(f"{len(items)} texts, block {blk_ids}, mean resp {np.mean([len(it['y']) for it in items]):.0f} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    mid = L // 2
    ALL = {l: list(range(H)) for l in range(L)}

    traj = [t.split("=", 1) for t in args.traj]
    pm = None
    if traj:
        from peft import PeftModel
        for tag, path in traj:
            an = tag.replace(".", "_")
            if pm is None:
                pm = PeftModel.from_pretrained(model, path, adapter_name=an)
            else:
                pm.load_adapter(path, adapter_name=an)
        pm.eval()

    class Base:
        def __enter__(self):
            self.cm = pm.disable_adapter() if pm is not None else None
            if self.cm is not None:
                self.cm.__enter__()
            return pm if pm is not None else model

        def __exit__(self, *a):
            if self.cm is not None:
                self.cm.__exit__(*a)

    def with_adapter(tag):
        pm.set_adapter(tag.replace(".", "_"))
        return pm

    acc = {}

    def add(name, it, hs, lp, lpT, lpN_kl):
        a = acc.setdefault(name, {"hs": 0.0, "n": 0, "kl": 0.0, "kl_rec_txt": [], "mid_txt": {}})
        n = len(it["y"])
        a["hs"] = a["hs"] + hs
        a["n"] += n
        k = kl(lpT, lp)
        a["kl"] += k * n
        a["kl_rec_txt"].append(1 - k / lpN_kl)
        a["mid_txt"][it["pid"]] = (hs[mid] / n)

    def evaluate(model_ctx, it, lpT, klN, conds):
        for name, pk, kn in conds:
            hs, lp, _ = run(model_ctx, it[pk], it["y"], kn, False, it["blk"])
            add(name, it, hs, lp, lpT, klN)
            del lp

    mass = {}
    scan_rank = None
    for i, it in enumerate(items):
        with Base() as m:
            hsT, lpT, _ = run(m, it["PT"], it["y"], None, False, it["blk"])
            hsN, lpN, mN = run(m, it["PN"], it["y"], None, True, it["blk"])
            klN = kl(lpT, lpN)
            add("base_T", it, hsT, lpT, lpT, klN)
            add("base_N", it, hsN, lpN, lpT, klN)
            mass.setdefault("base_N", []).append(mN.cpu())
            del lpN
            conds = [("base_D", "PD", None), ("base_Kall", "PN", ALL)]
            conds += [(f"base_Klayer{l}", "PN", {l: list(range(H))}) for l in range(L)]
            evaluate(m, it, lpT, klN, conds)
        for tag, _ in traj:
            m = with_adapter(tag)
            hs, lp, mt = run(m, it["PN"], it["y"], None, True, it["blk"])
            add(f"{tag}_N", it, hs, lp, lpT, klN)
            mass.setdefault(f"{tag}_N", []).append(mt.cpu())
            del lp
            evaluate(m, it, lpT, klN, [(f"{tag}_Kall", "PN", ALL)])
        del lpT
        if i % 10 == 0:
            print(f"pass1 text {i + 1}/{len(items)}", flush=True)

    def summarise():
        vth = acc["base_T"]["hs"] / acc["base_T"]["n"] - acc["base_N"]["hs"] / acc["base_N"]["n"]
        mN = acc["base_N"]["hs"] / acc["base_N"]["n"]
        nv = (vth * vth).sum(-1)
        res = {}
        for name, a in acc.items():
            sh = a["hs"] / a["n"] - mN
            alpha = ((sh * vth).sum(-1) / nv).tolist()
            txt = {p: float(((a["mid_txt"][p] - acc["base_N"]["mid_txt"][p]) * vth[mid]).sum() / nv[mid]) for p in a["mid_txt"]}
            res[name] = {"alpha": alpha, "alpha_mid": alpha[mid], "kl": a["kl"] / a["n"],
                         "kl_rec": 1 - (a["kl"] / a["n"]) / (acc["base_N"]["kl"] / acc["base_N"]["n"]),
                         "kl_rec_txt_mean": float(np.mean(a["kl_rec_txt"])), "alpha_mid_txt": txt}
        return res

    res = summarise()
    layer_eff = sorted(range(L), key=lambda l: -res[f"base_Klayer{l}"]["kl_rec"])
    print("K_all kl_rec %.3f alpha_mid %.3f | D kl_rec %.3f alpha_mid %.3f" % (res["base_Kall"]["kl_rec"], res["base_Kall"]["alpha_mid"],
                                                                             res["base_D"]["kl_rec"], res["base_D"]["alpha_mid"]), flush=True)
    print("top layers by kl_rec:", [(l, round(res[f"base_Klayer{l}"]["kl_rec"], 3)) for l in layer_eff[:8]], flush=True)

    scan_layers = layer_eff[: args.scan_layers]
    sub = items[: args.scan_texts]
    head_eff = {}
    for i, it in enumerate(sub):
        with Base() as m:
            _, lpT, _ = run(m, it["PT"], it["y"], None, False, it["blk"])
            _, lpN, _ = run(m, it["PN"], it["y"], None, False, it["blk"])
            klN = kl(lpT, lpN)
            del lpN
            for l in scan_layers:
                for h in range(H):
                    _, lp, _ = run(m, it["PN"], it["y"], {l: [h]}, False, it["blk"])
                    head_eff.setdefault((l, h), []).append(1 - kl(lpT, lp) / klN)
                    del lp
        del lpT
        print(f"pass2 text {i + 1}/{len(sub)}", flush=True)
    scan_rank = sorted(head_eff, key=lambda k: -np.mean(head_eff[k]))

    rng = np.random.default_rng(args.seed)
    ks = [int(x) for x in args.topk.split(",")]
    sets = {}
    for k in ks:
        top = scan_rank[:k]
        sets[f"base_Ktop{k}"] = top
        for r in range(args.n_rand):
            pick = rng.choice(L * H, k, replace=False)
            sets[f"base_Krand{k}_{r}"] = [(int(x) // H, int(x) % H) for x in pick]

    def knock_of(hl):
        d = {}
        for l, h in hl:
            d.setdefault(l, []).append(h)
        return d

    for i, it in enumerate(items):
        with Base() as m:
            _, lpT, _ = run(m, it["PT"], it["y"], None, False, it["blk"])
            _, lpN, _ = run(m, it["PN"], it["y"], None, False, it["blk"])
            klN = kl(lpT, lpN)
            del lpN
            evaluate(m, it, lpT, klN, [(name, "PN", knock_of(hl)) for name, hl in sets.items()])
        if traj:
            m = with_adapter(traj[-1][0])
            evaluate(m, it, lpT, klN, [(f"{traj[-1][0]}_Ktop{k}", "PN", knock_of(sets[f"base_Ktop{k}"])) for k in ks])
        del lpT
        if i % 10 == 0:
            print(f"pass3 text {i + 1}/{len(items)}", flush=True)

    res = summarise()
    out = {"args": vars(args), "n_layers": L, "n_heads": H, "mid": mid, "n_texts": len(items), "cond": res,
           "layer_rank": layer_eff, "scan_layers": scan_layers,
           "head_eff": {f"{l}.{h}": [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))] for (l, h), v in head_eff.items()},
           "head_rank": [f"{l}.{h}" for l, h in scan_rank], "knock_sets": {k: [f"{l}.{h}" for l, h in v] for k, v in sets.items()},
           "mass": {k: torch.stack(v).mean(0).tolist() for k, v in mass.items()},
           "mass_txt_top": {k: torch.stack(v)[:, [l for l, _ in scan_rank[:8]], [h for _, h in scan_rank[:8]]].tolist() for k, v in mass.items()}}
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)
    for name in ["base_D", "base_Kall"] + [f"base_Ktop{k}" for k in ks] + [f"base_Krand{k}_0" for k in ks]:
        print(f"{name:16s} kl_rec {res[name]['kl_rec']:+.3f} alpha_mid {res[name]['alpha_mid']:+.3f}")
    for tag, _ in traj:
        print(f"{tag:8s} N kl_rec {res[tag + '_N']['kl_rec']:+.3f} alpha {res[tag + '_N']['alpha_mid']:+.3f} | Kall kl_rec "
              f"{res[tag + '_Kall']['kl_rec']:+.3f} alpha {res[tag + '_Kall']['alpha_mid']:+.3f}")


if __name__ == "__main__":
    main()
