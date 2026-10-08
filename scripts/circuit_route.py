
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


def kl(lp_ref, lp):
    return float((lp_ref.exp() * (lp_ref - lp)).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--max_resp", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--head_layers", type=int, default=4)
    ap.add_argument("--refs_only", action="store_true", help="references only (base think / base no-think / RL on all texts); "
                    "writes alpha of the RL policy on all texts and on the head-scan texts (rl_sub)")
    ap.add_argument("--scan_layers", default="", help="comma list: heads-only mode -- skip the LoRA and component passes, take the "
                    "references on all texts and scan every head of these layers (for splitting a full head scan across jobs)")
    ap.add_argument("--head_texts", type=int, default=30)
    ap.add_argument("--head_start", type=int, default=0, help="first text of the head pass; it covers head_texts texts from here "
                    "(e.g. 30 with --head_texts 60: the texts that were not used to rank the heads)")
    ap.add_argument("--heads", default="", help="heads-only mode restricted to these heads: 'layer.head' (0-indexed) joined by '+'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    E = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    df = pd.read_parquet(args.data)
    S_T = [render(tok, p, None, enable_thinking=True) for p in df["prompt"]]
    S_N = [render(tok, p, None, enable_thinking=False) for p in df["prompt"]]
    rows = sorted([r for r in load_rows(args.gen_dir, "base_nothink") if r["sid"] == 0], key=lambda r: r["pid"])[: args.limit or None]
    items = [{"pid": r["pid"], "PN": E(S_N[r["pid"]]), "PT": E(S_T[r["pid"]]) + E("<think>\n"), "y": r["token_ids"][: args.max_resp]} for r in rows]

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    pm = PeftModel.from_pretrained(model, args.adapter, adapter_name="rl")
    pm.eval()
    dev = next(pm.parameters()).device
    layers = model.model.layers
    L = len(layers)
    H = model.config.num_attention_heads
    hd = model.config.head_dim if hasattr(model.config, "head_dim") else model.config.hidden_size // H
    mid, late = L // 2, (3 * L) // 4

    from peft.tuners.lora import LoraLayer
    lora_mods = []
    for name, mod in pm.named_modules():
        if isinstance(mod, LoraLayer) and "rl" in mod.scaling:
            assert "layers." in name, f"LoRA outside the decoder layers: {name}"
            li = int(name.split("layers.")[1].split(".")[0])
            kind = "attn" if ".self_attn." in name else "mlp"
            lora_mods.append((li, kind, mod, mod.scaling["rl"]))

    print(f"{len(lora_mods)} LoRA modules; types {sorted({type(m).__name__ for _, _, m, _ in lora_mods})}; scalings {sorted({s for *_, s in lora_mods})}", flush=True)

    def set_lora(pred):
        for li, kind, mod, s0 in lora_mods:
            mod.scaling["rl"] = s0 if pred(li, kind) else 0.0

    class base_ctx:
        def __enter__(self):
            set_lora(lambda li, k: False)

        def __exit__(self, *a):
            set_lora(lambda li, k: True)

    @torch.no_grad()
    def fwd(m, prefix, y):
        ids = torch.tensor([prefix + y], device=dev)
        out = m(input_ids=ids, output_hidden_states=True, use_cache=False)
        p = len(prefix)
        hs = torch.stack([h[0, p:].double().sum(0) for h in out.hidden_states])
        lp = torch.log_softmax(out.logits[0, p - 1:p - 1 + len(y)].float(), -1)
        return hs, lp

    acc = {}

    def add(name, pid, hs, n, k, kref):
        a = acc.setdefault(name, {"hs": {}, "n": {}, "kl": 0.0, "klref": 0.0})
        if pid in a["hs"]:
            return
        a["hs"][pid] = hs
        a["n"][pid] = n
        a["kl"] += k * n
        a["klref"] += kref * n

    store, patch = {}, {}

    def mk_hook(key):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if key in patch:
                h = patch[key]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            if store.get("_on"):
                store[key] = h.detach().clone()
        return hook

    def mk_pre(key):
        def pre(_m, args_):
            x = args_[0]
            if key in patch:
                hsel, val = patch[key]
                x = x.clone()
                x[..., hsel * hd:(hsel + 1) * hd] = val[..., hsel * hd:(hsel + 1) * hd]
                return (x,)
            if store.get("_on"):
                store[key] = x.detach().clone()
        return pre

    handles = []
    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(mk_hook(("attn", li))))
        handles.append(layer.mlp.register_forward_hook(mk_hook(("mlp", li))))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk_pre(("oin", li))))

    W = args.window
    windows = [(s, min(s + W, L)) for s in range(0, L, W)]
    cuts = list(range(W, L, W))

    def one_text(it, comp=True, heads=None):
        n = len(it["y"])
        with base_ctx():
            hsT, lpT = fwd(pm, it["PT"], it["y"])
            hsB, lpB = fwd(pm, it["PN"], it["y"])
        set_lora(lambda li, k: True)
        store.clear()
        store["_on"] = True
        hsR, lpR = fwd(pm, it["PN"], it["y"])
        store["_on"] = False
        kRB = kl(lpR, lpB)
        add("base_T", it["pid"], hsT, n, kl(lpR, lpT), kRB)
        add("base_N", it["pid"], hsB, n, kRB, kRB)
        add("rl", it["pid"], hsR, n, 0.0, kRB)
        del lpT
        res = {}
        if heads is None:
            conds = [(f"lora_win{a}-{b}", lambda li, k, a=a, b=b: a <= li < b) for a, b in windows]
            conds += [(f"lora_pre{c}", lambda li, k, c=c: li < c) for c in cuts] + [(f"lora_suf{c}", lambda li, k, c=c: li >= c) for c in cuts]
            conds += [("lora_attn", lambda li, k: k == "attn"), ("lora_mlp", lambda li, k: k == "mlp"), ("lora_off", lambda li, k: False)]
            for name, pred in conds:
                set_lora(pred)
                hs, lp = fwd(pm, it["PN"], it["y"])
                add(name, it["pid"], hs, n, kl(lpR, lp), kRB)
                del lp
            set_lora(lambda li, k: True)
            if comp:
                rl_out = {k: v for k, v in store.items() if isinstance(k, tuple) and k[0] in ("attn", "mlp")}
                with base_ctx():
                    for key, val in rl_out.items():
                        patch.clear()
                        patch[key] = val
                        hs, lp = fwd(pm, it["PN"], it["y"])
                        add(f"patch_{key[0]}{key[1]}", it["pid"], hs, n, kl(lpR, lp), kRB)
                        del lp
                    patch.clear()
        else:
            with base_ctx():
                for li in heads:
                    val = store[("oin", li)]
                    for h in (only[li] if only else range(H)):
                        patch.clear()
                        patch[("oin", li)] = (h, val)
                        hs, lp = fwd(pm, it["PN"], it["y"])
                        add(f"head_{li}.{h}", it["pid"], hs, n, kl(lpR, lp), kRB)
                        del lp
                patch.clear()
        store.clear()
        return res

    def parse_layers(spec):
        out = []
        for tok_ in spec.replace("+", ",").split(","):
            a_, _, b_ = tok_.partition("-")
            out += list(range(int(a_), int(b_ or a_) + 1))
        assert out == sorted(set(out)) and 0 <= out[0] and out[-1] < L, f"bad --scan_layers {spec!r}"
        return out

    only = {}
    for tok_ in filter(None, args.heads.split("+")):
        l_, h_ = map(int, tok_.split("."))
        assert 0 <= l_ < L and 0 <= h_ < H, f"bad --heads entry {tok_!r}"
        only.setdefault(l_, []).append(h_)
    scan = [] if args.refs_only else (sorted(only) if only else (parse_layers(args.scan_layers) if args.scan_layers else None))
    sub_items = items[args.head_start: args.head_start + args.head_texts]
    for i, it in enumerate(items):
        one_text(it) if scan is None else one_text(it, heads=[])
        if i % 10 == 0:
            print(f"pass1 {i + 1}/{len(items)}", flush=True)

    def summarise():
        def mean(name, pids):
            return sum(acc[name]["hs"][p] for p in pids) / sum(acc[name]["n"][p] for p in pids)

        allp = list(acc["base_N"]["hs"])
        vth = mean("base_T", allp) - mean("base_N", allp)
        nv = (vth * vth).sum(-1)
        out = {}
        for name, a in acc.items():
            pids = list(a["hs"])
            sh = mean(name, pids) - mean("base_N", pids)
            al = ((sh * vth).sum(-1) / nv).tolist()
            out[name] = {"alpha": al, "alpha_mid": al[mid], "alpha_late": al[late], "rl_rec": 1 - a["kl"] / a["klref"]}
        subp = [it["pid"] for it in sub_items]
        sh = mean("rl", subp) - mean("base_N", subp)
        al = ((sh * vth).sum(-1) / nv).tolist()
        out["rl_sub"] = {"alpha": al, "alpha_mid": al[mid], "alpha_late": al[late], "n_texts": len(subp)}
        return out

    res = summarise()
    if scan is None:
        attn_rank = sorted(range(L), key=lambda l: -(res[f"patch_attn{l}"]["alpha_late"]))
        head_layers = attn_rank[: args.head_layers]
        print("top attention layers by alpha_late:", [(l, round(res[f"patch_attn{l}"]["alpha_late"], 3)) for l in head_layers], flush=True)
    else:
        head_layers = scan
    for i, it in enumerate(sub_items if head_layers else []):
        one_text(it, heads=head_layers)
        print(f"pass2 {i + 1}/{len(sub_items)}", flush=True)
    res = summarise()
    json.dump({"args": vars(args), "L": L, "H": H, "mid": mid, "late": late, "head_layers": head_layers, "cond": res}, open(args.out, "w"))
    print("wrote", args.out)
    if scan is not None:
        return
    show = ["rl", "base_T", "lora_off", "lora_attn", "lora_mlp"] + [f"lora_win{a}-{b}" for a, b in windows]
    for name in show:
        r = res[name]
        print(f"{name:16s} alpha_mid {r['alpha_mid']:+.3f} alpha_late {r['alpha_late']:+.3f} rl_rec {r['rl_rec']:+.3f}")
    for kind in ("attn", "mlp"):
        top = sorted(range(L), key=lambda l: -res[f"patch_{kind}{l}"]["alpha_late"])[:6]
        print(kind, "patch top by alpha_late:", [(l, round(res[f'patch_{kind}{l}']['alpha_late'], 3), round(res[f'patch_{kind}{l}']['rl_rec'], 3)) for l in top])


if __name__ == "__main__":
    main()
