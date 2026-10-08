
import argparse
import json
import os
import re
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render
from think_vector import load_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--gen_dir", required=True, help="think_leak_gen output holding base_nothink.jsonl")
    ap.add_argument("--vectors", required=True, help="*_vectors.pt from think_vector.py (uses v_Fn)")
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--max_resp_tok", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    df = pd.read_parquet(args.data)
    P_N = [tok(render(tok, p, None, enable_thinking=False), add_special_tokens=False)["input_ids"] for p in df["prompt"]]
    items = [(P_N[r["pid"]], r["token_ids"][:args.max_resp_tok]) for r in load_rows(args.gen_dir, "base_nothink") if r["sid"] == 0]
    v = torch.load(args.vectors)["v_Fn"].double()

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    pm = PeftModel.from_pretrained(model, args.adapter, adapter_name="rl")
    pm.eval()
    dev = next(pm.parameters()).device
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    state = {"mask": None, "acc": None}

    def add(key, out):
        m = state["mask"]
        state["acc"][key] = state["acc"].get(key, 0) + (out.float() * m).sum((0, 1)).double()

    hooks = []
    for name, mod in pm.named_modules():
        m = re.search(r"layers\.(\d+)\.(?:self_attn\.o_proj|mlp\.down_proj)$", name)
        if m and hasattr(mod, "base_layer"):
            li, s = int(m.group(1)), mod.scaling["rl"]
            hooks.append(mod.base_layer.register_forward_hook(lambda _m, _i, o, li=li: add(("W", li), o)))
            hooks.append(mod.lora_B["rl"].register_forward_hook(lambda _m, _i, o, li=li, s=s: add(("U", li), o * s)))
    assert len(hooks) == 4 * L, len(hooks)

    @torch.no_grad()
    def run():
        state["acc"] = {}
        hsum, n = torch.zeros(L + 1, d, dtype=torch.float64, device=dev), 0
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
        for b in range(0, len(order), args.bs):
            idx = order[b:b + args.bs]
            T = max(len(items[i][0]) + len(items[i][1]) for i in idx)
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            msk = torch.zeros((len(idx), T, 1), dtype=torch.float32)
            for r, i in enumerate(idx):
                p, y = items[i]
                ids[r, :len(p) + len(y)] = torch.tensor(p + y)
                att[r, :len(p) + len(y)] = 1
                msk[r, len(p):len(p) + len(y)] = 1
            state["mask"] = msk.to(dev)
            hs = pm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
            hsum += torch.stack([(h.float() * state["mask"]).sum((0, 1)).double() for h in hs])
            n += int(msk.sum())
        return {k: x / n for k, x in state["acc"].items()}, hsum / n

    acc_rl, h_rl = run()
    with pm.disable_adapter():
        acc_b, h_b = run()
    for h in hooks:
        h.remove()
    zero = torch.zeros(d, dtype=torch.float64, device=dev)
    direct = torch.stack([zero] + [acc_rl[("U", i)] for i in range(L)]).cumsum(0)
    frozen = torch.stack([zero] + [acc_rl[("W", i)] - acc_b[("W", i)] for i in range(L)]).cumsum(0)
    dk = (h_rl - h_b)
    vd = v.to(dev)
    res = {"layers": [], "note": "index = hidden_states index; l=L is post-final-norm so closure is not expected there"}
    for l in range(1, L + 1):
        vh = torch.nn.functional.normalize(vd[l], dim=0)
        along = float(dk[l] @ vh)
        perp = dk[l] - along * vh
        ph = torch.nn.functional.normalize(perp, dim=0)
        row = {"l": l, "closure": float((direct[l] + frozen[l] - dk[l]).norm() / dk[l].norm().clamp(min=1e-30)),
               "d_along_v": along, "direct_along_v": float(direct[l] @ vh), "frozen_along_v": float(frozen[l] @ vh),
               "d_perp_norm": float(perp.norm()), "direct_along_perp": float(direct[l] @ ph), "frozen_along_perp": float(frozen[l] @ ph),
               "direct_norm": float(direct[l].norm()), "frozen_norm": float(frozen[l].norm())}
        res["layers"].append(row)
        if l % 4 == 2 or l == L:
            print(f"l={l:2d} closure={row['closure']:.3f} | think comp {along:+.3f}: direct {row['direct_along_v'] / along:+.2f} frozen {row['frozen_along_v'] / along:+.2f} "
                  f"| remainder {row['d_perp_norm']:.3f}: direct {row['direct_along_perp'] / row['d_perp_norm']:+.2f} frozen {row['frozen_along_perp'] / row['d_perp_norm']:+.2f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
