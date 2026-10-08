
import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file

WRITERS = ("o_proj", "down_proj")


def energy(B, A):
    return float(torch.trace((B.T @ B) @ (A @ A.T)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--subspace", required=True)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sub = torch.load(args.subspace)
    sd = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    mods = defaultdict(dict)
    for k in sd:
        m = re.match(r"(.*\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+))\.lora_([AB])\.weight", k)
        assert m, k
        mods[m.group(1)][m.group(4)] = k
        mods[m.group(1)]["layer"], mods[m.group(1)]["mod"] = int(m.group(2)), m.group(3)
    total = sum(energy(sd[v["B"]].float(), sd[v["A"]].float()) for v in mods.values())
    wtot = sum(energy(sd[v["B"]].float(), sd[v["A"]].float()) for v in mods.values() if v["mod"] in WRITERS)
    report = {"total_energy": total, "writer_energy_share": wtot / total, "specs": {}}
    gen = torch.Generator().manual_seed(args.seed)

    def target_Q(fam, k, v, B, A):
        if fam == "topsv":
            Qb, Rb = torch.linalg.qr(B)
            U, _, _ = torch.linalg.svd(Rb @ A, full_matrices=False)
            return Qb @ U[:, :k]
        return sub["Q"][fam][v["layer"] + 1][:, :k].float()

    for spec in args.specs:
        parts = spec.split(":")
        out, removed = dict(sd), 0.0
        for name, v in mods.items():
            B, A = sd[v["B"]].float(), sd[v["A"]].float()
            is_w = v["mod"] in WRITERS
            if parts[0] == "only":
                if (parts[1] == "writers") != is_w:
                    removed += energy(B, A)
                    out[v["B"]] = torch.zeros_like(sd[v["B"]])
                continue
            if not is_w:
                continue
            e0 = energy(B, A)
            if parts[0] in ("rm", "keep") and parts[1] != "rand":
                Q = target_Q(parts[1], int(parts[2][1:]), v, B, A)
                Bp = Q @ (Q.T @ B)
                Bn = B - Bp if parts[0] == "rm" else Bp
            else:
                fam, k = (parts[2], int(parts[3][1:])) if parts[0] == "rm" else (parts[1], int(parts[2][1:]))
                Q = target_Q(fam, k, v, B, A)
                f = energy(Q @ (Q.T @ B), A) / max(e0, 1e-30)
                if parts[0] == "scale":
                    Bn = B * (1.0 - f) ** 0.5
                else:
                    kr = max(1, round(f * B.shape[0]))
                    R, _ = torch.linalg.qr(torch.randn(B.shape[0], kr, generator=gen))
                    Bn = B - R @ (R.T @ B)
            removed += e0 - energy(Bn, A)
            out[v["B"]] = Bn.to(sd[v["B"]].dtype).contiguous()
        tag = args.prefix + spec.replace(":", "_")
        dst = os.path.join(args.out_root, tag)
        os.makedirs(dst, exist_ok=True)
        save_file(out, os.path.join(dst, "adapter_model.safetensors"))
        shutil.copy(os.path.join(args.adapter, "adapter_config.json"), dst)
        report["specs"][tag] = {"removed_of_total": removed / total, "removed_of_writers": removed / wtot}
        print(f"{tag:28s} removed {100 * removed / total:6.2f}% of update energy ({100 * removed / wtot:6.2f}% of writer energy)", flush=True)
    json.dump(report, open(os.path.join(args.out_root, args.prefix + "surgery_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
