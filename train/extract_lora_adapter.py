
import argparse
import json
import os

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, nargs="+", help="one shard, or every rank's shard of a multi-GPU FSDP checkpoint in rank order")
    ap.add_argument("--base", required=True, help="base model path for adapter_config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=None, help="default: = rank")
    a = ap.parse_args()

    sds = [torch.load(p, map_location="cpu", weights_only=False, mmap=True) for p in a.pt]
    sd = sds[0]
    out, mods, rank = {}, set(), None
    for k, v in sd.items():
        if ".lora_A." in k or ".lora_B." in k:
            nk = k.replace(".default.weight", ".weight")
            if hasattr(v, "to_local"):
                dims = [p.dim for p in v.placements if hasattr(p, "dim")]
                locs = [s[k].to_local() for s in sds]
                v = torch.cat(locs, dim=dims[0]) if dims else locs[0]
                assert tuple(v.shape) == tuple(sd[k].shape), (k, v.shape, sd[k].shape)
            out[nk] = v.detach().to(torch.bfloat16).contiguous()
            mods.add(k.split(".lora_")[0].rsplit(".", 1)[-1])
            if ".lora_A." in k and rank is None:
                rank = v.shape[0]
    assert out, "no lora keys in shard"
    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": a.base,
        "r": int(rank), "lora_alpha": float(a.alpha if a.alpha is not None else rank),
        "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
        "target_modules": sorted(mods), "modules_to_save": None,
    }
    json.dump(cfg, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    print(f"wrote {a.out}: {len(out)} tensors, r={rank}, modules={sorted(mods)}")


if __name__ == "__main__":
    main()
