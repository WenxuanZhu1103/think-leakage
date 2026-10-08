import argparse
import os
import shutil

from safetensors.torch import load_file, save_file


def write_scaled_adapter(src, dst, c):
    sd = load_file(os.path.join(src, "adapter_model.safetensors"))
    out = {}
    n_scaled = 0
    for k, v in sd.items():
        if ".lora_B." in k:
            out[k] = (v.float() * c).to(v.dtype)
            n_scaled += 1
        else:
            out[k] = v
    os.makedirs(dst, exist_ok=True)
    save_file(out, os.path.join(dst, "adapter_model.safetensors"))
    shutil.copy(os.path.join(src, "adapter_config.json"), dst)
    return n_scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--scales", type=float, nargs="+", required=True)
    a = ap.parse_args()
    for c in a.scales:
        dst = f"{a.out_root}/{a.prefix}_c{c}"
        if os.path.exists(os.path.join(dst, "adapter_model.safetensors")):
            continue
        n = write_scaled_adapter(a.src, dst, c)
        print(f"wrote {dst}: {n} lora_B tensors scaled by {c}")


if __name__ == "__main__":
    main()
