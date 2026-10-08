
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import render, text_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--traj", nargs="+", required=True, help="tag=adapter_dir; tag 'base' = no adapter")
    ap.add_argument("--data", default="data/dapo17k/train.parquet")
    ap.add_argument("--n_prompts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=10, help="rollouts per prompt for checkpoints (2 groups of the training group size 5)")
    ap.add_argument("--n_base", type=int, default=20, help="rollouts per prompt for the base (4 groups of 5)")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from verl.utils.reward_score import default_compute_score, math_reward
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    df = pd.read_parquet(args.data).sample(n=args.n_prompts, random_state=args.seed).reset_index(drop=True)
    df[["prompt", "reward_model"]].to_json(os.path.join(args.out_dir, "prompts.jsonl"), orient="records", lines=True, force_ascii=False)
    prompts = [render(tok, p, None, enable_thinking=False) for p in df["prompt"]]
    llm = LLM(model=args.base, enable_lora=True, max_lora_rank=64, max_loras=1, gpu_memory_utilization=0.85,
              max_model_len=args.max_tokens + 2048, dtype="bfloat16")
    for i, spec in enumerate(args.traj):
        tag, path = spec.split("=", 1)
        fn = os.path.join(args.out_dir, f"{tag}.jsonl")
        if os.path.exists(fn):
            continue
        lr = LoRARequest(tag, i + 1, path) if tag != "base" else None
        sp = SamplingParams(n=args.n_base if tag == "base" else args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                            skip_special_tokens=False, seed=args.seed)
        outs = llm.generate(prompts, sp, lora_request=lr)
        rows, rt, rb = [], 0, 0
        for j, o in enumerate(outs):
            gt = df.iloc[j]["reward_model"]["ground_truth"]
            for s, c in enumerate(o.outputs):
                try:
                    r = default_compute_score("math_dapo", c.text, gt)
                    r_train = float(r.get("score", 0) if isinstance(r, dict) else r)
                except Exception:
                    r_train = -1.0
                assert r_train in (-1.0, 1.0), r_train
                try:
                    r_boxed = float(math_reward.compute_score(c.text, gt))
                except Exception:
                    r_boxed = 0.0
                row = {"pid": j, "sid": s, "token_ids": list(c.token_ids), "text": c.text, "r_train": r_train, "r_boxed": r_boxed,
                       "stopped": c.finish_reason == "stop"}
                row.update(text_stats(c.text, len(c.token_ids)))
                rows.append(row)
                rt += r_train > 0
                rb += r_boxed > 0
        with open(fn + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(fn + ".tmp", fn)
        print(f"{tag}: n={len(rows)} r_train={rt / len(rows):.3f} r_boxed={rb / len(rows):.3f} "
              f"len={sum(r['n_tok'] for r in rows) / len(rows):.0f} mk/1k={1000 * sum(r['n_markers'] for r in rows) / max(1, sum(r['n_tok'] for r in rows)):.2f}", flush=True)


if __name__ == "__main__":
    main()
