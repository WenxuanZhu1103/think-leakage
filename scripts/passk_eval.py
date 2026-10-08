
import argparse
import json
import re
import os

import numpy as np
import pandas as pd
from scipy.special import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return float(1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir or hub id")
    ap.add_argument("--name", required=True, help="label for reporting")
    ap.add_argument("--data", required=True, help="eval parquet (prompt + reward_model.ground_truth)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="sample n in chunks of this size (distinct seed per chunk); 0 = single shot. "
        "Required for very large n: bounds host memory and keeps results streaming.",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--prefill", default=None, help="text appended after the generation prompt (e.g. '<think>\\n') to force a mode")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--save_texts", action="store_true")
    ap.add_argument("--min_tokens", type=int, default=0, help="force generation to continue (suppress EOS) until this many tokens: buys length without touching weights")
    ap.add_argument("--scorer", default="default", choices=["default", "aime_last_int"],
                    help="aime_last_int: strip <think>, take the last integer in [0,999] (Open-MOPD style flexible extractor)")
    ap.add_argument("--dtype", default=None, help="vLLM dtype override; float32 keeps tiny weight deltas that bf16 would round away")
    ap.add_argument(
        "--no_thinking",
        action="store_true",
        help="pass enable_thinking=False to the chat template (Qwen3 hybrid models). MUST match "
        "how the model was trained -- a thinking/non-thinking mismatch silently changes the task",
    )
    ap.add_argument(
        "--lora",
        default=None,
        help="LoRA adapter dir (adapter_config.json + adapter_model.safetensors); "
        "served natively by vLLM on top of --model, so no merged copy is needed",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.utils.reward_score import default_compute_score

    def aime_last_int(text, gt):
        body = text.split("</think>")[-1] if "</think>" in text else text
        nums = re.findall(r"-?\d+", body.replace(",", ""))
        if not nums:
            return 0.0
        try:
            return 1.0 if int(nums[-1]) == int(str(gt).strip()) else 0.0
        except ValueError:
            return 0.0

    df = pd.read_parquet(args.data)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            list(r["prompt"].tolist() if hasattr(r["prompt"], "tolist") else r["prompt"]),
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": False} if args.no_thinking else {}),
        )
        for _, r in df.iterrows()
    ]
    if args.prefill:
        prompts = [p + args.prefill for p in prompts]
    lora_request = None
    lora_kwargs = {}
    if args.lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("adapter", 1, args.lora)
        lora_kwargs = {"enable_lora": True, "max_lora_rank": 64}
    if args.dtype:
        lora_kwargs["dtype"] = args.dtype
    llm = LLM(
        model=args.model, gpu_memory_utilization=0.9, max_model_len=args.max_model_len, seed=0, **lora_kwargs
    )

    chunk = args.chunk if args.chunk > 0 else args.n
    if chunk > args.n:
        chunk = args.n
    assert args.n % chunk == 0, f"n={args.n} must be divisible by chunk={chunk}"
    n_prob = len(df)
    n_chunks = args.n // chunk
    c_tot = np.zeros(n_prob, dtype=np.int64)
    len_sum = np.zeros(n_prob, dtype=np.float64)
    dumps = [] if args.save_texts else None

    os.makedirs(args.out_dir, exist_ok=True)
    part_path = os.path.join(args.out_dir, f"partial_{args.name}.npz")
    start_chunk = 0
    if os.path.exists(part_path):
        try:
            z = np.load(part_path, allow_pickle=False)
            if int(z["n"]) == args.n and int(z["chunk"]) == chunk and int(z["n_prob"]) == n_prob:
                c_tot, len_sum, start_chunk = z["c_tot"], z["len_sum"], int(z["done"])
                print(f"resuming from chunk {start_chunk}/{n_chunks}", flush=True)
            else:
                print("partial file has a different config; ignoring", flush=True)
        except Exception as e:
            print(f"partial file unreadable ({e}); starting fresh", flush=True)

    for ci in range(start_chunk, n_chunks):
        outs = llm.generate(
            prompts,
            SamplingParams(n=chunk, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, min_tokens=args.min_tokens, seed=ci),
            lora_request=lora_request,
        )
        for i, out in enumerate(outs):
            gt = df.iloc[i]["reward_model"]["ground_truth"]
            ds = df.iloc[i]["data_source"]
            for cand in out.outputs:
                try:
                    res = aime_last_int(cand.text, gt) if args.scorer == "aime_last_int" else default_compute_score(ds, cand.text, gt)
                    ok = float(res["score"] if isinstance(res, dict) else res) > 0.5
                except Exception:
                    ok = False
                c_tot[i] += int(ok)
                len_sum[i] += len(cand.token_ids)
                if dumps is not None:
                    dumps.append(
                        {
                            "problem": i,
                            "chunk": ci,
                            "correct": bool(ok),
                            "n_tokens": len(cand.token_ids),
                            "truncated": cand.finish_reason == "length",
                            "text": cand.text,
                        }
                    )
        np.savez(
            part_path + ".tmp.npz", c_tot=c_tot, len_sum=len_sum, done=ci + 1,
            n=args.n, chunk=chunk, n_prob=n_prob,
        )
        os.replace(part_path + ".tmp.npz", part_path)
        print(
            f"[chunk {ci + 1}/{n_chunks}] running pass@1 = {c_tot.sum() / ((ci + 1) * chunk * n_prob):.4f}",
            flush=True,
        )

    rec = [
        {
            "problem": i,
            "n": args.n,
            "correct": int(c_tot[i]),
            "mean_len": float(len_sum[i] / args.n),
            "level": str(df.iloc[i].get("level", "?")),
        }
        for i in range(n_prob)
    ]
    rd = pd.DataFrame(rec)
    os.makedirs(args.out_dir, exist_ok=True)
    rd.to_parquet(os.path.join(args.out_dir, f"passk_{args.name}.parquet"))

    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if k <= args.n]
    result = {f"pass@{k}": float(np.mean([pass_at_k(r.n, r.correct, k) for r in rd.itertuples()])) for k in ks}
    result.update(
        {
            "name": args.name,
            "n": args.n,
            "temperature": args.temperature,
            "mean_response_len": float(rd.mean_len.mean()),
            "frac_all_wrong": float((rd.correct == 0).mean()),
            "frac_all_correct": float((rd.correct == rd.n).mean()),
        }
    )
    with open(os.path.join(args.out_dir, f"passk_{args.name}.json"), "w") as f:
        json.dump(result, f, indent=1)
    if dumps is not None:
        pd.DataFrame(dumps).to_parquet(os.path.join(args.out_dir, f"texts_{args.name}.parquet"))
        print(f"saved {len(dumps)} generations to texts_{args.name}.parquet")
    if os.path.exists(part_path):
        os.remove(part_path)
    print(f"\n=== {args.name} (n={args.n}, T={args.temperature}) ===")
    for k in ks:
        print(f"  pass@{k:<2} = {result[f'pass@{k}']:.4f}")
    print(
        f"  mean_len={result['mean_response_len']:.0f}  all_wrong={result['frac_all_wrong']:.3f}"
        f"  all_correct={result['frac_all_correct']:.3f}"
    )


if __name__ == "__main__":
    main()
