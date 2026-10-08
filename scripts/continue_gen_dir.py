
import argparse
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import MARKERS, text_stats


def aggregate(rows):
    n = len(rows)
    tot_tok = sum(r["n_tok"] for r in rows)
    firsts = [r["first_marker_relpos"] for r in rows if r["first_marker_relpos"] is not None]
    agg = {
        "n_gen": n, "mean_len": tot_tok / n, "stop_frac": sum(r["finish"] == "stop" for r in rows) / n,
        "acc": sum(r["correct"] for r in rows) / n,
        "frac_think_open": sum(r["has_think_open"] for r in rows) / n,
        "frac_think_close": sum(r["has_think_close"] for r in rows) / n,
        "markers_per_1k": 1000.0 * sum(r["n_markers"] for r in rows) / max(1, tot_tok),
        "markers_per_sample": sum(r["n_markers"] for r in rows) / n,
        "frac_with_marker": len(firsts) / n,
        "first_marker_relpos": (sum(firsts) / len(firsts)) if firsts else None,
    }
    for k in MARKERS:
        agg[f"{k}_per_1k"] = 1000.0 * sum(r[f"m_{k}"] for r in rows) / max(1, tot_tok)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--dst_dir", required=True)
    ap.add_argument("--adapters", nargs="*", default=[], help="tag=adapter_dir for every tag that is not the base model")
    ap.add_argument("--max_total", type=int, default=32768)
    ap.add_argument("--tags", nargs="*", default=[], help="only these tags (default: every jsonl in src_dir)")
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from verl.utils.reward_score import default_compute_score
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    os.makedirs(args.dst_dir, exist_ok=True)
    ad = dict(s.split("=", 1) for s in args.adapters)
    tags = sorted(os.path.splitext(f)[0] for f in os.listdir(args.src_dir) if f.endswith(".jsonl"))
    if args.tags:
        tags = [t for t in tags if t in set(args.tags)]
    stats_path = os.path.join(args.dst_dir, "leak_stats.json")
    res = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    df = pd.read_parquet(args.data)
    tok = AutoTokenizer.from_pretrained(args.base)
    llm = None
    for li, tag in enumerate(tags):
        src, dst = os.path.join(args.src_dir, f"{tag}.jsonl"), os.path.join(args.dst_dir, f"{tag}.jsonl")
        if tag in res and os.path.exists(dst):
            continue
        rows = [json.loads(x) for x in open(src)]
        if tag == "base_think" or any(r.get("mode") == "think" for r in rows):
            shutil.copy(src, dst)
            res[tag] = aggregate(rows)
            json.dump(res, open(stats_path, "w"), indent=1)
            print(tag, "copied (think-mode reference)", flush=True)
            continue
        if tag != "base_nothink" and tag not in ad:
            raise SystemExit(f"no adapter given for tag {tag}")
        todo = [i for i, r in enumerate(rows) if r["finish"] != "stop"]
        if todo:
            if llm is None:
                llm = LLM(model=args.base, enable_lora=True, max_lora_rank=64, max_loras=1, gpu_memory_utilization=0.85,
                          max_model_len=args.max_total + 2048, dtype="bfloat16", seed=args.seed)
            prompts, sps = [], []
            for i in todo:
                r = rows[i]
                pids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                prompts.append(TokensPrompt(prompt_token_ids=pids + r["token_ids"]))
                sps.append(SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=args.max_total - len(r["token_ids"]),
                                          skip_special_tokens=False, seed=args.seed * 100000 + li * 1000 + i))
            lr = LoRARequest(tag, li + 1, ad[tag]) if tag in ad else None
            outs = llm.generate(prompts, sps, lora_request=lr)
            for i, o in zip(todo, outs):
                r = rows[i]
                c = o.outputs[0]
                text = r["text"] + c.text
                j = r["pid"]
                try:
                    sc = default_compute_score(df.iloc[j].get("data_source", "math_dapo"), text, df.iloc[j]["reward_model"]["ground_truth"])
                    ok = bool((sc.get("score", 0) if isinstance(sc, dict) else sc) > 0.5)
                except Exception:
                    ok = False
                ids = r["token_ids"] + list(c.token_ids)
                new = dict(r, text=text, token_ids=ids, finish=c.finish_reason, correct=ok,
                           continued=True, prev_len=len(r["token_ids"]), prev_correct=r["correct"])
                new.update(text_stats(text, len(ids)))
                rows[i] = new
        with open(dst, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        res[tag] = aggregate(rows)
        json.dump(res, open(stats_path, "w"), indent=1)
        print(tag, f"continued {len(todo)}", json.dumps({k: res[tag][k] for k in ("acc", "mean_len", "stop_frac")}), flush=True)
    print("wrote", stats_path)


if __name__ == "__main__":
    main()
