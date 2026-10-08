
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from think_leak_gen import text_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_total", type=int, default=65536)
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from verl.utils.reward_score import default_compute_score
    from vllm import LLM, SamplingParams, TokensPrompt

    tok = AutoTokenizer.from_pretrained(args.base)
    df = pd.read_parquet(args.data)
    rows = [json.loads(x) for x in open(args.inp)]
    todo = [i for i, r in enumerate(rows) if r["finish"] != "stop"]
    print(f"{len(rows)} rows, {len(todo)} truncated", flush=True)
    if todo:
        native = 40960
        kw = {}
        if args.max_total > native:
            kw["hf_overrides"] = {"rope_scaling": {"rope_type": "yarn", "factor": args.max_total / 32768, "original_max_position_embeddings": 32768},
                                  "max_position_embeddings": args.max_total + 1024}
        llm = LLM(model=args.base, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=min(args.max_total + 1024, native) if args.max_total <= native else args.max_total + 1024,
                  seed=args.seed, **kw)
        prompts, sps = [], []
        for i in todo:
            r = rows[i]
            pids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            prompts.append(TokensPrompt(prompt_token_ids=pids + r["token_ids"]))
            room = args.max_total - len(r["token_ids"])
            if args.max_total <= native:
                room = min(room, native - len(pids) - len(r["token_ids"]))
            sps.append(SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=room,
                                      skip_special_tokens=False, seed=args.seed * 100000 + i))
        outs = llm.generate(prompts, sps)
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
            new = dict(r, text=text, token_ids=r["token_ids"] + list(c.token_ids), finish=c.finish_reason, correct=ok,
                       continued=True, prev_len=len(r["token_ids"]), prev_correct=r["correct"])
            new.update(text_stats(text, len(new["token_ids"])))
            rows[i] = new
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    cont = [r for r in rows if r.get("continued")]
    print(json.dumps({"n": n, "acc": sum(r["correct"] for r in rows) / n, "finished": sum(r["finish"] == "stop" for r in rows) / n,
                      "mean_len": sum(len(r["token_ids"]) for r in rows) / n, "continued": len(cont),
                      "continued_finished": sum(r["finish"] == "stop" for r in cont), "continued_correct": sum(r["correct"] for r in cont)}))


if __name__ == "__main__":
    main()
