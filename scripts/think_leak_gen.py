
import argparse
import json
import os
import re

import pandas as pd

MARKERS = {
    "wait": r"\bwait\b",
    "hmm": r"\bhmm+\b",
    "alternatively": r"\balternatively\b",
    "let_me_re": r"\blet me (?:re-?\w+|check|verify|double[- ]check|think|reconsider)",
    "no_comma": r"(?:^|[.!?\n]\s*)no,",
    "but_wait": r"\bbut wait\b",
    "actually": r"\bactually\b",
}
MARK_RE = {k: re.compile(v, re.I | re.M) for k, v in MARKERS.items()}
ANY_RE = re.compile("|".join(f"(?:{v})" for v in MARKERS.values()), re.I | re.M)


def text_stats(text, n_tok):
    out = {f"m_{k}": len(r.findall(text)) for k, r in MARK_RE.items()}
    m = ANY_RE.search(text)
    out["n_markers"] = len(ANY_RE.findall(text))
    out["first_marker_relpos"] = (m.start() / max(1, len(text))) if m else None
    out["has_think_open"] = "<think>" in text
    out["has_think_close"] = "</think>" in text
    out["n_tok"] = n_tok
    return out


def render(tok, msgs, template, **kw):
    if template is not None:
        kw["chat_template"] = template
    return tok.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--traj", nargs="*", default=[], help="tag=adapter_dir (evaluated under the no-think template)")
    ap.add_argument("--traj_think", nargs="*", default=[], help="tag=adapter_dir evaluated under the THINK template "
                    "(does no-think RL add anything to think mode?)")
    ap.add_argument("--data", default="data/aime/test.parquet")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--think_max_tokens", type=int, default=4096, help="budget for the base_think reference")
    ap.add_argument("--think_template", default=None, help="jinja file; default = tokenizer template (+enable_thinking=True)")
    ap.add_argument("--nothink_template", default=None, help="jinja file; default = tokenizer template (+enable_thinking=False)")
    ap.add_argument("--template_kwarg", action="store_true", help="pass enable_thinking=True/False (Qwen3-style)")
    ap.add_argument("--single_mode", action="store_true", help="model has no think mode (e.g. Qwen3-4B-Instruct-2507): "
                    "skip the base_think reference, every variant uses the model's only template")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke tests)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from verl.utils.reward_score import default_compute_score
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    df = pd.read_parquet(args.data)
    df = df.head(args.limit) if args.limit else df
    t_think = open(args.think_template).read() if args.think_template else None
    t_noth = open(args.nothink_template).read() if args.nothink_template else None
    kw_t = {"enable_thinking": True} if args.template_kwarg else {}
    kw_n = {"enable_thinking": False} if args.template_kwarg else {}
    p_think = [render(tok, p, t_think, **kw_t) for p in df["prompt"]]
    p_noth = [render(tok, p, t_noth, **kw_n) for p in df["prompt"]]
    assert args.single_mode or p_think[0] != p_noth[0], "think and no-think prompts are identical: template setup is wrong"
    assert not (args.single_mode and args.traj_think), "--traj_think needs a model with a think mode"
    print("THINK  prompt tail:", repr(p_think[0][-60:]))
    print("NOTHINK prompt tail:", repr(p_noth[0][-60:]), flush=True)

    variants = ([] if args.single_mode else [("base_think", None, "think")]) + [("base_nothink", None, "nothink")]
    for spec in args.traj:
        tag, path = spec.split("=", 1)
        variants.append((tag, path, "nothink"))
    for spec in args.traj_think:
        tag, path = spec.split("=", 1)
        variants.append((tag, path, "think"))

    llm = LLM(model=args.base, enable_lora=True, max_lora_rank=64, max_loras=1, gpu_memory_utilization=0.85,
              max_model_len=max(args.max_tokens, args.think_max_tokens) + 2048, dtype="bfloat16")
    stats_path = os.path.join(args.out_dir, "leak_stats.json")
    res = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    for i, (tag, path, mode) in enumerate(variants):
        fn = os.path.join(args.out_dir, f"{tag}.jsonl")
        if tag in res and os.path.exists(fn):
            continue
        mt = args.think_max_tokens if mode == "think" else args.max_tokens
        sp = SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=mt, skip_special_tokens=False)
        prompts = p_think if mode == "think" else p_noth
        lr = LoRARequest(tag, i + 1, path) if path else None
        outs = llm.generate(prompts, sp, lora_request=lr)
        rows = []
        for j, o in enumerate(outs):
            gt = df.iloc[j]["reward_model"]["ground_truth"]
            ds = df.iloc[j].get("data_source", "math_dapo")
            for s, cand in enumerate(o.outputs):
                try:
                    r = default_compute_score(ds, cand.text, gt)
                    ok = bool((r.get("score", 0) if isinstance(r, dict) else r) > 0.5)
                except Exception:
                    ok = False
                row = {"pid": j, "sid": s, "mode": mode, "prompt": prompts[j], "text": cand.text,
                       "token_ids": list(cand.token_ids),
                       "finish": cand.finish_reason, "correct": ok}
                row.update(text_stats(cand.text, len(cand.token_ids)))
                rows.append(row)
        with open(fn, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
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
        res[tag] = agg
        print(tag, json.dumps(agg), flush=True)
        json.dump(res, open(stats_path, "w"), indent=1)
    print("wrote", stats_path)


if __name__ == "__main__":
    main()
