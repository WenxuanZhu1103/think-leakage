import glob
import json
import sys

import numpy as np
from scipy.stats import hypergeom, spearmanr

C = "analysis_out/think_circuit"


def load(m):
    refs = json.load(open(f"{C}/route_{m}_refs.json"))
    L, H, late, mid = refs["L"], refs["H"], refs["late"], refs["mid"]
    hp = {}
    for f in sorted(glob.glob(f"{C}/route_{m}_scan*.json")):
        for k, v in json.load(open(f))["cond"].items():
            if k.startswith("head_"):
                assert k[5:] not in hp, (f, k)
                hp[k[5:]] = v
    g = json.load(open(f"{C}/gate_{m}.json"))
    M = np.array(g["mass"]["base_N"])
    return refs, L, H, late, mid, hp, M, g


def analyse(m, verbose=True):
    refs, L, H, late, mid, hp, M, g = load(m)
    rl = refs["cond"]["rl_sub"]["alpha"][late]
    order = np.argsort(-M.ravel())
    arank = {f"{i // H}.{i % H}": r + 1 for r, i in enumerate(order)}
    elig = [h for h in hp if int(h.split(".")[0]) < late]
    share = {h: hp[h]["alpha"][late] / rl for h in elig}
    mass = {h: float(M[int(h.split(".")[0]), int(h.split(".")[1])]) for h in elig}
    top = sorted(elig, key=lambda h: -share[h])
    atop = sorted(elig, key=lambda h: -mass[h])
    rng = np.random.default_rng(0)
    ranks = np.array([arank[h] for h in elig])
    out = {"model": m, "L": L, "H": H, "late": late, "n_heads": L * H, "n_scanned": len(hp), "n_eligible": len(elig),
           "complete": len(hp) == L * H, "rl_alpha_late_sub": rl}
    for k in (5, 10):
        obs = float(np.mean([arank[h] for h in top[:k]]))
        null = np.array([ranks[rng.choice(len(elig), k, replace=False)].mean() for _ in range(100000)])
        ov = len(set(top[:k]) & set(atop[:k]))
        out[f"top{k}"] = {"mean_rank": obs, "null_mean_rank": float(null.mean()), "p": float((null <= obs).mean()),
                          "overlap": ov, "overlap_p": float(hypergeom.sf(ov - 1, len(elig), k, k)),
                          "share_sum": float(sum(share[h] for h in top[:k]))}
    n_top = int(round(0.05 * L * H))
    in_top = {h for h in elig if arank[h] <= n_top}
    k_in = sum(h in in_top for h in top[:5])
    out["top5pct"] = {"n_top": n_top, "n_top_eligible": len(in_top), "top5_in": k_in,
                      "p": float(hypergeom.sf(k_in - 1, len(elig), len(in_top), 5)),
                      "top3_in": sum(h in in_top for h in top[:3]),
                      "p3": float(hypergeom.sf(sum(h in in_top for h in top[:3]) - 1, len(elig), len(in_top), 3)),
                      "exp3": 3 * len(in_top) / len(elig),
                      "top10_in": sum(h in in_top for h in top[:10]),
                      "share_ranks_of_top": sorted(top.index(h) + 1 for h in in_top)[:10]}
    rho, p = spearmanr([share[h] for h in elig], [mass[h] for h in elig])
    out["spearman"] = {"rho": float(rho), "p": float(p)}
    out["top"] = [{"head": h, "label": "L{}H{}".format(*(int(x) + 1 for x in h.split("."))), "share": share[h],
                   "attn_mass": mass[h], "attn_rank": arank[h], "rl_rec": hp[h]["rl_rec"]} for h in top[:10]]
    out["attn_top"] = [{"head": h, "label": "L{}H{}".format(*(int(x) + 1 for x in h.split("."))), "share": share[h],
                        "attn_mass": mass[h], "attn_rank": arank[h], "share_rank": top.index(h) + 1} for h in atop[:10]]
    sh = np.array([share[h] for h in elig])
    out["share_stats"] = {"sum": float(sh.sum()), "n_gt_5pct": int((sh > 0.05).sum()), "n_gt_10pct": int((sh > 0.10).sum()),
                          "n_lt_m5pct": int((sh < -0.05).sum()), "min": float(sh.min()), "median_abs": float(np.median(np.abs(sh)))}
    out["neg"] = [{"head": h, "share": share[h], "attn_rank": arank[h]} for h in top[::-1][:5]]
    if verbose:
        print(f"== {m}: scanned {len(hp)}/{L * H} heads, eligible (layers 1..{late}) {len(elig)}, RL alpha at layer {late} on scan texts {rl:.4f}")
        print("   top by share:", [(t["label"], f"{t['share']:.0%}", f"mass {t['attn_mass']:.3f}", f"rank {t['attn_rank']}") for t in out["top"][:8]])
        print("   top by attention (eligible):", [(t["label"], f"mass {t['attn_mass']:.3f}", f"share {t['share']:+.0%}", f"share-rank {t['share_rank']}") for t in out["attn_top"][:8]])
        for k in (5, 10):
            t = out[f"top{k}"]
            print(f"   top{k}: mean attn rank {t['mean_rank']:.1f} vs random {t['null_mean_rank']:.0f} (p={t['p']:.5f}); overlap with attn top{k}: {t['overlap']} (p={t['overlap_p']:.1e}); sum of shares {t['share_sum']:.2f}")
        print(f"   spearman rho {rho:.3f} p={p:.1e}; share stats {out['share_stats']}")
        print(f"   top-5% by attention: {out['top5pct']}")
        print("   most negative:", [(t["head"], f"{t['share']:.0%}", t["attn_rank"]) for t in out["neg"]])
    return out


if __name__ == "__main__":
    res = {m: analyse(m) for m in (sys.argv[1:] or ["8b", "4b", "17"])}
    json.dump(res, open(f"{C}/route_summary.json", "w"), indent=1)
