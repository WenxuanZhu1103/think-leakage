
import argparse
import glob
import json
import os

import numpy as np

EPS = 1e-6
SCORES = ["llr", "pb", "pbn", "mk", "len"]


def load(fn):
    R = [json.loads(x) for x in open(fn)]
    d = {}
    d["pid"] = np.array([r["pid"] for r in R])
    d["sid"] = np.array([r["sid"] for r in R])
    d["r"] = np.array([r["r_train"] for r in R], float)
    d["rb"] = np.array([r["r_boxed"] for r in R], float)
    d["len"] = np.log(np.array([r["n_tok"] for r in R], float))
    d["mk"] = 1000.0 * np.array([r["n_markers"] for r in R], float) / np.exp(d["len"])
    d["llr"] = np.array([r["llr"] for r in R])
    d["pb"] = np.array([r["pb_resp"] for r in R])
    d["pbn"] = np.array([r["pb_noans"] for r in R])
    d["pb_bnd"] = np.array([r["pb_bnd"] for r in R])
    for k in ("lp_self", "lp_next", "pk_resp", "pn_resp"):
        if k in R[0]:
            d[k] = np.array([r[k] for r in R])
    if "lp_self" not in d:
        d["lp_self"], d["pk_resp"] = np.array([r["lp_nt"] for r in R]), d["pb"].copy()
    d["group"] = d["pid"] * 100 + d["sid"] // 5
    return d


def advantages(r, group):
    A = np.zeros_like(r)
    active = np.zeros_like(r, bool)
    for g in np.unique(group):
        m = group == g
        x = r[m]
        if len(x) > 1 and x.std(ddof=1) > 0:
            A[m] = (x - x.mean()) / (x.std(ddof=1) + EPS)
            active[m] = True
    return A, active


def within(group, a, b, mask=None, all_groups=False):
    ok = np.isfinite(a) & np.isfinite(b)
    vals = []
    for g in np.unique(group):
        m = (group == g) & ok
        if mask is not None:
            if not (m & mask).any():
                if all_groups:
                    vals.append(0.0)
                continue
            m &= mask
        if m.sum() < 2:
            continue
        x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
        vals.append((x * y).mean())
    return float(np.mean(vals)) if vals else float("nan")


def is_pred(pid, s, u):
    ok = np.isfinite(s) & np.isfinite(u)
    preds, ess, wmax = [], [], []
    for p in np.unique(pid):
        m = (pid == p) & ok
        if m.sum() < 2:
            continue
        w = np.exp(u[m] - u[m].max())
        w = w / w.sum()
        preds.append(float((w * s[m]).sum() - s[m].mean()))
        ess.append(float(1.0 / (w * w).sum()))
        wmax.append(float(w.max()))
    return float(np.mean(preds)), float(np.mean(ess)), float(np.mean(wmax))


def within_corr(group, a, b, mask):
    xs, ys = [], []
    for g in np.unique(group):
        m = (group == g) & mask
        if m.sum() < 2:
            continue
        xs.append(a[m] - a[m].mean())
        ys.append(b[m] - b[m].mean())
    if not xs:
        return float("nan")
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-30))


def length_residual(s, L, pid):
    out = np.zeros_like(s)
    ok = np.isfinite(s)
    for half in (0, 1):
        fit, app = (pid % 2 != half) & ok, (pid % 2 == half)
        c = np.polyfit(L[fit], s[fit], 3)
        out[app] = s[app] - np.polyval(c, L[app])
    return out


def boot_prompts(pid, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    P = np.unique(pid)
    rows_of = {p: np.nonzero(pid == p)[0] for p in P}
    vals = []
    for _ in range(n):
        take = rng.choice(P, len(P), replace=True)
        idx = np.concatenate([rows_of[p] for p in take])
        pid_b = np.concatenate([np.full(len(rows_of[p]), j) for j, p in enumerate(take)])
        vals.append(fn(idx, pid_b))
    v = np.array(vals)
    return float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="analysis_out/think_dyn/<ARM>/scores")
    ap.add_argument("--n_boot", type=int, default=500)
    args = ap.parse_args()
    files = {os.path.basename(f)[:-6]: f for f in glob.glob(os.path.join(args.dir, "*.jsonl"))}
    order = ["base"] + sorted([t for t in files if t != "base"], key=lambda t: int(t[1:]))
    D = {t: load(files[t]) for t in order}
    base_pb = np.nanmean(D["base"]["pb"])
    vnorm = None
    print(f"{'ckpt':5s} {'acc':>5s} {'boxed':>5s} {'len':>5s} {'mk/1k':>5s} {'llr':>7s} {'pb-pb0':>7s} {'active':>6s} | "
          f"{'S(llr)':>7s} {'rho':>6s} {'S(pb)':>7s} {'rho':>6s} {'S(pb|L)':>8s} {'S(len)':>7s} {'S(mk)':>7s} | {'Cov(pb,u)':>9s} {'dE[pb]obs':>9s} {'Cov(llr,u)':>10s} {'dE[llr]obs':>10s}")
    res = {}
    for i, t in enumerate(order):
        d = D[t]
        A, act = advantages(d["r"], d["group"])
        row = {"acc": float((d["r"] > 0).mean()), "boxed": float((d["rb"] > 0).mean()), "len": float(np.exp(d["len"]).mean()), "mk": d["mk"].mean(),
               "llr": d["llr"].mean(), "pb": float(np.nanmean(d["pb"]) - base_pb), "active": act.mean()}
        for s in SCORES:
            row[f"S_{s}"] = within(d["group"], A, d[s], act)
            row[f"rho_{s}"] = within_corr(d["group"], A, d[s], act)
        pbL = length_residual(d["pb"], d["len"], d["pid"])
        row["S_pb_L"] = within(d["group"], A, pbL, act)
        row["S_pb_all"] = within(d["group"], A, d["pb"], act, all_groups=True)
        row["S_llr_all"] = within(d["group"], A, d["llr"], act, all_groups=True)
        gb = lambda idx, pid_b: pid_b * 100 + d["sid"][idx] // 5
        row["S_pb_L_ci"] = boot_prompts(d["pid"], lambda idx, pb_: within(gb(idx, pb_), A[idx], pbL[idx], act[idx]), args.n_boot)
        row["S_pb_ci"] = boot_prompts(d["pid"], lambda idx, pb_: within(gb(idx, pb_), A[idx], d["pb"][idx], act[idx]), args.n_boot)
        row["S_llr_ci"] = boot_prompts(d["pid"], lambda idx, pb_: within(gb(idx, pb_), A[idx], d["llr"][idx], act[idx]), args.n_boot)
        row["S_len_ci"] = boot_prompts(d["pid"], lambda idx, pb_: within(gb(idx, pb_), A[idx], d["len"][idx], act[idx]), args.n_boot)
        upd = ""
        if "lp_next" in d and i + 1 < len(order):
            u = d["lp_next"] - d["lp_self"]
            nt = json.loads(open(files[t]).readline()).get("next_tag")
            assert nt == order[i + 1], f"{t}: scored against {nt} but report pairs it with {order[i + 1]}"
            nxt = D[order[i + 1]]
            for s in ("pb", "llr"):
                c = within(d["pid"], d[s], u)
                obs = float(np.nanmean(nxt[s]) - np.nanmean(d[s]))
                row[f"cov_{s}_u"], row[f"dE_{s}_obs"] = c, obs
                row[f"is_{s}"], row["is_ess"], row["is_wmax"] = is_pred(d["pid"], d[s], u)
                row[f"cov_{s}_u_ci"] = boot_prompts(d["pid"], lambda idx, pb_, s=s: within(pb_, d[s][idx], u[idx]), args.n_boot)
            row["u_mean"], row["u_sd"] = float(u.mean()), float(u.std())
            Fkk, Fnk = float(np.nanmean(d["pk_resp"])), float(np.nanmean(d["pn_resp"]))
            Fnn = float(np.nanmean(nxt["pk_resp"])) if "pk_resp" in nxt else float("nan")
            row["fixed_text_act"], row["text_change"] = Fnk - Fkk, Fnn - Fnk
            row["text_change_base"] = float(np.nanmean(nxt["pb"]) - np.nanmean(d["pb"]))
            upd = f"| {row['cov_pb_u']:+9.4f} {row['dE_pb_obs']:+9.4f} {row['cov_llr_u']:+10.5f} {row['dE_llr_obs']:+10.5f}"
        res[t] = row
        print(f"{t:5s} {row['acc']:5.3f} {row['boxed']:5.3f} {row['len']:5.0f} {row['mk']:5.2f} {row['llr']:+7.4f} {row['pb']:+7.3f} {row['active']:6.2f} | "
              f"{row['S_llr']:+7.4f} {row['rho_llr']:+6.3f} {row['S_pb']:+7.4f} {row['rho_pb']:+6.3f} {row['S_pb_L']:+8.4f} {row['S_len']:+7.4f} {row['S_mk']:+7.4f} {upd}")
    print("\nUpdate attribution per checkpoint pair: linear Cov(s,u) vs self-normalised e^u prediction vs observed; ESS of 10 weights, max weight")
    for t in order:
        r = res[t]
        if "is_pb" in r:
            print(f"  {t:5s} pb: lin {r['cov_pb_u']:+.4f} IS {r['is_pb']:+.4f} obs {r['dE_pb_obs']:+.4f} | llr: lin {r['cov_llr_u']:+.5f} IS {r['is_llr']:+.5f} obs {r['dE_llr_obs']:+.5f} | ESS {r['is_ess']:.1f}/10 wmax {r['is_wmax']:.2f}")
    print("\n2x2 decomposition of the policy-measured projection (k -> k+1): fixed-text activation change vs text-distribution change")
    for t in order:
        r = res[t]
        if "fixed_text_act" in r:
            print(f"  {t:5s} fixed-text {r['fixed_text_act']:+.4f}   text(model k+1) {r['text_change']:+.4f}   text(base) {r['text_change_base']:+.4f}")
    print("\n95% prompt-bootstrap CIs (multiplicity-correct): S(pb) / S(llr) / S(len) / Cov(pb,u); S_all = averaged over all groups (inactive count as 0)")
    for t in order:
        r = res[t]
        c = r.get("cov_pb_u_ci", (float("nan"),) * 2)
        print(f"  {t:5s} S(pb) [{r['S_pb_ci'][0]:+.4f},{r['S_pb_ci'][1]:+.4f}] all {r['S_pb_all']:+.4f} | S(llr) [{r['S_llr_ci'][0]:+.5f},{r['S_llr_ci'][1]:+.5f}] all {r['S_llr_all']:+.5f} | "
              f"S(len) [{r['S_len_ci'][0]:+.4f},{r['S_len_ci'][1]:+.4f}] | Cov(pb,u) [{c[0]:+.4f},{c[1]:+.4f}]")
    json.dump(res, open(os.path.join(args.dir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
