import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="parquet of run A")
ap.add_argument("--b", required=True, help="parquet of run B")
ap.add_argument("--boot", type=int, default=20000)
a = ap.parse_args()
A = pd.read_parquet(a.a).sort_values("problem").reset_index(drop=True)
B = pd.read_parquet(a.b).sort_values("problem").reset_index(drop=True)
assert len(A) == len(B), f"problem sets differ: {len(A)} vs {len(B)}"
pa = (A.correct / A.n).to_numpy()
pb = (B.correct / B.n).to_numpy()
d = pa - pb
rng = np.random.default_rng(0)
idx = rng.integers(0, len(d), size=(a.boot, len(d)))
boot = d[idx].mean(1)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"A pass@1 = {pa.mean():.4f}   B pass@1 = {pb.mean():.4f}")
print(f"diff (A-B) = {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p(diff>0) = {(boot > 0).mean():.3f}")
print(f"problems where A>B: {(d > 0).sum()},  A<B: {(d < 0).sum()},  tie: {(d == 0).sum()}")
