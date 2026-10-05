"""Summarise C4: flip rates per dataset x attack x k x model, with 95% bootstrap CIs
over target decisions, plus the head-to-head question "is gradient optimisation
better than the best black-box attack at the same constraints?".

Usage: python summarize_c4.py [seed]
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

SEED = sys.argv[1] if len(sys.argv) > 1 else "0"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else ""
MODELS = ["tabpfn_diff", "tabpfn_std", "hgb", "rf", "lr", "knn"]
rows = []
for f in sorted(glob.glob(f"results/c4{VARIANT}_*_s{SEED}_shard*.jsonl")):
    ds = f.split(f"c4{VARIANT}_")[1].split(f"_s{SEED}")[0]
    for line in open(f):
        r = json.loads(line)
        r["dataset"] = ds
        rows.append(r)
df = pd.DataFrame(rows)
rng = np.random.default_rng(0)


def ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return (np.percentile(b, 2.5), np.percentile(b, 97.5))


out = {}
for ds, g in df.groupby("dataset"):
    print(f"\n=== {ds}: {g.t.nunique()} decisions ===")
    print("flip rate (95% CI); grad_opt counted only where the model agrees with TabPFN's "
          "decision")
    for k, gk in g.groupby("k"):
        print(f" k={k}")
        for attack, ga in gk.groupby("attack"):
            cells = []
            for m in MODELS:
                v = ga[f"{m}_flip"].dropna().astype(float)
                lo, hi = ci(v)
                cells.append(f"{m} {v.mean():.2f} [{lo:.2f},{hi:.2f}] n={len(v)}")
                out[f"{ds}|k{k}|{attack}|{m}"] = [float(v.mean()), lo, hi, int(len(v))]
            extra = ""
            if attack in ("grad_opt", "near_copies"):
                extra = (f"  valid {ga.valid_far.mean():.2f} dnn {ga.dnn_rel.mean():.2f}")
            print(f"  {attack:13s} " + " | ".join(cells) + extra)
        # paired: grad_opt vs near_copies on tabpfn_std and tabpfn_diff
        for m in ("tabpfn_diff", "tabpfn_std"):
            a = gk[gk.attack == "grad_opt"].set_index("t")[f"{m}_flip"]
            b = gk[gk.attack == "near_copies"].set_index("t")[f"{m}_flip"]
            j = pd.concat([a, b], axis=1, keys=["g", "n"]).dropna().astype(float)
            d = j.g - j.n
            lo, hi = ci(d)
            print(f"  paired grad_opt - near_copies on {m}: {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}]"
                  f" n={len(d)}")
            out[f"{ds}|k{k}|paired_grad_minus_near|{m}"] = [float(d.mean()), lo, hi, int(len(d))]
json.dump(out, open(f"results/c4{VARIANT}_summary_s{SEED}.json", "w"), indent=1)
