"""Summarise C3 jsonl shards into a flip-rate table."""
import glob
import json
import sys

import pandas as pd

rows = []
for f in glob.glob("results/c3_*_shard*.jsonl"):
    ds = f.split("c3_")[1].split("_shard")[0]
    for line in open(f):
        r = json.loads(line); r["dataset"] = ds; rows.append(r)
df = pd.DataFrame(rows)
df["margin"] = (df.p1 - 0.5).abs()
if "valid_far" in df:
    v = df.valid_far.fillna(True).astype(bool)
    print("validity rate of far-constrained methods:", df[df.method.isin(["grad_opt_far","near_copies_R"])].groupby("method").valid_far.mean().to_dict())
    df["flipped"] = df.flipped & v
pd.set_option("display.width", 200)
for ds, g in df.groupby("dataset"):
    nt = g.t.nunique()
    print(f"\n== {ds}: {nt} target decisions ==")
    tab = g.pivot_table(index="method", columns="k", values="flipped", aggfunc="mean").round(3)
    print("flip rate\n", tab)
    print("mean p(opposite class) after poisoning\n",
          g.pivot_table(index="method", columns="k", values="p_flip", aggfunc="mean").round(3))
    print("plausibility: dist to nearest real row / median real NN dist (mean)\n",
          g.pivot_table(index="method", columns="k", values="dnn_rel", aggfunc="median").round(2))
    print("dist to target / R_NN (median)\n",
          g.pivot_table(index="method", columns="k", values="dtarget_rel", aggfunc="median").round(2))
    # confident decisions only
    gc = g[g.margin > 0.3]
    print(f"flip rate, confident decisions only (|p-0.5|>0.3, n={gc.t.nunique()})\n",
          gc.pivot_table(index="method", columns="k", values="flipped", aggfunc="mean").round(3))
    o = g[g.method.str.startswith("grad")]
    print("optimiser secs per (target,k) mean:", o.groupby("method").secs.mean().round(2).to_dict())
    # min k to flip per target
    for m in g.method.unique():
        gm = g[g.method == m]
        mk = gm[gm.flipped].groupby("t").k.min()
        print(f"  min-k-to-flip {m:20s}: flipped@<=5 {len(mk)}/{nt}, median k {mk.median() if len(mk) else None}")
