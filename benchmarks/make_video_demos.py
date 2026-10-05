"""Build the demo-video tables: Auto MPG (real) + 8 scripted errors with stories. Writes demo/video/:
  auto_mpg.csv                 clean real table (UCI, CC BY 4.0)
  auto_mpg.proofread.json      Proofread report on the clean table (real errors only)
  auto_mpg_dirty.csv           with the 8 scripted errors
  auto_mpg_dirty.proofread.json
  answer_key.json              every planted error: story, truth, rank in Proofread's list, z-score, caught or not
Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python benchmarks/make_video_demos.py
"""
import json
from pathlib import Path
import numpy as np
from benchmarks.video_data import load_auto_mpg
from proofread import Proofreader

OUT = Path("demo/video"); OUT.mkdir(parents=True, exist_ok=True)
PLANTED = [
    (181, "weight", 4354.0, "pasted the Chevrolet Impala's weight (row 6) into the Honda Civic's row"),
    (0, "weight", 35040.0, "typed an extra zero"),
    (166, "displacement", 5.0, "entered litres (5.0 L) instead of cubic inches"),
    (131, "mpg", 23.0, "swapped two digits (32 → 23)"),
    (175, "horsepower", 0.0, "missing value recorded as 0"),
    (81, "acceleration", 1.7, "dropped the decimal place (17.0 → 1.7)"),
    (31, "model_year", 1917, "swapped two digits (1971 → 1917)"),
    (14, "origin", "USA", "wrong category: a Toyota marked as made in the USA"),
]
REAL = [(13, "weight", "1970 Buick Estate Wagon listed at 3,086 lb; published curb weight 4,762–4,775 lb "
         "(ultimatespecs.com) or 4,900–5,000 lb (Wikipedia). The value is also in copies such as seaborn's mpg.csv.")]

clean = load_auto_mpg()
clean.to_csv(OUT / "auto_mpg.csv", index=False)
pr = Proofreader(device="cuda:0")
rep_clean = pr.check(clean, label="origin", threshold=2.0, max_issues=40, categorical=False)   # numbers in demo/video/README.md
rep_clean.meta.update(title="Auto MPG (real, untouched)", source_file="auto_mpg.csv"); rep_clean.save(OUT / "auto_mpg.proofread.json")

dirty = clean.copy()
for i, c, v, _ in PLANTED:
    dirty[c] = dirty[c].astype(object) if isinstance(v, str) else dirty[c]
    dirty.at[i, c] = v
dirty.to_csv(OUT / "auto_mpg_dirty.csv", index=False)
rep = pr.check(dirty, label="origin", threshold=2.0, max_issues=60, categorical=False)
rep.meta.update(title="Auto MPG with 8 scripted errors", source_file="auto_mpg_dirty.csv")

order = [(int(r.row), str(r.column)) for r in rep.issues.itertuples()]
num = dirty.select_dtypes("number")      # z-scores as a user would compute them: on the table as it is (dirty)
key = []
for i, c, v, story in PLANTED + [(r, c, None, s) for r, c, s in REAL]:
    rank = order.index((i, c)) + 1 if (i, c) in order else None
    z = None
    if c in num.columns and v is not None:
        z = float((v - num[c].mean()) / num[c].std())
    iss = rep.issues.iloc[rank - 1] if rank else None
    key.append({"row": i, "car": clean.at[i, "car"], "year": int(clean.at[i, "model_year"]), "column": c,
                "true_value": None if v is None else (clean.at[i, c].item() if hasattr(clean.at[i, c], "item") else clean.at[i, c]),
                "recorded": dirty.at[i, c] if v is not None else clean.at[i, c], "story": story, "planted": v is not None,
                "proofread_rank": rank, "surprise": None if iss is None else round(float(iss.surprise), 2),
                "suggested": None if iss is None else (iss.suggested if isinstance(iss.suggested, str) else round(float(iss.suggested), 4)),
                "cause": None if iss is None else iss.cause, "column_z_score": None if z is None else round(z, 2),
                "surprise_if_not_listed": (round(float(rep.cell_surprise.at[i, c]), 2) if iss is None and c in rep.cell_surprise.columns else None),
                "z_score_flags_it": None if z is None else abs(z) > 3})
rep.meta["answer_key"] = key
rep.meta["ground_truth"] = [{"row": k["row"], "column": k["column"], "kind": k["story"], "true_value": k["true_value"]}
                            for k in key if k["planted"] and isinstance(k["true_value"], (int, float))]
rep.save(OUT / "auto_mpg_dirty.proofread.json")
(OUT / "answer_key.json").write_text(json.dumps(key, indent=2, default=str))
print(f"clean table: top issues")
for r in rep_clean.issues.head(6).itertuples():
    print(f"  {r.surprise:4.1f} row {r.row:3d} {clean.at[r.row, 'car']:30s} {r.column:12s} {r.value!s:>8.6} -> {r.suggested!s:>8.6}")
print(f"dirty table: {len(rep.issues)} issues listed; planted + real errors:")
for k in key:
    print(f"  rank {str(k['proofread_rank']):>4} | z {str(k['column_z_score']):>6} (z-flag {k['z_score_flags_it']}) | {k['car'][:24]:24s} {k['column']:12s} "
          f"{str(k['recorded']):>8} (true {k['true_value']}) -> TabPFN {k['suggested']} | {k['cause']} | surprise-if-missed {k['surprise_if_not_listed']}")
print("seconds", round(rep.meta["seconds"]))
