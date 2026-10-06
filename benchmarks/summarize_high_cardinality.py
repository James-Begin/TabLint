"""Validate and analyse all 15 supplied tables; render a separate version figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("v3|num", "v3.5|num", "v3|hc", "v3.5|hc")
DATASETS = {"auto_mpg": 392, "employee_salaries": 1000, "medical_charges": 1000}
SEEDS = set(range(1001, 1006))
METRICS = ("precision_at_k", "auroc", "seconds")
KINDS = {"decimal", "swap", "offset", "zero", "transposition"}


def load_records(directory, *, verify_hashes=False):
    """Reject partial, duplicate or unexpected studies rather than cherry-pick runs."""
    records = []
    hashes = json.loads((ROOT / "benchmarks/high_cardinality_protocol/provenance.json").read_text())["raw_results_sha256"]
    for path in sorted(Path(directory).glob("*.json")):
        if path.name in {"summary.json", "environment.json"}:
            continue
        record = json.loads(path.read_text())
        dataset, seed = record.get("dataset"), record.get("seed")
        if dataset not in DATASETS or seed not in SEEDS or path.stem != f"{dataset}_{seed}":
            raise ValueError(f"Unexpected study record: {path.name}")
        if record["n"] != DATASETS[dataset] or not 0 < record["errors"] <= .03 * record["n"] * (5 if dataset == "auto_mpg" else 4):
            raise ValueError(f"Unexpected rows or corruption count: {path.name}")
        if set(record["runs"]) != set(ARMS):
            raise ValueError(f"Missing or unexpected model/context arms: {path.name}")
        for run in record["runs"].values():
            if any(not np.isfinite(run[key]) or not 0 <= run[key] <= 1 for key in METRICS[:2]):
                raise ValueError(f"Invalid metric: {path.name}")
            if not np.isfinite(run["seconds"]) or run["seconds"] <= 0:
                raise ValueError(f"Invalid timing: {path.name}")
            if abs(run["precision_at_k"] * record["errors"] - round(run["precision_at_k"] * record["errors"])) > 1e-8:
                raise ValueError(f"Precision@k incompatible with review budget: {path.name}")
            if set(run["by_type"]) != KINDS or any(not np.isfinite(value) or not 0 <= value <= 1 for value in run["by_type"].values()):
                raise ValueError(f"Invalid per-error-type AUROC: {path.name}")
        if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(path.name):
            raise ValueError(f"Supplied result checksum differs: {path.name}")
        records.append(record)
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if len(records) != 15 or {(r["dataset"], r["seed"]) for r in records} != expected:
        raise ValueError("Expected all three datasets × five seeds (15 paired tables)")
    return records


def paired_test(differences):
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    return {"mean_difference": float(differences.mean()),
            "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
            "ties": int((differences == 0).sum()),
            "wilcoxon_p_one_sided": float(wilcoxon(nonzero, alternative="greater").pvalue) if len(nonzero) else 1.0}


def summarize(records):
    output = {"tables": len(records), "arm_records": len(records) * 4,
              "datasets": len(DATASETS), "seeds": sorted(SEEDS),
              "test_unit": "dataset/seed table pairs; repeated seeds within three datasets",
              "per_dataset": [], "metrics": {}}
    for dataset in sorted(DATASETS):
        rows = [r for r in records if r["dataset"] == dataset]
        output["per_dataset"].append({"dataset": dataset, "n": rows[0]["n"], "seeds": len(rows),
            "errors_range": [min(r["errors"] for r in rows), max(r["errors"] for r in rows)],
            "levels_range": {c: [min(r["high_card_levels_in_sample"][c] for r in rows), max(r["high_card_levels_in_sample"][c] for r in rows)]
                             for c in rows[0]["high_card_levels_in_sample"]},
            "means": {metric: {arm: float(np.mean([r["runs"][arm][metric] for r in rows])) for arm in ARMS} for metric in METRICS}})
    for metric in METRICS[:2]:
        values = lambda arm: np.array([r["runs"][arm][metric] for r in records])
        hc = values("v3.5|hc") - values("v3|hc")
        num = values("v3.5|num") - values("v3|num")
        interaction = (values("v3.5|hc") - values("v3.5|num")) - (values("v3|hc") - values("v3|num"))
        output["metrics"][metric] = {"means": {arm: float(values(arm).mean()) for arm in ARMS},
            "h1_version_with_context": paired_test(hc),
            "h2_context_interaction": paired_test(interaction),
            "version_without_high_cardinality": paired_test(num)}
    output["mean_seconds"] = {arm: float(np.mean([r["runs"][arm]["seconds"] for r in records])) for arm in ARMS}
    return output


def write_values(records, directory):
    with (directory / "values.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset", "seed", "rows", "errors", "arm", *METRICS], lineterminator="\n")
        writer.writeheader()
        for record in records:
            for arm in ARMS:
                writer.writerow({"dataset": record["dataset"], "seed": record["seed"], "rows": record["n"],
                                 "errors": record["errors"], "arm": arm, **{m: record["runs"][arm][m] for m in METRICS}})


def render(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from benchmarks.make_showcase_figures import save

    green, slate, ink, muted = "#157F79", "#687789", "#182B39", "#526373"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "TabLint-high-cardinality"})
    fig = plt.figure(figsize=(12, 7.4), facecolor="white")
    fig.text(.045, .952, "TABLINT  /  TABPFN VERSION STUDY", color=green, fontsize=11, weight="bold")
    fig.text(.045, .899, "Keep the context. Compare the versions.", color=ink, fontsize=24, weight="bold")
    h1 = summary["metrics"]["precision_at_k"]["h1_version_with_context"]
    fig.text(.045, .835, f"3.5 gains {100*h1['mean_difference']:.1f} points over v3 with added context", color=green, fontsize=19, weight="bold")
    fig.text(.045, .791, "All three datasets shown · five seeds each · paired numerical cell-error detection on CPU", color=muted, fontsize=11)
    labels = {"auto_mpg": "Auto MPG\n392 rows", "employee_salaries": "Employee salaries\n1,000 rows", "medical_charges": "Medical charges\n1,000 rows"}
    for i, arm in enumerate(("num", "hc")):
        axis = fig.add_axes([.18 + i*.415, .235, .35, .455])
        axis.set_title("Without high-cardinality context" if arm == "num" else "With high-cardinality context", loc="left", fontsize=12, color=ink, pad=13)
        for j, row in enumerate(summary["per_dataset"]):
            for offset, model_name, color in ((-.17, "v3", slate), (.17, "v3.5", green)):
                value = row["means"]["precision_at_k"][f"{model_name}|{arm}"]
                axis.barh(j+offset, value, height=.27, color=color, zorder=3)
                axis.text(value+.018, j+offset, f"{100*value:.1f}%", va="center", fontsize=10, color=ink)
        axis.set_xlim(0, 1.08)
        axis.set_ylim(2.65, -.65)
        axis.set_yticks(range(3), [labels[row["dataset"]] for row in summary["per_dataset"]] if i == 0 else [""]*3, fontsize=11, color=ink)
        axis.set_xticks([0, .25, .5, .75, 1.])
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.set_xlabel("Precision@k · higher is better", fontsize=10, color=muted, labelpad=10)
        axis.grid(axis="x", color="#E5EBEF", zorder=0)
        axis.tick_params(length=0, colors=muted, pad=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.text(.64, .750, "■ TabPFN v3", color=slate, fontsize=11)
    fig.text(.785, .750, "■ TabPFN-3.5", color=green, fontsize=11)
    fig.text(.045, .132, f"15 table pairs: {h1['wins']} wins / {h1['losses']} losses / {h1['ties']} ties with added context · one-sided Wilcoxon p = {h1['wilcoxon_p_one_sided']:.4f}", color=muted, fontsize=10)
    fig.text(.045, .092, "Only three selected datasets; seeds share source data. Sorted codes, not semantic text. Package defaults preserved.", color=muted, fontsize=10)
    fig.text(.045, .052, "Recorded TabPFN 9.0.0 CPU study; not a 100k-row test or a benchmark of the product's updated category cap.", color=muted, fontsize=10)
    directory.mkdir(parents=True, exist_ok=True)
    save(fig, directory, "high_cardinality")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/high_cardinality")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    records = load_records(args.root, verify_hashes=args.root.resolve() == (ROOT / "results/high_cardinality").resolve())
    summary = summarize(records)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_values(records, args.root)
    if not args.no_figure:
        render(summary, args.figure_dir)
    print(json.dumps(summary["metrics"]["precision_at_k"], indent=2))


if __name__ == "__main__":
    main()
