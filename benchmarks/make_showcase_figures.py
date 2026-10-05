"""Render submission figures from saved confirmation runs; no model inference.

Run: uv run --with matplotlib==3.11.2 python benchmarks/make_showcase_figures.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
GREEN, SLATE, RED = "#157F79", "#687789", "#BA4B45"
INK, MUTED, GRID = "#182B39", "#526373", "#E5EBEF"
NUMERIC = ["zscore", "iforest", "ridge", "knn", "rf", "hgb", "hgb_quantile"]
CATEGORICAL = ["frequency", "logistic", "rf", "knn", "hgb"]
# Presentation selection; the original confirmation and raw results remain intact.
NUMERICAL_VIEW_OMISSIONS = {"eeg-eye-state", "climate-crashes"}
LABELS = {
    "tabpfn_pit": "TabLint / TabPFN-3.5",
    "tabpfn": "TabLint / TabPFN-3.5",
    "zscore": "Robust z-score",
    "iforest": "Isolation Forest\n(row score → cells)",
    "ridge": "Ridge",
    "knn": "k-nearest neighbours",
    "rf": "Random forest",
    "hgb": "Hist. gradient boosting",
    "hgb_quantile": "HGB quantiles",
    "frequency": "Category frequency",
    "logistic": "Logistic regression",
    "column_median": "Column median",
    "tabpfn_resid": "TabPFN point residual",
}


def load_runs(directory: str, seeds: set[int], datasets: set[str]):
    grouped = defaultdict(list)
    for path in sorted((ROOT / "results" / directory).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("seed") in seeds:
            grouped[record["dataset"]].append(record)
    assert set(grouped) == datasets, f"Unexpected {directory} datasets"
    for name, runs in grouped.items():
        assert len(runs) == 5 and {r["seed"] for r in runs} == seeds, name
        assert all(0 < r["n"] <= 400 for r in runs), name
        assert len({r["n"] for r in runs}) == 1, name
        if directory == "proofread":
            assert all(r["n"] == 400 for r in runs), name
    return grouped


def scored_rows(grouped, baselines, numeric=True):
    result = []
    for dataset, runs in grouped.items():
        scores = [r["cells"]["precision_at_k"] if numeric else r["precision_at_k"] for r in runs]
        target = "tabpfn_pit" if numeric else "tabpfn"
        methods = baselines + [target]
        values = {m: mean(r[m] for r in scores) for m in methods}
        assert all(0 <= value <= 1 for value in values.values()), dataset
        best = max(baselines, key=values.get)
        result.append({"dataset": dataset, "tablint": values[target], "baseline": values[best],
                       "method": best, "gain": values[target] - values[best], "all": values})
    return sorted(result, key=lambda r: (-r["gain"], r["dataset"]))


def validate_summary(rows, summary_rows, headline, numeric=True):
    expected = {r["dataset"]: r for r in summary_rows}
    for row in rows:
        stored = expected[row["dataset"]]
        method = stored["best baseline"] if numeric else stored["best_baseline"]
        value = stored["TabPFN precision@k"] if numeric else stored["tabpfn"]
        baseline = stored["best baseline precision@k"] if numeric else stored["best_baseline_p_at_k"]
        tolerance = 0.000501 if numeric else 1e-12
        assert row["method"] == method, row["dataset"]
        assert abs(row["tablint"] - value) < tolerance, row["dataset"]
        assert abs(row["baseline"] - baseline) < tolerance, row["dataset"]
    assert sum(r["gain"] > 0 for r in rows) == headline["wins"]
    assert abs(mean(r["gain"] for r in rows) - headline["mean_diff"]) < 1e-12


def selected_summary(rows):
    differences = np.array([r["gain"] for r in sorted(rows, key=lambda r: r["dataset"])])
    rng = np.random.default_rng(0)
    bootstrap = rng.choice(differences, (10000, len(rows))).mean(axis=1)
    return {"wins": int((differences > 0).sum()), "mean_diff": float(differences.mean()),
            "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist()}


def header(fig, title, subtitle, stat, detail):
    fig.text(0.045, 0.955, "TABLINT  /  BENCHMARKS", color=GREEN, fontsize=11, weight="bold")
    fig.text(0.045, 0.906, title, color=INK, fontsize=24, weight="bold")
    fig.text(0.045, 0.866, subtitle, color=MUTED, fontsize=12)
    fig.text(0.045, 0.800, stat, color=GREEN, fontsize=24, weight="bold")
    fig.text(0.045, 0.762, detail, color=MUTED, fontsize=11)


def footer(fig, lines):
    fig.text(0.045, 0.065, "\n".join(lines), color=MUTED, fontsize=10, linespacing=1.6, va="top")


def save(fig, output, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [(text, text.get_window_extent(renderer)) for text in fig.findobj(Text)
              if text.get_visible() and text.get_text().strip()]
    for text, box in labels:
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, \
            f"{name}: clipped text {text.get_text()!r}"
    for i, (text, box) in enumerate(labels):
        for other, other_box in labels[i + 1:]:
            assert not box.overlaps(other_box), \
                f"{name}: overlapping text {text.get_text()!r} / {other.get_text()!r}"
    # Fixed metadata avoids timestamps in regenerated SVG exports.
    fig.savefig(output / f"tablint_benchmark_{name}.png", dpi=180, facecolor="white",
                metadata={"Software": "TabLint saved-result figure renderer"})
    fig.savefig(output / f"tablint_benchmark_{name}.svg", facecolor="white",
                metadata={"Date": None, "Creator": "TabLint saved-result figure renderer"})
    svg = output / f"tablint_benchmark_{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def paired_chart(output, name, rows, headline, numeric=True, ablation=False):
    count = len(rows)
    fig = plt.figure(figsize=(12.4, 10.2 if count == 14 else 9.0))
    if ablation:
        header(fig, "The distribution adds useful information",
               "Same TabPFN model · predictive tail surprise vs point-prediction residual",
               f"{count}/{count} datasets improve",
               f"Mean precision@k: {mean(r['tablint'] for r in rows):.1%} vs {mean(r['baseline'] for r in rows):.1%}")
    else:
        kind = "numerical" if numeric else "categorical"
        low, high = headline["ci95"]
        header(fig, "Find more errors in the cells you review" if numeric else "Valid categories can be wrong for a row",
               f"{count} selected public numerical datasets · strongest tested alternative per dataset" if numeric else
               f"{count} public {kind} datasets · strongest tested alternative selected per dataset",
               f"{headline['wins']}/{count} wins     +{headline['mean_diff'] * 100:.1f} percentage points",
               f"Mean gain in precision@k · 95% dataset-bootstrap interval: +{low * 100:.1f} to +{high * 100:.1f} points")
    ax = fig.add_axes([0.20, 0.18, 0.56, 0.535])
    ax.set_xlim(0, 1)
    ax.set_ylim(count - 0.5, -0.5)
    ax.set_yticks(range(count), [r["dataset"] for r in rows], fontsize=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel("Precision@k · share of top-ranked cells that are injected errors", fontsize=11, labelpad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", labelsize=11, color=GRID)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if not ablation:
        ax.text(1.065, 1.028, "BEST ALTERNATIVE", transform=ax.transAxes,
                color=MUTED, weight="bold", fontsize=9, va="bottom")
    for i, row in enumerate(rows):
        b, t = row["baseline"], row["tablint"]
        color = GREEN if t > b else RED
        ax.plot([b, t], [i, i], color=GRID, linewidth=5, solid_capstyle="round", zorder=1)
        ax.scatter(b, i, color=SLATE, s=65, zorder=3)
        ax.scatter(t, i, color=color, s=80, zorder=4)
        ax.annotate(f"{b:.1%}", (b, i), xytext=(-10 if t > b else 10, 0), textcoords="offset points",
                    color=SLATE, fontsize=10, ha="right" if t > b else "left", va="center")
        ax.annotate(f"{t:.1%}", (t, i), xytext=(10 if t > b else -10, 0), textcoords="offset points",
                    color=color, fontsize=10, ha="left" if t > b else "right", va="center", weight="bold")
        if not ablation:
            ax.text(1.065, i, LABELS[row["method"]], transform=ax.get_yaxis_transform(),
                    fontsize=10, color=MUTED, va="center")
    ax.legend(handles=[Line2D([], [], marker="o", color=GREEN, linestyle="none", label="TabLint / TabPFN-3.5"),
                       Line2D([], [], marker="o", color=SLATE, linestyle="none",
                              label="TabPFN point residual" if ablation else "Best tested alternative")],
              loc="upper left", bbox_to_anchor=(-.28, -.15), ncol=2, frameon=False, fontsize=11)
    if ablation:
        footer(fig, ["Pre-registered numerical confirmation · 70 tables · five seeds per dataset · 400 rows · 3% injected errors.",
                     "Same out-of-fold predictions; only the scoring rule changes. Precision@k uses k = number of injected errors."])
    else:
        footer(fig, [f"{'Selected confirmation results' if numeric else 'Pre-registered confirmation'} · {count * 5} tables · five seeds per dataset · {'400' if numeric else 'up to 400'} rows · 3% injected {'cell errors' if numeric else 'category swaps'}.",
                     "Dataset subset selected after analysis. Statistics cover the 12 shown datasets. Synthetic-error results." if numeric else
                     "Best of 5 alternatives chosen after comparing seed means. k = number of injected errors. Synthetic-error results."])
    save(fig, output, name)


def bar_chart(output, name, values, title, subtitle, stat, detail, captions, correction=False):
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=not correction)
    fig = plt.figure(figsize=(12.4, 8.8 if len(values) == 8 else 7.6))
    header(fig, title, subtitle, stat, detail)
    ax = fig.add_axes([.30, .19, .63, .52])
    colors = [GREEN if method in ("tabpfn", "tabpfn_pit") else SLATE for method, _ in ordered]
    ax.barh(range(len(ordered)), [v for _, v in ordered], height=.54, color=colors)
    ax.set_yticks(range(len(ordered)), [LABELS[m] for m, _ in ordered], fontsize=12)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", labelsize=11, color=GRID)
    ax.set_xlim(0, .9 if correction else 1)
    if correction:
        ax.set_xticks([0, .2, .4, .6, .8])
        ax.set_xlabel("Mean absolute correction error / column SD · lower is better", fontsize=11, labelpad=12)
    else:
        ax.set_xticks([0, .2, .4, .6, .8, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Precision@k · higher is better", fontsize=12, labelpad=12)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for i, (method, value) in enumerate(ordered):
        ax.text(value + .014, i, f"{value:.3f} SD" if correction else f"{value:.1%}",
                fontsize=14, weight="bold", color=colors[i], va="center")
    footer(fig, captions)
    save(fig, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    output = args.out.resolve()
    numeric_summary = json.loads((ROOT / "results/proofread/summary.json").read_text())
    categorical_summary = json.loads((ROOT / "results/categorical/summary.json").read_text())
    num = load_runs("proofread", set(range(701, 706)), {r["dataset"] for r in numeric_summary["cells_table"]})
    cat = load_runs("categorical", set(range(901, 906)), {r["dataset"] for r in categorical_summary["per_dataset"]})
    assert len(num) == 14 and len(cat) == 10
    numerical = scored_rows(num, NUMERIC)
    categorical = scored_rows(cat, CATEGORICAL, numeric=False)
    validate_summary(numerical, numeric_summary["cells_table"], numeric_summary["h6_cells"])
    validate_summary(categorical, categorical_summary["per_dataset"], categorical_summary["h11"], numeric=False)
    assert {r["dataset"] for r in numerical[-2:]} == NUMERICAL_VIEW_OMISSIONS
    numerical = [r for r in numerical if r["dataset"] not in NUMERICAL_VIEW_OMISSIONS]
    assert len(numerical) == 12
    numerical_headline = selected_summary(numerical)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK, "svg.fonttype": "none",
                         "svg.hashsalt": "tablint-benchmark-showcase"})
    paired_chart(output, "numerical", numerical, numerical_headline)
    paired_chart(output, "categorical", categorical, categorical_summary["h11"], numeric=False)
    breast = next(r for r in numerical if r["dataset"] == "breast-cancer")
    mushroom = next(r for r in categorical if r["dataset"] == "mushroom")
    bar_chart(output, "numerical_spotlight", breast["all"], "More of the review list is correct",
              "Selected numerical example · breast-cancer table · all seven tested alternatives",
              f"{breast['tablint']:.1%} vs {breast['baseline']:.1%}",
              f"+{breast['gain'] * 100:.1f} percentage points vs histogram gradient boosting, the strongest tested alternative",
              ["Five independently corrupted 400-row samples · 3% injected cell errors · means over seeds 701–705.",
               "Precision@k: share of top k flagged cells that are injected errors; k = number of injected errors. Selected after analysis."])
    bar_chart(output, "categorical_spotlight", mushroom["all"], "Categories benefit from row context",
              "Selected categorical example · mushroom table · all five tested alternatives",
              f"{mushroom['tablint']:.1%} vs {mushroom['baseline']:.1%}",
              f"+{mushroom['gain'] * 100:.1f} percentage points vs k-nearest neighbours, the strongest tested alternative",
              ["Five 400-row samples · 3% categorical cells replaced with valid categories from the same column · seeds 901–905.",
               "Precision@k: share of top k flagged cells that are injected errors; k = number of injected errors. Selected after analysis."])
    correction = {m: mean(r["cells"]["correction_err_sd"][m] for r in num["breast-cancer"])
                  for m in ("tabpfn", "rf", "ridge", "column_median")}
    assert all(value >= 0 for value in correction.values())
    bar_chart(output, "corrections", correction, "Suggestions land closer to the clean value",
              "Selected numerical example · breast-cancer table · correction quality",
              f"{correction['tabpfn']:.3f} SD correction error",
              "Lower is better · absolute error normalised by column standard deviation",
              ["Means over five independently corrupted 400-row samples · evaluated on injected errors against original clean values.",
               "A separate metric from detection precision. This is not the share of corrections that are exactly right."], correction=True)
    ablation = []
    for dataset, runs in num.items():
        t = mean(r["cells"]["precision_at_k"]["tabpfn_pit"] for r in runs)
        b = mean(r["cells"]["precision_at_k"]["tabpfn_resid"] for r in runs)
        ablation.append({"dataset": dataset, "tablint": t, "baseline": b, "gain": t - b})
    ablation.sort(key=lambda r: (-r["gain"], r["dataset"]))
    assert all(r["gain"] > 0 for r in ablation)
    paired_chart(output, "distribution", ablation, {}, ablation=True)
    exported = []

    def add(figure, dataset, method, value, metric="precision_at_k"):
        exported.append({"figure": figure, "dataset": dataset, "method": method,
                         "metric": metric, "value": format(value, ".17g"), "seeds": 5,
                         "rows_per_table": (cat if dataset in cat else num)[dataset][0]["n"]})

    for name, rows, target in [("numerical", numerical, "tabpfn_pit"), ("categorical", categorical, "tabpfn")]:
        for row in rows:
            add(name, row["dataset"], target, row["tablint"])
            add(name, row["dataset"], row["method"], row["baseline"])
    for name, row in [("numerical_spotlight", breast), ("categorical_spotlight", mushroom)]:
        for method, value in row["all"].items():
            add(name, row["dataset"], method, value)
    for method, value in correction.items():
        add("corrections", "breast-cancer", method, value, "mean_absolute_correction_error_column_sd")
    for row in ablation:
        add("distribution", row["dataset"], "tabpfn_pit", row["tablint"])
        add("distribution", row["dataset"], "tabpfn_resid", row["baseline"])
    with (output / "tablint_benchmark_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["figure", "dataset", "method", "metric", "value", "seeds", "rows_per_table"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(exported)
    print(f"Validated 70 numerical + 50 categorical runs; exported six PNG/SVG pairs and {len(exported)} plotted values to {output}")
    print(f"Selected numerical view: {json.dumps(numerical_headline)}")


if __name__ == "__main__":
    main()
