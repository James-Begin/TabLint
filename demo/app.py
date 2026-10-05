"""Local, replay-only robustness audit viewer; never imports an inference engine."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
SURROGATE = "tabpfn_3_5_differentiable_surrogate"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "results" / "reports"
MAX_REPORT_BYTES = 16 * 1024 * 1024


def sanitize(value: Any) -> Any:
    """Convert non-finite JSON numbers to null, recursively (no imputation)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (OverflowError, ValueError):
        return None


def probability(value: Any) -> float | None:
    result = number(value)
    return result if result is not None and 0 <= result <= 1 else None


def status(value: Any) -> str:
    return "Yes" if value is True else "No" if value is False else "Not reported"


def mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def records(value: Any) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def text(value: Any) -> str:
    """Plain display text, including nested values without HTML rendering."""
    if value is None:
        return "Not reported"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def load_report(path: str | Path) -> tuple[dict, list[str]]:
    """Read a schema-1.0 JSON object, returning sanitized data and quality notes.

    Raises ValueError for unsupported schema/shape/size; filesystem and JSON
    errors propagate to the caller. Missing optional data is never invented.
    """
    with Path(path).open("rb") as handle:
        payload = handle.read(MAX_REPORT_BYTES + 1)
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError("Report exceeds the local viewer's 16 MiB size limit.")
    raw = json.loads(payload)
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError('Expected a JSON object with schema_version="1.0".')
    report = sanitize(raw)
    notes = []
    if report != raw:
        notes.append("Non-finite values were replaced with null; they are not plotted or imputed.")
    for field, kind in (("config", dict), ("baseline", dict), ("attacks", list),
                        ("warnings", list), ("diagnostics", dict)):
        if not isinstance(report.get(field), kind):
            notes.append(f"Missing or malformed {field}; unavailable data is shown explicitly.")
    baseline = mapping(report.get("baseline"))
    if baseline.get("model") != SURROGATE:
        notes.append("Baseline model does not match the contracted differentiable surrogate; check provenance.")
    if probability(baseline.get("p_positive")) is None:
        notes.append("Baseline positive-class probability is missing or outside [0, 1].")
    if type(baseline.get("predicted_class")) is not int:
        notes.append("Baseline predicted_class is missing or not an integer.")
    attacks = report.get("attacks")
    if isinstance(attacks, list) and len(records(attacks)) != len(attacks):
        notes.append("Non-object attack entries were omitted from the viewer; retained in the download.")
    for index, attack in enumerate(records(attacks)):
        if any(probability(attack.get(key)) is None for key in ("p_before", "p_after")):
            notes.append(f"Attack entry {index + 1} has missing or invalid probabilities; no substitution is made.")
        rows = attack.get("rows")
        labels = attack.get("labels")
        if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
            notes.append(f"Attack entry {index + 1} has missing or malformed raw rows.")
        elif type(attack.get("n_rows")) is int and attack["n_rows"] != len(rows):
            notes.append(f"Attack entry {index + 1}: reported n_rows differs from stored row count.")
        if not isinstance(labels, list) or not all(type(label) is int for label in labels):
            notes.append(f"Attack entry {index + 1} has missing or malformed labels.")
        elif isinstance(rows, list) and len(labels) != len(rows):
            notes.append(f"Attack entry {index + 1}: label count differs from stored row count.")
    return report, notes


def summarize_report(report: dict) -> list[dict]:
    """One table record per attack; reported flags are not threshold-recomputed."""
    result = []
    for index, attack in enumerate(records(report.get("attacks"))):
        before, after = probability(attack.get("p_before")), probability(attack.get("p_after"))
        result.append({
            "Entry": index + 1,
            "Method": text(attack.get("method")),
            "Rows (reported)": number(attack.get("n_rows")),
            "P(positive) before": before,
            "P(positive) after": after,
            "Change": after - before if before is not None and after is not None else None,
            "Flipped (reported)": status(attack.get("flipped")),
            "Valid (reported)": status(attack.get("valid")),
            "Evaluations": number(attack.get("evaluations")),
            "Elapsed (s)": number(attack.get("elapsed_s")),
        })
    return result


def discover_reports(directory: Path) -> list[Path]:
    """Only direct local JSON files; reject symlinks escaping this directory."""
    root = directory.resolve()
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob("*.json")
         if path.is_file() and path.resolve().parent == root),
        key=lambda path: path.name,
    )


def report_directory(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default=os.environ.get("DECISION_STRESS_REPORTS_DIR", str(DEFAULT_REPORT_DIR)))
    args = parser.parse_args(argv)
    return Path(args.reports_dir).expanduser().resolve()


def show_empty(st: Any, directory: Path) -> None:
    st.info("No audit reports found. This showcase replays measured artifacts only; it does not run inference.")
    st.write("Report directory:")
    st.code(str(directory), language=None)
    st.subheader("Generate artifacts separately")
    st.write(
        "Run your audit producer in its inference environment, then export each completed audit "
        'as a .json object with schema_version="1.0" into this directory. Include config, baseline, '
        "attacks, warnings, and diagnostics. See demo/README.md for the artifact contract. "
        "Legacy JSONL results are not this contract and must be exported by the producer, not renamed."
    )
    st.write("Return here and refresh after reports are written. No weights, GPU, uploads, or credentials are needed by the viewer.")


def show_rows(st: Any, pd: Any, report: dict, attack: dict) -> None:
    rows = attack.get("rows")
    if not isinstance(rows, list) or not rows:
        st.info("No raw synthetic rows stored for this attack.")
        return
    if not all(isinstance(row, list) for row in rows):
        st.warning("Malformed raw rows; inspect the JSON download instead.")
        return
    labels = attack.get("labels") if isinstance(attack.get("labels"), list) else []
    row_index = st.selectbox("Synthetic row", range(len(rows)), format_func=lambda i: f"Row {i + 1}")
    st.write("Stored row label:", text(labels[row_index] if row_index < len(labels) else None))
    target = report.get("target") if isinstance(report.get("target"), list) else []
    names = report.get("feature_names") if isinstance(report.get("feature_names"), list) else []
    row = rows[row_index]
    width = max(len(row), len(target), len(names))
    data = []
    for index in range(width):
        left = target[index] if index < len(target) else None
        right = row[index] if index < len(row) else None
        comparable = index < len(target) and index < len(row) and left is not None and right is not None
        a, b = number(left), number(right)
        delta = number(b - a) if a is not None and b is not None else None
        data.append({
            "Feature": text(names[index]) if index < len(names) else f"Feature {index + 1}",
            "Target": text(left), "Synthetic": text(right), "Numeric difference": delta,
            "Comparison": "Changed" if comparable and left != right else "Same" if comparable else "Unavailable",
        })
    if not data:
        st.info("The stored row has no features.")
        return
    frame = pd.DataFrame(data)

    def highlight(row_values: Any) -> list[str]:
        return ["background-color: #fff1b8; color: #202020" if column == "Synthetic" and row_values["Comparison"] == "Changed" else "" for column in row_values.index]

    st.dataframe(frame.style.apply(highlight, axis=1), hide_index=True, width="stretch")
    st.caption("Highlighted cells differ from the stored target. Values are raw artifact values; feature units, encodings, and realism are not inferred.")


def show_history(st: Any, pd: Any, attack: dict) -> None:
    history = records(attack.get("optimization_history"))
    if not history:
        st.info("No optimization history recorded for this attack.")
        return
    # Plot only actual scalar numeric fields; never interpolate or invent a loss.
    keys = sorted({key for item in history for key, value in item.items() if number(value) is not None})
    if not keys:
        st.info("Stored history has no finite numeric series.")
        st.json(history)
        return
    selected = st.selectbox("History series", keys)
    st.line_chart(pd.DataFrame({selected: [number(item.get(selected)) for item in history]}))
    st.caption("Horizontal axis: stored history entry order, not assumed optimization step. Gaps are missing values; series units are producer-defined.")
    with st.expander("Stored history entries"):
        st.json(history)


def main() -> None:
    # Lazy UI imports keep artifact loading/summarization independently testable.
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Chain of Custody", layout="wide")
    st.title("Chain of Custody")
    st.write("TabPFN learns from rows; we differentiate through that learning to stress-test a prediction.")
    st.caption("TabPFN-3.5 robustness audit · local artifact replay · no inference in this interface")
    st.warning(
        "No counterexample found ≠ a robustness certificate. Range and nearest-neighbor bounds ≠ domain realism. "
        "This audit makes no causal, fairness, or legal claim. Reference duplicate-row attacks are unconstrained comparisons, not evidence of feasible real-world changes."
    )
    default_dir = report_directory()
    root = Path(__file__).resolve().parents[1] / "results"
    choices = {"Curated examples": default_dir}
    if (root / "confirmation").is_dir():
        choices["Pre-registered confirmation (all 90, uncurated)"] = root / "confirmation"
    if (root / "development_stable").is_dir():
        choices["Development runs (stable gradients)"] = root / "development_stable"
    with st.sidebar:
        st.header("Local replay")
        label = st.radio("Artifact set", list(choices))
        directory = choices[label]
        if "uncurated" in label:
            st.caption("Every sampled target is included, successes and failures alike.")
        st.write("Reports directory")
        st.code(str(directory), language=None)
        st.button("Refresh reports")
        st.caption("Read-only local JSON. No file uploads, secrets, model imports, or weight loading.")
    try:
        files = [f for f in discover_reports(directory)
                 if f.name not in ("protocol.json", "manifest.json", "paired_summary.json")]
    except OSError as exc:
        st.error(f"Cannot read report directory: {exc}")
        return
    if not files:
        show_empty(st, directory)
        return
    chosen = st.sidebar.selectbox("Audit report", files, format_func=lambda path: path.name)
    try:
        report, notes = load_report(chosen)
    except (OSError, ValueError, RecursionError) as exc:
        st.error(f"Cannot replay this report: {exc}")
        st.info("Select another report or ask the producer to export a valid schema-1.0 artifact.")
        return
    st.sidebar.download_button(
        "Download report JSON", data=json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        file_name=chosen.name, mime="application/json",
    )
    st.sidebar.caption("Download is the loaded report, with non-finite numbers normalized to null.")
    for note in notes:
        st.warning(note)
    for warning in report.get("warnings", []) if isinstance(report.get("warnings"), list) else []:
        st.warning(text(warning))

    baseline = mapping(report.get("baseline"))
    st.subheader("Measured surrogate outcome")
    st.write("Baseline model:", text(baseline.get("model")))
    st.caption("The differentiable surrogate is not the deployed predictor. Probabilities and reported decision flips below belong to the surrogate audit; transfer verification is separate.")
    summary = summarize_report(report)
    if not summary:
        st.info("No attack entries recorded; a baseline alone does not establish robustness.")
        st.metric("Baseline P(positive)", f"{probability(baseline.get('p_positive')):.4f}" if probability(baseline.get("p_positive")) is not None else "Not reported")
    else:
        table = pd.DataFrame(summary)
        st.dataframe(table, hide_index=True, width="stretch")
        chart = table.set_index(table["Entry"].astype(str) + " · " + table["Method"])[["P(positive) before", "P(positive) after"]]
        if chart.notna().any().any():
            st.bar_chart(chart)
        st.caption("Each bar is a stored positive-class probability; invalid or missing probabilities are omitted, not filled. Entry numbers distinguish repeated methods/budgets.")
        attacks = records(report.get("attacks"))
        index = st.selectbox("Inspect an attack", range(len(attacks)), format_func=lambda i: f"Entry {i + 1} · {text(attacks[i].get('method'))} · rows {text(attacks[i].get('n_rows'))}")
        attack = attacks[index]
        before, after = probability(attack.get("p_before")), probability(attack.get("p_after"))
        columns = st.columns(4)
        columns[0].metric("P(positive) before", f"{before:.4f}" if before is not None else "Not reported")
        columns[1].metric("P(positive) after", f"{after:.4f}" if after is not None else "Not reported")
        columns[2].metric("Decision flipped (reported)", status(attack.get("flipped")))
        columns[3].metric("Attack valid (reported)", status(attack.get("valid")))
        if attack.get("aborted"):
            st.error("Optimizer aborted. The displayed rows are a retained incumbent, not a completed gradient search.")
        if attack.get("unconstrained"):
            st.info("Unconstrained reference: not comparable to the bounded attack methods.")
        st.write("Baseline predicted class:", text(baseline.get("predicted_class")))
        st.caption("Flip flags are replayed from the producer, not inferred from rounded probabilities. The viewer does not change a threshold; decision-rule provenance belongs in config. A reported flip with invalid or unknown validity is not an established feasible counterexample.")
        values_tab, constraints_tab, history_tab = st.tabs(["Synthetic rows", "Constraints & effort", "Optimization trace"])
        with values_tab:
            show_rows(st, pd, report, attack)
        with constraints_tab:
            constraints = mapping(attack.get("constraints"))
            st.dataframe(pd.DataFrame([
                {"Check": key, "Reported status": status(constraints.get(key))}
                for key in ("category_valid", "in_bounds", "protected_unchanged")
            ]), hide_index=True, width="stretch")
            st.write("Reported target-distance summary (dtarget_min):", text(constraints.get("dtarget_min")))
            st.write("Maximum observed nearest-neighbor distance (dnn_max):", text(constraints.get("dnn_max")))
            st.caption("These are reported distance summaries, not an optimality proof for the attack budget. Distance units, bound definitions, and applicability are producer-defined in config; no missing check counts as passing.")
            st.write("Evaluations:", text(attack.get("evaluations")))
            st.write("Elapsed seconds:", text(attack.get("elapsed_s")))
            st.write("Masked gradient entries:", text(attack.get("masked_gradient_entries")))
            with st.expander("Full stored constraints"):
                st.json(constraints)
        with history_tab:
            show_history(st, pd, attack)

    verification = records(report.get("verification"))
    if not verification:
        st.info("No model-transfer verification stored. Surrogate results do not establish a deployed-model decision flip.")
    else:
        st.subheader("Model-transfer verification · separate from surrogate search")
        st.caption("Model identifiers below are exactly those recorded by the producer. Different models and pipelines may disagree; a model name alone does not authenticate a deployed checkpoint.")
        st.dataframe(pd.DataFrame([
            {"Attack method": text(item.get("attack_method")), "Model (reported)": text(item.get("model")),
             "P(positive) before": probability(item.get("p_before")), "P(positive) after": probability(item.get("p_after")),
             "Flipped (reported)": status(item.get("flipped"))} for item in verification
        ]), hide_index=True, width="stretch")
        for index, item in enumerate(verification):
            with st.expander(f"Verification {index + 1}: recorded repeat controls and provenance"):
                st.json(item)
        st.caption("Untouched repeats and max_abs_repeat_delta are reported controls, not automatic pass/fail tests. Missing controls leave repeatability unassessed; no tolerance is assumed by this viewer.")

    detection_file = chosen.parent / "detection" / chosen.name
    if detection_file.is_file():
        try:
            detection = json.loads(detection_file.read_text())
        except (OSError, ValueError):
            detection = {}
        if isinstance(detection, dict) and detection:
            st.subheader("Could an auditor find the appended rows?")
            st.caption("Each detector scores every row of the augmented context; AUROC 0.5 = chance. Hits = appended rows ranked in the top k.")
            st.dataframe(pd.DataFrame([
                {"Detector": name.replace("_", " "), "AUROC": number(detection.get(f"{name}_auroc")),
                 "Appended rows in top k": number(detection.get(f"{name}_hits_in_topk")), "k": number(detection.get("k"))}
                for name in ("loo_influence", "label_suspicion", "outlier")
            ]), hide_index=True, width="stretch")

    influence = mapping(report.get("influence"))
    if influence:
        st.subheader("Training-row sensitivity · exact leave-one-out")
        st.caption(text(influence.get("caveat")))
        measured = records(influence.get("rows"))
        if measured:
            frame = pd.DataFrame(measured)
            if "delta_p_positive" in frame:
                frame["Absolute effect"] = frame["delta_p_positive"].abs()
                frame = frame.sort_values("Absolute effect", ascending=False)
            st.dataframe(frame, hide_index=True, width="stretch")
        st.write("All context rows evaluated:", status(influence.get("exact_over_all_context_rows")))
        st.write("Elapsed seconds:", text(influence.get("elapsed_s")))

    with st.expander("Audit provenance, diagnostics & stored context"): 
        for key in ("config", "dataset", "dataset_metadata", "diagnostics", "context_preview"):
            if key in report:
                st.write(key)
                st.json(report[key])
        st.caption("Stored context may contain sensitive data. Use authorized, de-identified artifacts; this interface does not redact downloads or authenticate viewers.")


if __name__ == "__main__":
    main()
