"""Benchmark overview: pre-registered confirmation results, read from stored summaries."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2] / "results" / "confirmation"
st.set_page_config(page_title="Benchmark · Chain of Custody", layout="wide")
st.title("Pre-registered confirmation benchmark")
st.write("Gradient search through TabPFN-3.5's learning process vs. query-matched black-box search. "
         "90 audits: 3 public datasets × 6 fresh context seeds × 5 targets, k = 3 appended rows.")
summary_path = ROOT / "paired_summary.json"
if not summary_path.is_file():
    st.info("No confirmation summary found. Run experiments/summarize_paired.py results/confirmation.")
    st.stop()
summary = json.loads(summary_path.read_text())
rows = []
for ds, r in summary["by_dataset"].items():
    p = r["paired_gradient_minus_best_search"]["tabpfn_standard"]
    rates = r["rates"]["tabpfn_standard"]
    rows.append({"Dataset": ds, "Gradient": rates["gradient"], "Random search": rates["random_search"],
                 "Coordinate search": rates["coordinate_search"], "Paired difference": p["diff"],
                 "CI low": p["ci95"][0], "CI high": p["ci95"][1], "CI level %": p.get("ci_level"),
                 "Pre-registered H1": "supported" if p["ci95"][0] > 0 else "not supported"})
frame = pd.DataFrame(rows)
st.subheader("Primary endpoint: verified flips on standard TabPFN-3.5")
st.dataframe(frame, hide_index=True, width="stretch")
st.bar_chart(frame.set_index("Dataset")[["Gradient", "Random search", "Coordinate search"]])
st.caption("Counted over all sampled targets; invalid or aborted searches count as non-flips. "
           "See docs/CONFIRMATION_PREREG.md (written before the run) and docs/RESULTS.md.")
st.subheader("Transfer of the same rows to other models (secondary)")
transfer = []
for ds, r in summary["by_dataset"].items():
    for receiver, p in r["paired_gradient_minus_best_search"].items():
        transfer.append({"Dataset": ds, "Receiver": receiver, "Gradient": r["rates"][receiver]["gradient"],
                         "Best search": r["rates"][receiver][p["baseline"]], "Difference": p["diff"],
                         "CI low": p["ci95"][0], "CI high": p["ci95"][1]})
st.dataframe(pd.DataFrame(transfer), hide_index=True, width="stretch")
det_path = ROOT / "detection" / "summary.json"
if det_path.is_file():
    st.subheader("Detection: can an auditor find the appended rows?")
    det = json.loads(det_path.read_text())
    st.dataframe(pd.DataFrame([
        {"Dataset": k.split("|")[0], "Subset": k.split("|")[1], "n": v["n"],
         **{f"{d.replace('_', ' ')} AUROC": v[d]["auroc"] for d in ("loo_influence", "label_suspicion", "outlier")},
         "LOO top-k recall": v["loo_influence"]["topk_recall"]} for k, v in det.items()
    ]), hide_index=True, width="stretch")
    st.caption("Low outlier AUROC is largely by construction: the search starts near and stays near real rows.")
st.warning("Supported only on continuous numeric data in this benchmark. No robustness certificates, "
           "minimum budgets or domain-realism claims.")
