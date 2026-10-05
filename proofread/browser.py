"""TabLint — spellcheck for tables, powered by TabPFN-3.5 (CSV upload and saved-report viewer).

Run:  uv run --extra demo tablint app
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
from proofread.core import Report

DEMOS = Path(os.environ.get("PROOFREAD_DEMO_DIR", ROOT / "results" / "proofread_demos"))
BENCH = ROOT / "results" / "proofread" / "summary.json"

st.set_page_config(page_title="TabLint · TabPFN-3.5", layout="wide")


@st.cache_data
def load(path: str):
    return Report.load(path)


def range_chart(r) -> alt.Chart:
    lo, hi = float(r["low"]), float(r["high"])
    pts = pd.DataFrame([{"what": "recorded value", "x": float(r["value"])}, {"what": "TabLint suggestion", "x": float(r["suggested"])}])
    band = alt.Chart(pd.DataFrame([{"lo": lo, "hi": hi}])).mark_bar(height=26, opacity=.35, color="#4c78a8").encode(
        x=alt.X("lo:Q", title=str(r["column"])), x2="hi:Q")
    dots = alt.Chart(pts).mark_point(size=260, filled=True).encode(
        x="x:Q", color=alt.Color("what:N", scale=alt.Scale(domain=["recorded value", "TabLint suggestion"], range=["#d62728", "#2ca02c"]),
                                 legend=alt.Legend(orient="bottom", title=None)), tooltip=["what", "x"])
    return (band + dots).properties(height=90, title="TabPFN-3.5: 80% plausible range given the rest of the row")


def main():
    st.title("TabLint")
    st.caption("Spellcheck for tables. TabPFN-3.5 predicts every cell from the rest of its row and underlines values that "
               "fall far outside its predictive distribution — then suggests the likely fix.")
    files = sorted(DEMOS.glob("*.json"))
    tab_check, tab_bench, tab_how = st.tabs(["Check a table", "Benchmark (pre-registered)", "How it works"])
    with tab_check:
        _check_table(files)
    with tab_bench:
        if BENCH.exists():
            b = json.loads(BENCH.read_text())
            st.markdown(b.get("headline", ""))
            st.dataframe(pd.DataFrame(b["cells_table"]), hide_index=True, width="stretch")
            if b.get("labels_table"):
                st.markdown("**Label errors** (10% flipped labels; AUROC)")
                st.dataframe(pd.DataFrame(b["labels_table"]), hide_index=True, width="stretch")
            st.caption("Design and decision rules fixed before running: docs/PROOFREAD_PREREG.md. All 14 datasets are shown.")
        else:
            st.info("Benchmark summary not found (run benchmarks/summarize_proofread.py).")
    with tab_how:
        st.markdown("""
1. **Every cell is a prediction task.** For each numeric column, TabPFN-3.5 is given the other rows (out-of-fold) and
   predicts the column from the rest of the row — without dataset-specific gradient training.
2. **Full predictive distribution.** TabPFN returns a predictive distribution, not just a point estimate, so "surprising"
   adapts per row: a value can be ordinary for the column yet surprising for *this* row.
3. **Surprise score** = −log₁₀ of the two-sided tail probability of the recorded value.
4. **Likely cause.** TabLint tests common slips (×10, ÷10, ×1000, swapped leading digits, sign flip, 0 for missing):
   if the corrected value is typical, it says so.
5. **Context.** Each flag lists the most related columns (rank correlation on this table) and their values in that row,
   so you can see what the prediction was based on. This is context, not a causal attribution.
6. **Categories and labels** are checked the same way with the TabPFN-3.5 classifier (out-of-fold probability of the recorded label).
""")


def _check_table(files):
    with st.sidebar:
        st.header("Table")
        titles = {f: load(str(f)).meta.get("title", f.stem) for f in files}
        options = files + (["__live__"] if "live_report" in st.session_state else [])
        titles["__live__"] = "Your upload"
        chosen = (st.radio("Example", options, index=len(options) - 1 if "live_report" in st.session_state else 0,
                           format_func=lambda f: titles[f]) if options else None)
        with st.expander("Check your own CSV", expanded=not files):
            up = st.file_uploader("CSV file", type=["csv"])
            if up is not None:
                raw = pd.read_csv(up)
                label = st.selectbox("Label column (optional)", ["(none)"] + list(raw.columns))
                cap = st.number_input("Max rows (random sample)", 1, 5000, max(1, min(len(raw), 600)), step=1)
                fast = st.checkbox("TabPFN-3.5-Fast (quicker)", value=False)
                if st.button("Check with TabLint"):
                    from proofread import Proofreader
                    df = raw.sample(int(cap), random_state=0) if len(raw) > cap else raw
                    with st.spinner(f"TabPFN-3.5 is reading {len(df)} rows × {df.shape[1]} columns ..."):
                        r = Proofreader(fast=fast).check(df, label=None if label == "(none)" else label, threshold=2.0, max_issues=80)
                    r.meta["title"] = f"Your upload: {up.name}"
                    st.session_state["live_report"] = r
                    st.rerun()
        top = st.slider("Show top issues", 5, 60, 20)
        st.caption("Examples replay stored TabPFN-3.5 results (no GPU needed). Uploads run TabPFN-3.5 live: "
                   "GPU if available, otherwise CPU (a few minutes for a few hundred rows).")
    if chosen is None:
        st.info("Upload a CSV in the sidebar to check your own table.")
        return
    rep = st.session_state.get("live_report") if chosen == "__live__" else load(str(chosen))
    iss = rep.issues.head(top).copy()
    truth = rep.meta.get("ground_truth")
    tmap = {(t["row"], t["column"]): t for t in truth} if truth else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(rep.data)); c2.metric("Columns checked", rep.meta.get("columns_checked"))
    kinds = rep.issues.get("kind", pd.Series(dtype=str))
    c3.metric("Flagged cells", int(kinds.isin(["cell", "category"]).sum()))
    c4.metric("Flagged labels", int((kinds == "label").sum()))
    if iss.empty:
        st.info("No issues meet the report threshold.")
        st.dataframe(rep.data, width="stretch")
        return
    if truth:
        cells = iss[iss.kind == "cell"]
        hits = sum((r.row, r.column) in tmap for r in cells.itertuples())
        st.success(f"Ground truth known for this demo ({len(truth)} injected errors). Of the top {len(cells)} flagged cells, "
                   f"**{hits} are real injected errors** ({hits / max(len(cells), 1):.0%}).")
    for p in rep.meta.get("patterns", []):
        st.info("Column pattern — " + p["message"])
    st.subheader("Flagged rows")
    rows = sorted(set(iss.row))
    st.dataframe(_styled(rep, iss, rows), width="stretch", height=320)
    st.subheader("Issues")
    show = iss[["kind", "row", "column", "value", "suggested", "surprise", "cause", "evidence"]].copy()
    if truth:
        show["real error?"] = [("✓ " + tmap[(r.row, r.column)]["kind"]) if (r.row, r.column) in tmap else ("—" if r.kind == "cell" else "") for r in iss.itertuples()]
    sel = st.dataframe(show, width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row")
    pick = sel.selection.rows[0] if sel and sel.selection.rows else 0
    r = iss.iloc[pick]
    st.subheader(f"Row {r['row']} · {r['column']}")
    a, b = st.columns([2, 1])
    with a:
        if r["kind"] == "cell":
            st.altair_chart(range_chart(r), width="stretch")
        else:
            st.write(f"Recorded category/label **{r['value']}**; TabPFN-3.5 out-of-fold prediction: **{r['suggested']}** ({r['cause']}).")
    with b:
        st.markdown(f"**Recorded:** {r['value']}  \n**Suggested:** {r['suggested'] if r['kind'] in ('label', 'category') else f'{float(r['suggested']):.4g}'}  \n"
                    f"**Surprise:** {float(r['surprise']):.1f} (−log₁₀ probability)  \n**Likely cause:** {r['cause']}")
        if r.get("evidence"):
            st.markdown(f"**Most related columns in this row:** {r['evidence']}")
        if (r["row"], r["column"]) in tmap:
            t = tmap[(r["row"], r["column"])]
            st.markdown(f"**Ground truth:** injected *{t['kind']}* error; true value {t['true_value']:.4g}")
    st.caption("A flag is a prompt to check the source record, not proof of an error.")


def _styled(rep, iss, rows):
    cells = {(r.row, r.column) for r in iss.itertuples() if r.kind == "cell"}
    labs = {r.row for r in iss.itertuples() if r.kind == "label"}
    cats = {(r.row, r.column) for r in iss.itertuples() if r.kind == "category"}
    sub = rep.data.loc[rows]

    def style(df):
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, c in cells:
            if i in out.index and c in out.columns:
                out.loc[i, c] = "background-color:#ffd6d6;color:#900;font-weight:600"
        for i, c in cats:
            if i in out.index and c in out.columns:
                out.loc[i, c] = "background-color:#ffe8b3;font-weight:600"
        if rep.label in out.columns:
            for i in labs:
                if i in out.index:
                    out.loc[i, rep.label] = "background-color:#ffe8b3;font-weight:600"
        return out
    return sub.style.apply(style, axis=None).format(precision=4)


main()
