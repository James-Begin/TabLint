"""Notebook integration: interactive widget (anywidget), pandas accessor, IPython magic.

    from proofread import Report
    w = Report.load("data.proofread.json").widget(); w          # interactive: click, accept, dismiss
    w.cleaned, w.decisions                                        # results back in Python

    import proofread                                              # registers df.proofread
    w = df.tablint.view(label="outcome")                         # runs TabPFN-3.5, then shows the widget

    %load_ext proofread
    %tablint df --label outcome
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

STATIC = Path(__file__).parent / "static"


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v if isinstance(v, (int, str, bool)) or v is None else str(v)


def _widget_class():
    import anywidget
    import traitlets

    class ProofreadWidget(anywidget.AnyWidget):
        _esm = STATIC / "widget.js"
        _css = STATIC / "widget.css"
        columns = traitlets.List([]).tag(sync=True)
        rows = traitlets.List([]).tag(sync=True)
        issues = traitlets.List([]).tag(sync=True)
        status = traitlets.List([]).tag(sync=True)
        selected = traitlets.Int(0).tag(sync=True)
        only_flagged = traitlets.Bool(False).tag(sync=True)
        meta = traitlets.Dict({}).tag(sync=True)

        def __init__(self, report, max_rows: int = 2000, **kw):
            self._report = report
            self._max_rows = max_rows
            data = report.data
            iss = report.issues.reset_index(drop=True)
            issues = [{k: _jsonable(v) for k, v in r.items()} for r in iss.to_dict("records")]
            flagged = sorted({int(r["row"]) for r in issues})
            super().__init__(columns=[str(c) for c in data.columns], issues=issues, status=["open"] * len(issues),
                             only_flagged=len(data) > max_rows,
                             meta={"n": int(len(data)), "model": report.meta.get("model", "TabPFN-3.5"),
                                   "patterns": [{k: _jsonable(v) for k, v in p.items()} for p in report.meta.get("patterns", []) or []]},
                             **kw)
            self._flagged = flagged
            self._fill()
            self.observe(lambda _: self._fill(), names="only_flagged")

        def _fill(self):
            d = self._report.data
            idx = self._flagged if self.only_flagged else list(range(min(len(d), self._max_rows)))
            for i in self._flagged:            # flagged rows are always included
                if i not in idx:
                    idx.append(i)
            idx = sorted(set(idx))
            self.rows = [{"i": int(i), "cells": [_jsonable(v) for v in d.iloc[i].tolist()]} for i in idx]

        @property
        def cleaned(self) -> pd.DataFrame:
            """The table with every accepted suggestion applied."""
            from .core import set_cell
            out = self._report.data.copy()
            for r, s in zip(self.issues, self.status):
                if s == "accepted":
                    set_cell(out, int(r["row"]), r["column"], float(r["suggested"]) if r["kind"] == "cell" else r["suggested"])
            return out

        @property
        def decisions(self) -> pd.DataFrame:
            """One row per issue with its current decision (open / accepted / dismissed)."""
            df = pd.DataFrame(self.issues).reindex(columns=["kind", "row", "column", "value", "suggested", "surprise", "cause"])
            df["decision"] = self.status
            return df

    return ProofreadWidget


def widget(report, **kw):
    return _widget_class()(report, **kw)


def report_html(report, top: int = 12, max_rows: int = 8) -> str:
    """Static HTML (no JS, no jinja2): flagged rows with highlighted cells + top issues. Renders in nbviewer/GitHub."""
    import html as H
    esc = lambda v: H.escape(f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v))
    iss = report.issues.head(top)
    cells = {(int(r.row), str(r.column)): r for r in iss.itertuples()}
    rows = list(dict.fromkeys(int(r.row) for r in iss.itertuples()))[:max_rows]
    cols = [str(c) for c in report.data.columns]
    shown = [c for c in cols if any((i, c) in cells for i in rows)] or cols[:8]
    context = [c for c in cols if c not in shown][: max(0, 8 - len(shown))]
    show = [c for c in cols if c in shown or c in context]
    th = "".join(f"<th>{H.escape(c)}</th>" for c in ["row"] + show)
    trs = []
    for i in rows:
        tds = [f"<td style='color:#8c959f'>{i}</td>"]
        for c in show:
            r = cells.get((i, c))
            style = ("background:#ffd6d6;color:#8b0000;font-weight:700;text-decoration:underline wavy #d00" if r is not None and r.kind == "cell"
                     else "background:#ffe8b3;font-weight:700" if r is not None else "")
            tip = f" title='{H.escape(r.cause)}'" if r is not None else ""
            tds.append(f"<td style='text-align:right;{style}'{tip}>{esc(report.data.at[i, c])}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    li = []
    for r in iss.itertuples():
        sug = esc(r.suggested) if r.kind != "cell" else f"{float(r.suggested):.4g}"
        rng = "" if r.kind != "cell" else f" (80%: {float(r.low):.4g}–{float(r.high):.4g})"
        li.append(f"<li><b>row {int(r.row)} · {H.escape(str(r.column))}</b> = <span style='color:#cf222e'>{esc(r.value)}</span> "
                  f"→ TabPFN-3.5 expects <span style='color:#1a7f37'>{sug}</span>{rng} · {H.escape(r.cause)}</li>")
    pats = "".join(f"<div style='color:#9a6700'>⚑ {H.escape(p['message'])}</div>" for p in report.meta.get("patterns", []) or [])
    n_cell = int((report.issues.kind == "cell").sum()); n_lab = int((report.issues.kind == "label").sum())
    n_cat = int((report.issues.kind == "category").sum())
    return (f"<div style='font-family:ui-monospace,Menlo,monospace;font-size:12px'>"
            f"<div><b>TabLint</b> · {len(report.data)} rows · {n_cell} suspicious numeric cells · {n_cat} suspicious categories · {n_lab} suspicious labels · "
            f"{H.escape(str(report.meta.get('model', 'TabPFN-3.5')))}</div>{pats}"
            f"<table style='border-collapse:collapse;margin:6px 0'><tr>{th}</tr>{''.join(trs)}</table>"
            f"<ol style='margin:4px 0 0 18px;padding:0'>{''.join(li)}</ol>"
            f"<div style='color:#8c959f'>Showing the top {len(iss)} issues. For the interactive view, use <code>report.widget()</code>. "
            f"A flag is a prompt to check the source, not proof of an error.</div></div>")


@pd.api.extensions.register_dataframe_accessor("tablint")
@pd.api.extensions.register_dataframe_accessor("proofread")
class ProofreadAccessor:
    """`df.tablint.check(...)` → Report; `df.tablint.view(...)` → widget. `df.proofread` stays compatible."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def check(self, label: str | None = None, **kw):
        from .core import Proofreader
        pr_kw = {k: kw.pop(k) for k in ("device", "folds", "seed", "fast") if k in kw}
        return Proofreader(**pr_kw).check(self._df, label=label, **kw)

    def view(self, label: str | None = None, report=None, **kw):
        return widget(report if report is not None else self.check(label=label, **kw))


def load_ipython_extension(ipython):
    """`%load_ext proofread` registers `%proofread <dataframe> [--label COL] [--threshold T] [--fast]`."""
    import argparse
    import shlex
    from IPython.display import display

    def proofread_magic(line):
        p = argparse.ArgumentParser(prog="%proofread", add_help=False)
        p.add_argument("df"); p.add_argument("--label"); p.add_argument("--threshold", type=float, default=2.0)
        p.add_argument("--fast", action="store_true")
        a = p.parse_args(shlex.split(line))
        df = ipython.user_ns[a.df]
        w = df.proofread.view(label=a.label, threshold=a.threshold, fast=a.fast)
        display(w)
        return w

    ipython.register_magic_function(proofread_magic, magic_kind="line", magic_name="tablint")
    ipython.register_magic_function(proofread_magic, magic_kind="line", magic_name="proofread")
