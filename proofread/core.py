"""Cell- and label-error detection with TabPFN-3.5 conditional predictive distributions."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd

ISSUE_COLUMNS = ["kind", "row", "column", "value", "suggested", "low", "high", "surprise", "cause", "evidence"]

SLIPS = {
    "decimal slip (×10)": lambda v: v / 10,
    "decimal slip (÷10)": lambda v: v * 10,
    "extra zeros (×1000)": lambda v: v / 1000,
    "missing value recorded as 0": None,   # handled specially
    "swapped leading digits": "transpose",
    "sign flip": lambda v: -v,
}


def _transpose(v):
    s = f"{abs(v):.6e}"; m, e = s.split("e"); dg = m.replace(".", "")
    if len(dg) < 2 or dg[0] == dg[1]:
        return None
    dg = dg[1] + dg[0] + dg[2:]
    return float(("-" if v < 0 else "") + dg[0] + "." + dg[1:] + "e" + e)


@dataclass
class Report:
    data: pd.DataFrame
    issues: pd.DataFrame
    cell_surprise: pd.DataFrame
    label: str | None
    meta: dict = field(default_factory=dict)

    def highlight(self, top: int | None = None):
        flagged = self.issues if top is None else self.issues.head(top)
        cells = {(r.row, r.column) for r in flagged.itertuples() if r.kind == "cell"}
        cats = {(r.row, r.column) for r in flagged.itertuples() if r.kind == "category"}
        labels = {r.row for r in flagged.itertuples() if r.kind == "label"}

        def style(df):
            out = pd.DataFrame("", index=df.index, columns=df.columns)
            for (i, c) in cells:
                if i in out.index and c in out.columns:
                    out.loc[i, c] = "background-color:#ffd6d6;text-decoration:underline wavy #d00"
            for (i, c) in cats:
                if i in out.index and c in out.columns:
                    out.loc[i, c] = "background-color:#ffe8b3;text-decoration:underline wavy #c80"
            if self.label in out.columns:
                for i in labels:
                    if i in out.index:
                        out.loc[i, self.label] = "background-color:#ffe8b3;text-decoration:underline wavy #c80"
            return out
        return self.data.style.apply(style, axis=None)

    def _repr_html_(self):
        from .notebook import report_html
        return report_html(self)

    def widget(self, **kw):
        """Interactive notebook widget (JupyterLab, Notebook 7, VS Code, Colab). Needs `anywidget`."""
        from .notebook import widget
        return widget(self, **kw)

    def save(self, path):
        """Write a self-contained JSON report (data, issues, per-cell surprise, metadata)."""
        import json
        from pathlib import Path
        obj = {"schema": "proofread/1", "label": self.label, "meta": self.meta,
               "data": json.loads(self.data.to_json(orient="split")),
               "issues": json.loads(self.issues.to_json(orient="records")),
               "cell_surprise": json.loads(self.cell_surprise.round(4).to_json(orient="split"))}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, default=str))
        return path

    @classmethod
    def load(cls, path):
        import io, json
        from pathlib import Path
        obj = json.loads(Path(path).read_text())
        if obj.get("schema") != "proofread/1":
            raise ValueError("not a proofread report")
        data = pd.read_json(io.StringIO(json.dumps(obj["data"])), orient="split")
        surprise = pd.read_json(io.StringIO(json.dumps(obj["cell_surprise"])), orient="split")
        issues = pd.DataFrame(obj["issues"]).reindex(columns=ISSUE_COLUMNS)
        return cls(data, issues, surprise, obj["label"], obj["meta"])

    def to_markdown(self, top: int = 15) -> str:
        lines = ["# Proofread report", "", f"Rows: {len(self.data)}, columns checked: {self.meta.get('columns_checked')}, "
                 f"label: {self.label}. Model: {self.meta.get('model', 'TabPFN-3.5')} (out-of-fold, {self.meta.get('folds')} folds).", "",
                 "| # | Row | Column | Value | Suggested | 80% plausible range | Surprise | Likely cause |", "|---:|---:|---|---:|---:|---|---:|---|"]
        for k, r in enumerate(self.issues.head(top).itertuples(), 1):
            rng = "" if r.kind != "cell" else f"{r.low:.4g} – {r.high:.4g}"
            sug = r.suggested if r.kind != "cell" else f"{r.suggested:.4g}"
            val = f"{r.value:.6g}" if isinstance(r.value, (int, float, np.floating)) else r.value
            lines.append(f"| {k} | {r.row} | {r.column} | {val} | {sug} | {rng} | {r.surprise:.1f} | {r.cause} |")
        lines += ["", "Surprise = −log₁₀ of the two-sided tail probability under TabPFN's predictive distribution "
                  "given the rest of the row (labels: −log₁₀ P(recorded label)). A flag is a prompt to check the source, "
                  "not proof of an error."]
        if self.meta.get("patterns"):
            lines += ["", "## Column-level patterns"] + [f"- {p['message']}" for p in self.meta["patterns"]]
        return "\n".join(lines)


class Proofreader:
    """Check cells and labels with TabPFN-3.5, retaining categorical row context."""

    def __init__(self, device: str = "auto", folds: int = 5, seed: int = 0, min_distinct: int = 11, fast: bool = False):
        """device: "auto" (CUDA if available, else CPU), "cpu", "cuda:N" or "mps".
        fast: use the TabPFN-3.5-Fast checkpoint (see the measured trade-off in docs/VERSION_COMPARISON.md)."""
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device, self.folds, self.seed, self.min_distinct, self.fast = device, folds, seed, min_distinct, fast

    def _version(self):
        from tabpfn.constants import ModelVersion
        return ModelVersion.V3_5_FAST if self.fast else ModelVersion.V3_5

    # -- models ---------------------------------------------------------------
    def _reg(self, cat_idx=None, *, max_categories=30):
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor.create_default_for_version(self._version(), device=self.device, random_state=self.seed,
                                                          ignore_pretraining_limits=True, categorical_features_indices=cat_idx or None,
                                                          inference_config={"MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": max_categories})

    def _clf(self, cat_idx=None, *, max_categories=30):
        from tabpfn import TabPFNClassifier
        # 4 estimators: the configuration measured in the pre-registered label benchmark (H7).
        return TabPFNClassifier.create_default_for_version(self._version(), device=self.device, n_estimators=4,
                                                           random_state=self.seed, ignore_pretraining_limits=True,
                                                           categorical_features_indices=cat_idx or None,
                                                           inference_config={"MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": max_categories})

    # -- main entry -----------------------------------------------------------
    def check(self, df: pd.DataFrame, label: str | None = None, columns=None, max_issues: int = 50,
              threshold: float = 2.0, categorical: bool = True) -> Report:
        """Return a Report. ``threshold`` = minimum surprise (−log10 two-sided tail probability) to list a cell.
        ``columns`` selects check targets, not input features. High-cardinality string/category columns remain
        categorical context even when ``categorical=False`` disables categorical-cell checks.
        Benchmark (docs/PROOFREAD_RESULTS.md, addendum A2): retained scope: 2 → 87% precision / 57% recall; 3 → 94% / 20%."""
        import torch
        from sklearn.model_selection import KFold, StratifiedKFold
        t0 = time.time()
        df = df.reset_index(drop=True)
        selected = list(df.columns if columns is None else columns)
        num = [c for c in df.columns if c != label and pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c])]
        cont = [c for c in num if c in selected and df[c].nunique(dropna=True) >= self.min_distinct]
        X = df[num].to_numpy(np.float64)
        # A category can be useful context even when it has too many levels to check as a target.
        text_cats = self.categorical_context_columns(df, label)
        C = np.column_stack([self._categorical_codes(df[c]) for c in text_cats]) if text_cats else np.zeros((len(df), 0))
        numeric_cats = set(self.categorical_columns(df, label)) & set(num)
        y_codes = None
        if label is not None:
            y_codes, y_levels = pd.factorize(df[label])
            y_context = self._categorical_codes(df[label])
        n = len(df)
        surprise = pd.DataFrame(0.0, index=df.index, columns=cont)
        stats = {}; q20 = {}; q80 = {}
        kf = KFold(self.folds, shuffle=True, random_state=self.seed)
        for c in cont:
            j = num.index(c)
            obs = X[:, j]
            ok = np.isfinite(obs)
            Z = np.c_[np.delete(X, j, axis=1), C]
            remaining_num = [x for x in num if x != c]
            reg_cat = [i for i, name in enumerate(remaining_num) if name in numeric_cats]
            reg_cat += list(range(len(remaining_num), Z.shape[1]))
            if y_codes is not None:
                reg_cat.append(Z.shape[1])
                Z = np.c_[Z, y_context]
            if Z.shape[1] == 0:
                continue                                   # nothing else in the row to predict this column from
            med, lo, hi, pit = (np.full(n, np.nan) for _ in range(4))
            rows = np.where(ok)[0]
            for tr, te in kf.split(rows):
                tr, te = rows[tr], rows[te]
                # TabPFN applies a separate cap to numeric category codes, even when explicitly declared.
                # A feature cannot have more distinct values (including missing) than this table has rows.
                out = self._reg(reg_cat, max_categories=max(30, n)).fit(Z[tr], obs[tr]).predict(Z[te], output_type="full")
                lg = out["logits"]
                yy = torch.tensor(obs[te], dtype=lg.dtype, device=lg.device)
                F = out["criterion"].cdf(lg, yy.unsqueeze(-1)).squeeze(-1).detach().double().cpu().numpy().reshape(-1)
                pit[te] = np.clip(np.nan_to_num(F, nan=.5), 1e-12, 1 - 1e-12)
                med[te] = out["median"]
                q = out["quantiles"]  # default quantiles 0.1, 0.2, ..., 0.9
                lo[te], hi[te] = q[0], q[-1]
                q20.setdefault(c, np.full(n, np.nan))[te] = q[1]
                q80.setdefault(c, np.full(n, np.nan))[te] = q[-2]
                stats.setdefault(c, {})["criterion"] = out["criterion"]
            s = -np.log10(2 * np.minimum(pit, 1 - pit))
            surprise[c] = np.nan_to_num(s, nan=0.0)
            stats[c].update(median=med, low=lo, high=hi, pit=pit, q20=q20[c], q80=q80[c])
        cont = [c for c in cont if c in stats]
        integer_col = {c: bool(pd.api.types.is_integer_dtype(df[c]) or
                               (df[c].dropna() == np.round(df[c].dropna())).all()) for c in cont}
        related = df[num].corr(method="spearman").abs() if len(num) > 1 else None
        patterns = self._patterns(df, num)
        pattern_cols = {p["column"] for p in patterns}
        recs = []
        for c in cont:
            for i in np.where(surprise[c].to_numpy() >= threshold)[0]:
                v = float(df.at[i, c]); st = stats[c]
                sug = float(st["median"][i])
                if integer_col[c]:
                    sug = float(round(sug))                      # integer column: suggest a whole number
                recs.append(dict(kind="cell", row=int(i), column=c, value=v, suggested=sug,
                                 low=float(st["low"][i]), high=float(st["high"][i]), surprise=float(surprise.at[i, c]),
                                 cause=self._cause(v, st["low"][i], st["high"][i], st["median"][i], c in pattern_cols),
                                 evidence=self._context(df, related, c, i)))
        XL = np.c_[X, C]
        if y_codes is not None and len(set(y_codes)) > 1 and XL.shape[1] > 0:
            P = np.zeros((n, len(y_levels)))
            lab_cat = [i for i, name in enumerate(num) if name in numeric_cats]
            lab_cat += list(range(X.shape[1], XL.shape[1]))
            for tr, te in StratifiedKFold(self.folds, shuffle=True, random_state=self.seed).split(XL, y_codes):
                m = self._clf(lab_cat, max_categories=max(30, n)).fit(XL[tr], y_codes[tr])
                P[np.ix_(te, m.classes_)] = m.predict_proba(XL[te])
            p_given = P[np.arange(n), y_codes]
            ls = -np.log10(np.clip(p_given, 1e-12, 1))
            for i in np.where(ls >= max(1.0, threshold - 2))[0]:
                recs.append(dict(kind="label", row=int(i), column=label, value=df.at[i, label],
                                 suggested=y_levels[int(P[i].argmax())], low=np.nan, high=np.nan, surprise=float(ls[i]),
                                 cause=f"model gives the recorded label {p_given[i]:.1%}", evidence=""))
        cat_cols = self.categorical_columns(df, label, selected) if categorical else []
        if cat_cols:
            for c, info in self.categorical_scores(df, cat_cols, label).items():
                for i in np.where(info["surprise"] >= threshold)[0]:
                    recs.append(dict(kind="category", row=int(i), column=c, value=df.at[i, c], suggested=info["suggested"][i],
                                     low=np.nan, high=np.nan, surprise=float(info["surprise"][i]),
                                     cause=self._category_cause(df[c], df.at[i, c], info["p_recorded"][i], info["suggested"][i], info["p_top"][i]),
                                     evidence=""))
        issues = pd.DataFrame(recs, columns=ISSUE_COLUMNS)
        issues = issues.sort_values("surprise", ascending=False).head(max_issues).reset_index(drop=True)
        meta = {"model": "TabPFN-3.5-Fast" if self.fast else "TabPFN-3.5", "device": self.device,
                "patterns": patterns, "columns_checked": len(cont), "categorical_columns_checked": len(cat_cols),
                "categorical_context_columns": [c for c in df.columns if c in numeric_cats or c in text_cats],
                "folds": self.folds, "seconds": time.time() - t0, "n": n, "threshold": threshold}
        return Report(df, issues, surprise, label, meta)

    # -- helpers ----------------------------------------------------------------
    # -- categorical cells ------------------------------------------------------
    @staticmethod
    def _categorical_codes(series):
        """Sorted observed-value codes for a categorical input; preserve missing values as NaN.

        The vocabulary uses feature values only, with no target statistics or frequency encoding.
        Casting to object also makes codes independent of a pandas category dtype's declared order.
        """
        codes, _ = pd.factorize(series.astype(object), sort=True)
        values = codes.astype(np.float64)
        values[codes < 0] = np.nan
        return values

    @staticmethod
    def categorical_context_columns(df, label=None):
        """All string/object, bool and declared pandas categories, with no cardinality cutoff.

        Numerical identifiers can be declared with ``df[col] = df[col].astype('category')``.
        Raw strings are category identities here; this does not supply semantic text embeddings.
        """
        return [c for c in df.columns if c != label and (
            isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]))]

    def categorical_columns(self, df, label=None, columns=None, max_categories: int = 30):
        """Categorical *targets* to check, not the list of categorical input features.

        Text/bool/category targets have 2..30 levels (and at most half as many levels as rows);
        numeric targets have 2..(min_distinct-1) distinct values. High-cardinality columns stay
        in the input context without being flagged merely because each ID is rare.
        """
        out = []
        for c in (df.columns if columns is None else columns):
            if c == label:
                continue
            s = df[c].dropna(); k = s.nunique()
            if k < 2:
                continue
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                if k < self.min_distinct:
                    out.append(c)
            elif k <= max_categories and k <= 0.5 * len(s):
                out.append(c)
        return out

    def categorical_scores(self, df, cat_cols, label=None):
        """Out-of-fold P(recorded category | rest of row) from the TabPFN-3.5 classifier, for each categorical column.
        Features include all numeric columns, all other categorical inputs regardless of cardinality, and the label."""
        from sklearn.model_selection import KFold
        n = len(df)
        context_cats = self.categorical_context_columns(df, label)
        num = [c for c in df.columns if c != label and c not in cat_cols and c not in context_cats
               and pd.api.types.is_numeric_dtype(df[c])]
        all_cats = list(dict.fromkeys(list(cat_cols) + context_cats + ([label] if label is not None else [])))
        codes = {c: self._categorical_codes(df[c]) for c in all_cats}
        out = {}
        for c in cat_cols:
            t_codes, levels = pd.factorize(df[c])
            feats = [df[x].to_numpy(np.float64) for x in num]
            cat_idx = []
            for other in [x for x in all_cats if x != c]:
                cat_idx.append(len(feats)); feats.append(codes[other])
            Z = np.column_stack(feats) if feats else np.zeros((n, 1))
            rows = np.where(t_codes >= 0)[0]
            P_rec = np.ones(n); top = np.full(n, -1); p_top = np.ones(n)
            for tr, te in KFold(self.folds, shuffle=True, random_state=self.seed).split(rows):
                tr, te = rows[tr], rows[te]
                ytr = t_codes[tr]
                if len(np.unique(ytr)) < 2:
                    P_rec[te] = (t_codes[te] == ytr[0]).astype(float); top[te] = ytr[0]; continue
                m = self._clf(cat_idx, max_categories=max(30, n))
                proba = m.fit(Z[tr], ytr).predict_proba(Z[te])
                cls = list(m.classes_)
                col = {k: i for i, k in enumerate(cls)}
                P_rec[te] = [proba[r, col[t]] if t in col else 0.0 for r, t in enumerate(t_codes[te])]
                top[te] = np.asarray(cls)[proba.argmax(1)]; p_top[te] = proba.max(1)
            sug = np.array([levels[int(k)] if k >= 0 else None for k in top], dtype=object)
            out[c] = {"surprise": np.where(t_codes >= 0, -np.log10(np.clip(P_rec, 1e-12, 1)), 0.0), "p_recorded": P_rec,
                      "suggested": sug, "p_top": p_top}
        return out

    @staticmethod
    def _category_cause(series, value, p_rec, suggested, p_top):
        """Typo-style hint for rare text categories close to a common one; otherwise the model's view."""
        import difflib
        counts = series.value_counts()
        n_val = int(counts.get(value, 0))
        if isinstance(value, str) and n_val <= 2:
            common = [str(k) for k, v in counts.items() if v >= 5 and k != value]
            close = difflib.get_close_matches(value, common, n=1, cutoff=0.75)
            if close:
                return f"possible typo of '{close[0]}' ('{value}' appears {n_val}×)"
            return f"rare category ('{value}' appears {n_val}×)"
        return f"unlikely given the rest of the row: TabPFN gives '{value}' {p_rec:.1%}, most likely '{suggested}' ({p_top:.0%})"

    @staticmethod
    def _cause(v, lo, hi, med=None, placeholder_column=False):
        """Name a common slip only if the corrected value lies inside TabPFN's 80% predictive interval AND is much
        closer to TabPFN's median than the recorded value (at most a third of the distance)."""
        if placeholder_column:
            return "implausible given the rest of the row (this column uses a placeholder code, so check the source)"
        far = abs(v - med) if med is not None and np.isfinite(med) else np.inf
        inside = lambda x: (x is not None and np.isfinite(x) and lo <= x <= hi
                            and (med is None or not np.isfinite(med) or abs(x - med) <= far / 3))
        if v == 0 and not inside(0):
            return "possible missing value recorded as 0"
        for name, fn in (("decimal slip (×10)", lambda x: x / 10), ("decimal slip (÷10)", lambda x: x * 10),
                         ("extra zeros (×1000)", lambda x: x / 1000), ("sign flip", lambda x: -x),
                         ("swapped leading digits", _transpose)):
            cand = fn(v)
            if inside(cand):
                return f"possible {name}: {v:.6g} → {cand:.6g} would be typical"
        return "implausible given the rest of the row"

    @staticmethod
    def _patterns(df, num):
        """Column-level placeholder codes: one value covering >=5% of rows, outside the 1-99% range of the rest."""
        out = []
        for c in num:
            v = df[c].dropna()
            if len(v) < 20:
                continue
            top, cnt = v.value_counts().index[0], int(v.value_counts().iloc[0])
            rest = v[v != top]
            if cnt / len(v) >= 0.05 and len(rest) > 10 and not (rest.quantile(.01) <= top <= rest.quantile(.99)):
                out.append({"column": c, "value": float(top), "share": cnt / len(v),
                            "message": f"{c}: the value {top:g} appears in {cnt / len(v):.0%} of rows and lies outside the range of the "
                                       f"other values — likely a placeholder / missing-value code. Proofread treats it as data, so "
                                       f"those cells are not flagged individually."})
        return out

    @staticmethod
    def _context(df, related, col, row, k=3):
        """Most related columns (|Spearman| on this table) and their values in the flagged row — context, not attribution."""
        if related is None:
            return ""
        top = related[col].drop(col).sort_values(ascending=False).head(k)
        return "; ".join(f"{c} = {df.at[row, c]:.4g}" for c in top.index if np.isfinite(top[c]))


def set_cell(df: pd.DataFrame, row: int, column, value):
    """Write a suggested value into df without dtype errors (integer columns get a rounded integer)."""
    dt = df[column].dtype
    if pd.api.types.is_integer_dtype(dt) and isinstance(value, (float, np.floating)) and np.isfinite(value):
        value = int(round(value))
    elif pd.api.types.is_bool_dtype(dt) or (pd.api.types.is_numeric_dtype(dt) and isinstance(value, str)):
        df[column] = df[column].astype(object)
    df.at[row, column] = value
