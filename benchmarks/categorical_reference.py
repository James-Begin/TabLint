"""Frozen scoring policy for the published 50-run categorical confirmation.

The production engine now retains high-cardinality context and uses sorted feature codes.
Keep this measured reference policy (first-seen codes and checked-category context only)
so rerunning the original confirmation does not silently change its inputs.
"""
import numpy as np
import pandas as pd
from proofread import Proofreader


class CategoricalConfirmation(Proofreader):
    def categorical_scores(self, df, cat_cols, label=None):
        """Out-of-fold P(recorded category | rest of row) from the TabPFN-3.5 classifier, for each categorical column.
        Features: all numeric columns, the other categorical columns (as TabPFN categorical features) and the label."""
        from sklearn.model_selection import KFold
        n = len(df)
        num = [c for c in df.columns if c != label and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]
        codes = {c: pd.factorize(df[c])[0].astype(np.float64) for c in cat_cols + ([label] if label else [])}
        for c in codes:
            codes[c][codes[c] < 0] = np.nan
        out = {}
        for c in cat_cols:
            t_codes, levels = pd.factorize(df[c])
            feats = [df[x].to_numpy(np.float64) for x in num]
            cat_idx = []
            for other in [x for x in cat_cols if x != c] + ([label] if label else []):
                cat_idx.append(len(feats)); feats.append(codes[other])
            Z = np.column_stack(feats) if feats else np.zeros((n, 1))
            rows = np.where(t_codes >= 0)[0]
            P_rec = np.ones(n); top = np.full(n, -1); p_top = np.ones(n)
            for tr, te in KFold(self.folds, shuffle=True, random_state=self.seed).split(rows):
                tr, te = rows[tr], rows[te]
                ytr = t_codes[tr]
                if len(np.unique(ytr)) < 2:
                    P_rec[te] = (t_codes[te] == ytr[0]).astype(float); top[te] = ytr[0]; continue
                from tabpfn import TabPFNClassifier
                m = TabPFNClassifier.create_default_for_version(self._version(), device=self.device, n_estimators=4, random_state=self.seed,
                                                                ignore_pretraining_limits=True, categorical_features_indices=cat_idx or None)
                proba = m.fit(Z[tr], ytr).predict_proba(Z[te])
                cls = list(m.classes_)
                col = {k: i for i, k in enumerate(cls)}
                P_rec[te] = [proba[r, col[t]] if t in col else 0.0 for r, t in enumerate(t_codes[te])]
                top[te] = np.asarray(cls)[proba.argmax(1)]; p_top[te] = proba.max(1)
            sug = np.array([levels[int(k)] if k >= 0 else None for k in top], dtype=object)
            out[c] = {"surprise": np.where(t_codes >= 0, -np.log10(np.clip(P_rec, 1e-12, 1)), 0.0), "p_recorded": P_rec,
                      "suggested": sug, "p_top": p_top}
        return out
