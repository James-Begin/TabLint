"""Precompute demo reports for the app (GPU): real Pima table; MagicTelescope & breast-cancer with injected,
logged errors (ground truth kept alongside so the app can show which flags are true errors)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import openml
from proofread import Proofreader
from benchmarks.proofread_confirm import inject

OUT = Path('results/proofread_demos'); OUT.mkdir(parents=True, exist_ok=True)
import os
pr = Proofreader(device=os.environ.get('PROOFREAD_DEVICE', 'cuda:0'))

ds = openml.datasets.get_dataset(37, download_data=True, download_qualities=False, download_features_meta_data=False)
df, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format='dataframe')
df = df.rename(columns={'preg': 'pregnancies', 'plas': 'glucose', 'pres': 'blood_pressure', 'skin': 'skin_fold',
                        'insu': 'insulin', 'mass': 'bmi', 'pedi': 'pedigree'}); df['diabetes'] = y.astype(str)
rep = pr.check(df, label='diabetes', threshold=2.0, max_issues=60)
rep.meta.update(title='Pima diabetes (raw, real errors)', source='https://www.openml.org/d/37', ground_truth=None)
rep.save(OUT / 'pima_real.json'); print(rep.to_markdown(12))

for name, oid, label in (('magic', 1120, None), ('breast', None, None)):
    if oid:
        d = openml.datasets.get_dataset(oid, download_data=True, download_qualities=False, download_features_meta_data=False)
        X, yy, _, cols = d.get_data(target=d.default_target_attribute, dataset_format='dataframe')
        X = X.sample(400, random_state=7).reset_index(drop=True); yy = yy.loc[X.index] if False else None
        dfc = X.astype(float)
    else:
        from sklearn.datasets import load_breast_cancer
        b = load_breast_cancer(as_frame=True)
        dfc = b.data.sample(400, random_state=7).reset_index(drop=True)
        dfc['diagnosis'] = np.where(b.target.loc[b.data.sample(400, random_state=7).index].to_numpy() == 0, 'malignant', 'benign')
    numc = [c for c in dfc.columns if pd.api.types.is_numeric_dtype(dfc[c]) and dfc[c].nunique() > 10]
    Xe, mask, kind = inject(dfc[numc].to_numpy(float), 0.02, np.random.default_rng(7))
    dirty = dfc.copy(); dirty[numc] = Xe
    truth = [{'row': int(i), 'column': numc[j], 'kind': str(kind[i, j]), 'true_value': float(dfc[numc].to_numpy(float)[i, j])}
             for i, j in zip(*np.where(mask))]
    lab = 'diagnosis' if 'diagnosis' in dirty.columns else None
    rep = pr.check(dirty, label=lab, threshold=2.0, max_issues=80)
    rep.meta.update(title=f'{name}: 2% injected errors (ground truth known)', ground_truth=truth)
    rep.save(OUT / f'{name}_injected.json')
    flagged = {(r.row, r.column) for r in rep.issues.itertuples() if r.kind == 'cell'}
    tset = {(t['row'], t['column']) for t in truth}
    k = len(tset); top = [(r.row, r.column) for r in rep.issues.itertuples() if r.kind == 'cell'][:k]
    print(name, 'errors', k, 'precision@k', sum(t in tset for t in top) / k, 'flagged', len(flagged), 'true among flagged', len(flagged & tset))
