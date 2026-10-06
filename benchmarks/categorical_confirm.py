"""Pre-registered H11 (docs/CATEGORICAL_PREREG.md): categorical-cell error detection."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, hashlib, io, json, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from proofread import Proofreader

DATASETS = ['credit-g', 'adult', 'mushroom', 'car', 'bank-marketing', 'nursery', 'cmc', 'kr-vs-kp', 'penguins', 'tips']
OPENML = {'credit-g': 31, 'adult': 1590, 'mushroom': 24, 'car': 40975, 'bank-marketing': 1461, 'nursery': 26, 'cmc': 23, 'kr-vs-kp': 3}
BASE = ['frequency', 'logistic', 'rf', 'knn', 'hgb']


def load(name):
    if name in OPENML:
        import openml
        d = openml.datasets.get_dataset(OPENML[name], download_data=True, download_qualities=False, download_features_meta_data=False)
        X, y, _, _ = d.get_data(target=d.default_target_attribute, dataset_format='dataframe')
        df = X.copy(); df['target'] = y
    else:
        df = pd.read_csv(io.StringIO(urllib.request.urlopen(f'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{name}.csv', timeout=60).read().decode()))
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == bool:
            df[c] = df[c].astype(object)
    return df


def inject(df, cat_cols, rate, rng):
    dirty = df.copy(); mask = pd.DataFrame(False, index=df.index, columns=cat_cols)
    cells = [(i, c) for c in cat_cols for i in df.index[df[c].notna()]]
    for k in rng.choice(len(cells), max(1, int(rate * len(cells))), replace=False):
        i, c = cells[k]; col = df[c].dropna()
        others = col[col != df.at[i, c]]
        if others.empty:
            continue
        dirty[c] = dirty[c].astype(object)
        dirty.at[i, c] = others.iloc[rng.integers(len(others))]
        mask.at[i, c] = True
    return dirty, mask


def baseline_scores(df, cat_cols, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
    out = {b: {} for b in BASE}
    for c in cat_cols:
        t, levels = pd.factorize(df[c]); rows = np.where(t >= 0)[0]
        freq = df[c].map(df[c].value_counts(normalize=True)).astype(float).fillna(1.0).to_numpy()
        out['frequency'][c] = -np.log(np.clip(freq, 1e-12, 1))
        F = df.drop(columns=[c])
        cats = [x for x in F.columns if x in cat_cols or not pd.api.types.is_numeric_dtype(F[x])]
        nums = [x for x in F.columns if x not in cats]
        Fs = F.copy()
        for x in cats:
            Fs[x] = Fs[x].astype(str)
        onehot = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore'), cats), ('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), nums)])
        ordinal = ColumnTransformer([('c', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cats), ('n', SimpleImputer(strategy='median'), nums)])
        models = {'logistic': lambda: make_pipeline(onehot, LogisticRegression(max_iter=2000)),
                  'rf': lambda: make_pipeline(onehot, RandomForestClassifier(200, n_jobs=2, random_state=seed)),
                  'knn': lambda: make_pipeline(onehot, KNeighborsClassifier(10)),
                  'hgb': lambda: make_pipeline(ordinal, HistGradientBoostingClassifier(categorical_features=list(range(len(cats))), random_state=seed))}
        for b, mk in models.items():
            P = np.ones(len(df))
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(rows):
                tr, te = rows[tr], rows[te]
                if len(np.unique(t[tr])) < 2:
                    P[te] = (t[te] == t[tr][0]).astype(float); continue
                m = mk().fit(Fs.iloc[tr], t[tr]); pr = m.predict_proba(Fs.iloc[te]); col = {k: j for j, k in enumerate(m.classes_)}
                P[te] = [pr[r, col[v]] if v in col else 0.0 for r, v in enumerate(t[te])]
            out[b][c] = np.where(t >= 0, -np.log(np.clip(P, 1e-12, 1)), 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--seeds', default='901,902,903,904,905'); ap.add_argument('--out', default='results/categorical')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    proto = out / 'protocol.json'
    if not proto.exists():
        proto.write_text(json.dumps({'design': 'benchmarks/categorical_confirm.py', 'source_sha256': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                         for f in ('proofread/core.py', 'benchmarks/categorical_confirm.py')}}, indent=2))
    pr = Proofreader(device='cuda:0')
    for ds in a.datasets.split(','):
        full = load(ds)
        for seed in map(int, a.seeds.split(',')):
            f = out / f'{ds}_{seed}.json'
            if f.exists():
                continue
            t0 = time.time()
            df = full.sample(min(400, len(full)), random_state=seed).reset_index(drop=True)
            cat_cols = pr.categorical_columns(df)
            dirty, mask = inject(df, cat_cols, 0.03, np.random.default_rng(seed))
            m = mask.to_numpy().ravel(order='F'); k = int(m.sum())
            sc = {'tabpfn': {c: v['surprise'] for c, v in pr.categorical_scores(dirty, cat_cols).items()}}
            sc.update(baseline_scores(dirty, cat_cols, seed))
            rec = {'dataset': ds, 'seed': seed, 'n': len(df), 'categorical_columns': cat_cols, 'errors': k, 'precision_at_k': {}, 'auroc': {}}
            for name, d in sc.items():
                s = np.concatenate([np.nan_to_num(d[c], nan=0, posinf=1e12) for c in cat_cols])
                rec['precision_at_k'][name] = float(m[np.argsort(-s, kind='stable')[:k]].mean())
                rec['auroc'][name] = float(roc_auc_score(m, s))
            rec['seconds'] = time.time() - t0
            f.write_text(json.dumps(rec, indent=2))
            p = rec['precision_at_k']
            print(ds, seed, len(cat_cols), 'cols', k, 'errors | P@k tabpfn %.2f best-base %.2f (%s)' % (p['tabpfn'], max(p[b] for b in BASE), max(BASE, key=p.get)), round(rec['seconds']), flush=True)


if __name__ == '__main__':
    main()
