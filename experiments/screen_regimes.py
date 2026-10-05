"""Screen binary numeric tasks for a TabPFN-vs-logistic gap at context 200. See docs/REGIME_SCREEN_PREREG.md.

Usage: CUDA_VISIBLE_DEVICES=g PYTHONPATH=. python experiments/screen_regimes.py --out results/regime_screen
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from experiments.verify import factory

OPENML = {'phoneme': 1489, 'eeg-eye-state': 1471, 'MagicTelescope': 1120, 'banknote': 1462, 'spambase': 44,
          'wilt': 40983, 'ozone-level': 1487, 'kc1': 1067, 'blood-transfusion': 1464, 'climate-crashes': 1467,
          'ilpd': 1480, 'diabetes': 37, 'steel-plates': 1504, 'electricity': 151, 'pc1': 1068, 'jm1': 1053,
          'satellite': 40900, 'mozilla4': 1046, 'madelon': 1485}


def synthetic(name, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    if name == 'synthetic-xor':
        X = rng.normal(size=(n, 8)); y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    else:  # two moons with noise + 4 nuisance dims
        from sklearn.datasets import make_moons
        m, y = make_moons(n, noise=0.25, random_state=seed)
        X = np.c_[m, rng.normal(size=(n, 4))]
    return X.astype(np.float32), y, 0


def load(name):
    if name.startswith('synthetic'):
        return synthetic(name)
    import openml
    ds = openml.datasets.get_dataset(OPENML[name], download_data=True, download_qualities=False,
                                     download_features_meta_data=False)
    df, y, cat, names = ds.get_data(target=ds.default_target_attribute, dataset_format='dataframe')
    keep = [n for n, c in zip(names, cat) if not c]
    dropped = len(names) - len(keep)
    X = df[keep].apply(lambda s: s.astype(float)).to_numpy(np.float32)
    y = y.astype('category')
    if y.nunique() != 2:
        raise ValueError(f'{name}: not binary')
    y = (y.cat.codes.to_numpy() == y.value_counts().idxmax() if False else y.cat.codes.to_numpy()).astype(int)
    ok = np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if len(y) > 3000:
        idx = np.random.default_rng(0).choice(len(y), 3000, replace=False); X, y = X[idx], y[idx]
    return X, y, dropped


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/regime_screen')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = list(OPENML) + ['synthetic-xor', 'synthetic-moons']
    for name in names:
        if (out / f'{name}.json').exists():
            continue
        try:
            X, y, dropped = load(name)
        except Exception as e:
            (out / f'{name}.json').write_text(json.dumps({'dataset': name, 'status': 'load_failed', 'error': str(e)[:300]}))
            print(name, 'load failed', str(e)[:100], flush=True); continue
        rec = {'dataset': name, 'status': 'ok', 'n': int(len(y)), 'd': int(X.shape[1]),
               'categorical_dropped': int(dropped), 'minority_rate': float(min(y.mean(), 1 - y.mean())), 'seeds': {}}
        for seed in range(101, 111):
            try:
                ic, it = train_test_split(np.arange(len(y)), train_size=200, stratify=y, random_state=seed)
                it = it[:1000]
                res = {}
                for m in ('logistic', 'hgb', 'rf', 'tabpfn_standard'):
                    t = time.time(); mod = factory(m, seed, 'cuda:0').fit(X[ic], y[ic])
                    p = mod.predict_proba(X[it])[:, 1]
                    res[m] = {'acc': float(accuracy_score(y[it], p >= .5)), 'auc': float(roc_auc_score(y[it], p)), 's': time.time() - t}
                rec['seeds'][str(seed)] = res
            except Exception as e:
                rec['seeds'][str(seed)] = {'error': str(e)[:200]}
        (out / f'{name}.json').write_text(json.dumps(rec, indent=2))
        ok = [r for r in rec['seeds'].values() if 'logistic' in r]
        if ok:
            print(name, 'd', rec['d'], 'acc logistic %.3f tabpfn %.3f' % (np.mean([r['logistic']['acc'] for r in ok]),
                  np.mean([r['tabpfn_standard']['acc'] for r in ok])), flush=True)


if __name__ == '__main__':
    main()
