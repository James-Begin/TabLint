"""EXPLORATORY: full conformal classification with TabPFN-3.5 vs split conformal (dev seeds).

Full CP: for each test x and candidate label c, refit on train ∪ {(x,c)} (one forward pass), score all n+1
points s_i = 1 - p(y_i | x_i) in-sample; p-value = (1 + #{i<=n: s_i >= s_{n+1}}) / (n+1); keep c if p > alpha.
Split CP: fit on half, calibrate on half (standard). Metrics: coverage, mean set size, singleton rate.
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from experiments.verify import factory
from benchmarks.common import load_any


def split_cp(model, X, y, Xt, alpha, seed):
    tr, ca = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
    m = factory(model, seed, 'cuda:0').fit(X[tr], y[tr])
    s = 1 - m.predict_proba(X[ca])[np.arange(len(ca)), y[ca]]
    q = np.quantile(s, min(1, np.ceil((len(ca) + 1) * (1 - alpha)) / len(ca)), method='higher')
    return (1 - m.predict_proba(Xt)) <= q


def full_cp(model, X, y, Xt, alpha, seed):
    sets = np.zeros((len(Xt), 2), bool)
    for j in range(len(Xt)):
        for c in (0, 1):
            Xa = np.vstack([X, Xt[j:j + 1]]); ya = np.r_[y, c]
            p = factory(model, seed, 'cuda:0').fit(Xa, ya).predict_proba(Xa)
            s = 1 - p[np.arange(len(ya)), ya]
            pval = (1 + (s[:-1] >= s[-1]).sum()) / (len(ya))
            sets[j, c] = pval > alpha
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='eeg-eye-state,MagicTelescope,breast-cancer,credit-g')
    ap.add_argument('--seeds', default='101'); ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--test', type=int, default=150); ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--out', default='results/pilots/conformal')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets.split(','):
        X, y = load_any(ds)
        for seed in map(int, a.seeds.split(',')):
            ic, it = train_test_split(np.arange(len(y)), train_size=a.n, stratify=y, random_state=seed)
            it = it[:a.test]; Xc, yc, Xt, yt = X[ic], y[ic], X[it], y[it]
            rec = {'dataset': ds, 'seed': seed, 'n': a.n, 'alpha': a.alpha, 'methods': {}}
            runs = [('split_' + m, lambda m=m: split_cp(m, Xc, yc, Xt, a.alpha, seed)) for m in ('logistic', 'rf', 'hgb', 'tabpfn_standard')]
            runs += [('full_' + m, lambda m=m: full_cp(m, Xc, yc, Xt, a.alpha, seed)) for m in ('logistic', 'tabpfn_standard')]
            for name, fn in runs:
                t = time.time(); S = fn()
                rec['methods'][name] = {'coverage': float(S[np.arange(len(yt)), yt].mean()), 'size': float(S.sum(1).mean()),
                                        'singleton_correct': float(((S.sum(1) == 1) & S[np.arange(len(yt)), yt]).mean()),
                                        'empty': float((S.sum(1) == 0).mean()), 'seconds': time.time() - t}
            (out / f'{ds}_{seed}_{a.n}.json').write_text(json.dumps(rec, indent=2))
            print(ds, seed, a.n, {k: (round(v['coverage'], 2), round(v['size'], 2), round(v['singleton_correct'], 2)) for k, v in rec['methods'].items()}, flush=True)


if __name__ == '__main__':
    main()
