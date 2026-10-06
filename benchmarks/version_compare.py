"""Retained benchmark implementation; methodology and scope: docs/BENCHMARKS.md and docs/BENCHMARK_AMENDMENT.md."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from benchmarks.common import load_any
from benchmarks.proofread_confirm import DATASETS, inject

VERSIONS = ['V2', 'V3', 'V3_5', 'V3_5_FAST']


def reg(version, seed):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(getattr(ModelVersion, version), device='cuda:0', random_state=seed,
                                                      ignore_pretraining_limits=True)


def scores(version, Xe, y, seed):
    n, d = Xe.shape; S = np.zeros((n, d))
    for j in range(d):
        Z = np.c_[np.delete(Xe, j, 1), y]; t = Xe[:, j]; pit = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Z):
            o = reg(version, seed).fit(Z[tr], t[tr]).predict(Z[te], output_type='full')
            lg = o['logits']; yy = torch.tensor(t[te], dtype=lg.dtype, device=lg.device)
            F = o['criterion'].cdf(lg, yy.unsqueeze(-1)).squeeze(-1)
            pit[te] = np.clip(np.nan_to_num(F.detach().double().cpu().numpy().reshape(-1), nan=.5), 1e-9, 1 - 1e-9)
        S[:, j] = -np.log(2 * np.minimum(pit, 1 - pit))
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--seeds', default='801,802,803,804,805'); ap.add_argument('--out', default='results/version_compare')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets.split(','):
        X0, y0 = load_any(ds); cont = [j for j in range(X0.shape[1]) if len(np.unique(X0[:, j])) > 10]
        for seed in map(int, a.seeds.split(',')):
            f = out / f'{ds}_{seed}.json'
            if f.exists():
                continue
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y0), min(400, len(y0)), replace=False)
            X, y = X0[idx].astype(np.float64)[:, cont], y0[idx]
            if len(set(y)) < 2:
                continue
            Xe, mask, kind = inject(X, .03, np.random.default_rng(seed))
            k = int(mask.sum()); m = mask.ravel()
            rec = {'dataset': ds, 'seed': seed, 'errors': k, 'n': len(y), 'd': X.shape[1], 'versions': {}}
            for v in VERSIONS:
                t0 = time.time()
                s = np.nan_to_num(scores(v, Xe, y, seed).ravel(), nan=0, posinf=1e12)
                rec['versions'][v] = {'precision_at_k': float(m[np.argsort(-s, kind='stable')[:k]].mean()),
                                      'auroc': float(roc_auc_score(m, s)), 'seconds': time.time() - t0}
            f.write_text(json.dumps(rec, indent=2))
            print(ds, seed, {v: (round(r['precision_at_k'], 2), round(r['seconds'])) for v, r in rec['versions'].items()}, flush=True)


if __name__ == '__main__':
    main()
