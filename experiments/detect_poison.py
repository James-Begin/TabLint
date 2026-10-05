"""Can an auditor detect the appended rows? (secondary, descriptive analysis)

For each audit report, append the gradient attack's EXACT rows and labels to the
reconstructed context (hash-checked) and score every row of the augmented context:
  loo_influence   |P(target) change| when the row is removed (exact refit; TabPFN-3.5,
                  1 estimator, fingerprint off to avoid row-count hashing confounds)
  label_suspicion 1 - P(given label) from 5-fold out-of-fold TabPFN (same settings)
  outlier         standardized distance to nearest OTHER row
Report AUROC (appended vs real rows), and how many of the k appended rows are in the
top-k of each score. No detector is tuned on these labels.

Usage: CUDA_VISIBLE_DEVICES=g python experiments/detect_poison.py DIR --shard i --nshards n
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from auditkit.cli import dataset
from fz.core import make_std_clf


def clf(seed):
    return make_std_clf(n_estimators=1, seed=seed, device='cuda:0',
                        inference_config={'FINGERPRINT_FEATURE': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--method', default='gradient')
    a = ap.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('explicit GPU required')
    files = sorted(p for p in Path(a.directory).glob('*.json')
                   if p.name not in ('protocol.json', 'manifest.json', 'paired_summary.json'))
    files = files[a.shard::a.nshards]
    out_dir = Path(a.directory) / 'detection'
    out_dir.mkdir(exist_ok=True)
    cache = {}
    for f in files:
        r = json.loads(f.read_text())
        if not isinstance(r, dict) or r.get('schema_version') != '1.0':
            continue
        name = r['dataset']['dataset']
        if name not in cache:
            cache[name] = dataset(name)
        X, y, _ = cache[name]
        ids = np.asarray(r['experiment']['context_ids'])
        Xc, yc = X[ids], y[ids]
        assert hashlib.sha256(Xc.tobytes() + yc.tobytes()).hexdigest() == r['experiment']['context_sha256']
        tid = r['experiment']['target_id']
        xt = X[tid:tid + 1]
        atk = next(x for x in r['attacks'] if x['method'] == a.method)
        rows = np.asarray(atk['rows'], np.float32)
        labels = np.asarray(atk['labels'], int)
        Xa = np.concatenate([Xc, rows]).astype(np.float32)
        ya = np.concatenate([yc, labels])
        is_poison = np.r_[np.zeros(len(yc), bool), np.ones(len(labels), bool)]
        seed = r['config']['seed']
        t0 = time.time()
        base = float(clf(seed).fit(Xa, ya).predict_proba(xt)[0, 1])
        loo = np.zeros(len(ya))
        for i in range(len(ya)):
            keep = np.arange(len(ya)) != i
            loo[i] = abs(float(clf(seed).fit(Xa[keep], ya[keep]).predict_proba(xt)[0, 1]) - base)
        oof = np.zeros(len(ya))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xa, ya):
            p = clf(seed).fit(Xa[tr], ya[tr]).predict_proba(Xa[te])
            oof[te] = 1 - p[np.arange(len(te)), ya[te]]
        mu, sd = Xc.mean(0), Xc.std(0)
        sd[sd == 0] = 1
        Z = (Xa - mu) / sd
        dist = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        outlier = dist.min(1)
        scores = {'loo_influence': loo, 'label_suspicion': oof, 'outlier': outlier}
        k = int(is_poison.sum())
        rec = {'report': f.name, 'dataset': name, 'seed': r['experiment']['seed'], 'target_id': tid,
               'method': a.method, 'attack_valid': atk['valid'],
               'standard_flip': next((v.get('flipped') for v in r.get('verification', [])
                                      if v['model'] == 'tabpfn_standard' and v['attack_method'] == a.method), None),
               'p_target_with_rows': base, 'k': k, 'n_rows': len(ya), 'seconds': None}
        for nm, s in scores.items():
            rec[f'{nm}_auroc'] = float(roc_auc_score(is_poison, s))
            top = np.argsort(-s, kind='stable')[:k]
            rec[f'{nm}_hits_in_topk'] = int(is_poison[top].sum())
        rec['seconds'] = time.time() - t0
        (out_dir / f.name).write_text(json.dumps(rec, indent=2, allow_nan=False))
        print(f.name, {kk: round(v, 3) if isinstance(v, float) else v for kk, v in rec.items()
                       if 'auroc' in kk or 'hits' in kk}, round(rec['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
