"""D. Leave-one-out detection with non-TabPFN models (CPU). See docs/COMPARISONS_PREREG.md.

Usage: PYTHONPATH=. python experiments/detect_models.py DIR --shard i --nshards n
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from auditkit.cli import dataset
from experiments.verify import factory

MODELS = ('hgb', 'rf', 'logistic')


def auroc(positive, score):
    r = rankdata(score)
    n1, n0 = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def topk_hits(positive, score, k):
    # ties broken pessimistically for the detector: appended rows placed last among equal scores
    order = np.lexsort((positive.astype(int), -score))
    return int(positive[order[:k]].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    d = Path(a.directory); out = d / 'detection_models'; out.mkdir(exist_ok=True)
    files = sorted(p for p in d.glob('*.json') if p.name not in ('protocol.json', 'manifest.json', 'paired_summary.json'))[a.shard::a.nshards]
    cache = {}
    for f in files:
        r = json.loads(f.read_text())
        if r.get('schema_version') != '1.0':
            continue
        name = r['dataset']['dataset']
        X, y, _ = cache.setdefault(name, dataset(name))
        ids = np.asarray(r['experiment']['context_ids']); Xc, yc = X[ids], y[ids]
        assert hashlib.sha256(Xc.tobytes() + yc.tobytes()).hexdigest() == r['experiment']['context_sha256']
        xt = X[r['experiment']['target_id']:r['experiment']['target_id'] + 1]
        atk = next(x for x in r['attacks'] if x['method'] == 'gradient')
        Xa = np.concatenate([Xc, np.asarray(atk['rows'], np.float32)]).astype(np.float32)
        ya = np.concatenate([yc, np.asarray(atk['labels'], int)])
        pos = np.r_[np.zeros(len(yc), bool), np.ones(len(atk['labels']), bool)]
        k, seed = int(pos.sum()), r['config']['seed']
        rec = {'report': f.name, 'dataset': name, 'seed': r['experiment']['seed'], 'k': k, 'attack_valid': atk['valid']}
        for m in MODELS:
            t = time.time()
            base = float(factory(m, seed).fit(Xa, ya).predict_proba(xt)[0, 1])
            loo = np.zeros(len(ya))
            for i in range(len(ya)):
                keep = np.arange(len(ya)) != i
                loo[i] = abs(float(factory(m, seed).fit(Xa[keep], ya[keep]).predict_proba(xt)[0, 1]) - base)
            rec[f'{m}_auroc'] = auroc(pos, loo)
            rec[f'{m}_hits_in_topk'] = topk_hits(pos, loo, k)
            rec[f'{m}_frac_zero_scores'] = float((loo == 0).mean())
            rec[f'{m}_seconds'] = time.time() - t
        (out / f.name).write_text(json.dumps(rec, indent=2, allow_nan=False))
        print(f.name, {kk: round(v, 3) for kk, v in rec.items() if 'auroc' in kk}, flush=True)


if __name__ == '__main__':
    main()
