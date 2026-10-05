"""E. Receiver-targeted query-matched search. See docs/COMPARISONS_PREREG.md.

Usage: CUDA_VISIBLE_DEVICES=g PYTHONPATH=. python experiments/targeted_search.py DIR --shard i --nshards n
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '2')
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from auditkit.cli import dataset
from auditkit.engine import AuditConfig, FeatureDomain, candidate_metrics
from experiments.verify import factory

RECEIVERS = ('tabpfn_standard', 'hgb', 'rf', 'logistic')


def search(method, model_fn, domain, Xc, yc, xt, start, labels, seeds, budget, cfg_seed, j, before_decision):
    rng = np.random.default_rng(cfg_seed + 100 + j)
    prob = lambda rows: float(model_fn().fit(np.concatenate([Xc, rows]), np.concatenate([yc, labels])).predict_proba(xt)[0, 1])
    best = start.copy(); bp = prob(best)
    valid = candidate_metrics(domain, best, xt, seeds)['valid']
    objective = 1 - int(before_decision)
    for _ in range(budget):
        if method == 'random_search':
            raw = start + rng.normal(size=start.shape) * domain.std * rng.choice([.1, .3, .8])
        else:
            raw = best.astype(float).copy()
            col = int(rng.integers(raw.shape[1])); row = int(rng.integers(len(raw)))
            raw[row, col] += rng.choice([-1, 1]) * rng.choice([.1, .3, .8]) * domain.std[col]
        cand = domain.project(raw, seeds)
        p = prob(cand)
        ok = candidate_metrics(domain, cand, xt, seeds)['valid']
        score, bscore = (p, bp) if objective else (1 - p, 1 - bp)
        if ok and (not valid or score > bscore):
            best, bp, valid = cand, p, True
    return bool(valid and (bp >= .5) != bool(before_decision)), bp, bool(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('explicit GPU required')
    d = Path(a.directory); out = d / 'targeted'; out.mkdir(exist_ok=True)
    files = sorted(p for p in d.glob('*.json') if p.name not in ('protocol.json', 'manifest.json', 'paired_summary.json'))[a.shard::a.nshards]
    cache = {}
    for f in files:
        r = json.loads(f.read_text())
        if r.get('schema_version') != '1.0':
            continue
        name = r['dataset']['dataset']
        X, y, meta = cache.setdefault(name, dataset(name))
        ids = np.asarray(r['experiment']['context_ids']); Xc, yc = X[ids], y[ids]
        assert hashlib.sha256(Xc.tobytes() + yc.tobytes()).hexdigest() == r['experiment']['context_sha256']
        tid = r['experiment']['target_id']; xt = X[tid:tid + 1]
        c = r['config']
        cfg = AuditConfig(seed=c['seed'], steps=c['steps'], lr=c['lr'], k=c['k'], device='cpu',
                          groups=tuple(tuple(g) for g in c['groups']), integer_indices=tuple(c['integer_indices']),
                          protected_indices=tuple(c['protected_indices']), min_target_distance=c['min_target_distance'],
                          max_real_distance=c['max_real_distance'])
        domain = FeatureDomain.fit(Xc, cfg, meta['names'])
        init = next(x for x in r['attacks'] if x['method'] == 'near_rows')
        grad = next(x for x in r['attacks'] if x['method'] == 'gradient')
        seeds = Xc[init['seed_indices']]; start = np.asarray(init['rows'], np.float32)
        labels = np.asarray(init['labels'], int); budget = grad['evaluations']
        sur_dec = int(r['baseline']['p_positive'] >= .5)
        rec = {'report': f.name, 'dataset': name, 'seed': r['experiment']['seed'], 'budget': budget, 'receivers': {}}
        for m in RECEIVERS:
            t = time.time()
            fn = lambda m=m: factory(m, c['seed'], 'cuda:0')
            before = float(fn().fit(Xc, yc).predict_proba(xt)[0, 1]); dec = int(before >= .5)
            entry = {'clean_p': before, 'clean_agrees_with_surrogate': dec == sur_dec}
            for j, method in enumerate(('random_search', 'coordinate_search')):
                fl, p, v = search(method, fn, domain, Xc, yc, xt, start, labels, seeds, budget, c['seed'], j, dec)
                entry[method] = {'flipped': fl, 'p_after': p, 'valid': v}
            tv = [v for v in r.get('verification', []) if v['model'] == m and v['attack_method'] == 'gradient']
            entry['transfer_gradient_flipped'] = bool(tv[0]['flipped'] and tv[0]['candidate_valid']) if tv else None
            entry['seconds'] = time.time() - t
            rec['receivers'][m] = entry
        (out / f.name).write_text(json.dumps(rec, indent=2, allow_nan=False))
        print(f.name, {m: (e['random_search']['flipped'], e['coordinate_search']['flipped']) for m, e in rec['receivers'].items()}, flush=True)


if __name__ == '__main__':
    main()
