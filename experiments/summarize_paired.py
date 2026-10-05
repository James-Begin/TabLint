"""Paired, cluster-resampled comparison of audit methods on identical targets.

Primary endpoint: VERIFIED flip on a receiver model, using a valid candidate, over
ALL sampled targets (failures/aborts count as non-flips). Comparisons are paired
by target. Uncertainty: two-level bootstrap, resampling context seeds (clusters)
and then targets within each resampled seed.

Usage: python experiments/summarize_paired.py results/<dir> [--receiver tabpfn_standard]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ['gradient', 'random_search', 'coordinate_search', 'near_rows', 'random_real']


def load(directory):
    rows = []
    for p in sorted(Path(directory).glob('*.json')):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or r.get('schema_version') != '1.0':
            continue
        rows.append(r)
    return rows


def outcome(report, method, receiver):
    a = next((a for a in report['attacks'] if a['method'] == method), None)
    if a is None:
        return None
    if receiver == 'surrogate':
        return int(bool(a['valid'] and a['flipped']))
    v = next((v for v in report.get('verification', [])
              if v['model'] == receiver and v['attack_method'] == method), None)
    if v is None or 'flipped' not in v:
        return 0  # missing verification counts as no verified flip
    return int(bool(v.get('candidate_valid') and v['flipped']))


def cluster_ci(table, a, b, n=4000, seed=0, level=95.0):
    """table: {cluster: [(x_a, x_b), ...]}. Returns mean diff and 95% CI."""
    rng = np.random.default_rng(seed)
    keys = list(table)
    diffs = []
    for _ in range(n):
        ks = rng.choice(keys, len(keys), replace=True)
        vals = []
        for k in ks:
            arr = np.asarray(table[k], float)
            vals.append(arr[rng.integers(0, len(arr), len(arr))])
        v = np.concatenate(vals)
        diffs.append(v[:, 0].mean() - v[:, 1].mean())
    allv = np.concatenate([np.asarray(t, float) for t in table.values()])
    tail = (100.0 - level) / 2
    return float(allv[:, 0].mean() - allv[:, 1].mean()), float(np.percentile(diffs, tail)), \
        float(np.percentile(diffs, 100.0 - tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory')
    ap.add_argument('--level', type=float, default=95.0, help='CI level, e.g. 98.33 for Bonferroni(3)')
    ap.add_argument('--receivers', default='surrogate,tabpfn_standard,tabpfn_nofingerprint,hgb,'
                    'hgb_regularized,rf,rf_regularized,logistic')
    a = ap.parse_args()
    reports = load(a.directory)
    receivers = a.receivers.split(',')
    out = {'directory': a.directory, 'n_reports': len(reports), 'by_dataset': {}}
    datasets = sorted({r['dataset']['dataset'] for r in reports})
    for ds in datasets:
        rs = [r for r in reports if r['dataset']['dataset'] == ds]
        res = {'targets': len(rs), 'context_seeds': sorted({r['experiment']['seed'] for r in rs}),
               'gradient_aborted': sum(next(x for x in r['attacks'] if x['method'] == 'gradient')
                                       .get('aborted', False) for r in rs),
               'rates': {}, 'paired_gradient_minus_best_search': {}}
        for rec in receivers:
            res['rates'][rec] = {m: float(np.mean([outcome(r, m, rec) or 0 for r in rs]))
                                 for m in METHODS}
            # Best query-matched black-box baseline, chosen per receiver by overall rate.
            best = max(('random_search', 'coordinate_search'), key=lambda m: res['rates'][rec][m])
            table = defaultdict(list)
            for r in rs:
                table[r['experiment']['seed']].append((outcome(r, 'gradient', rec) or 0,
                                                        outcome(r, best, rec) or 0))
            d, lo, hi = cluster_ci(table, 'gradient', best, level=a.level)
            res['paired_gradient_minus_best_search'][rec] = {
                'baseline': best, 'diff': d, 'ci95': [lo, hi], 'ci_level': a.level}
        out['by_dataset'][ds] = res
        print(f"\n== {ds}: {res['targets']} targets, seeds {res['context_seeds']}, "
              f"gradient aborts {res['gradient_aborted']}")
        print(f"{'receiver':22s}" + ''.join(f'{m:>18s}' for m in METHODS) + f'   grad - best search [{a.level:g}% CI]')
        for rec in receivers:
            p = res['paired_gradient_minus_best_search'][rec]
            print(f'{rec:22s}' + ''.join(f"{res['rates'][rec][m]:18.2f}" for m in METHODS) +
                  f"   {p['diff']:+.2f} [{p['ci95'][0]:+.2f},{p['ci95'][1]:+.2f}] vs {p['baseline']}")
    (Path(a.directory) / 'paired_summary.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
