"""Summarize the regime screen; applies the pre-registered selection rule. Prints markdown table."""
import json, sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/regime_screen')
rng = np.random.default_rng(0)
rows, out = [], {}
for p in sorted(D.glob('*.json')):
    if p.name == 'summary.json':
        continue
    r = json.loads(p.read_text())
    if r['status'] != 'ok':
        out[r['dataset']] = {'status': r['status'], 'error': r.get('error')}; continue
    s = [v for v in r['seeds'].values() if 'logistic' in v]
    acc = {m: np.array([v[m]['acc'] for v in s]) for m in ('logistic', 'hgb', 'rf', 'tabpfn_standard')}
    diff = acc['tabpfn_standard'] - acc['logistic']
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    e = {'status': 'ok', 'n': r['n'], 'd': r['d'], 'categorical_dropped': r['categorical_dropped'], 'seeds': len(s),
         'acc': {m: float(a.mean()) for m, a in acc.items()}, 'diff_vs_logistic': [float(diff.mean()), float(lo), float(hi)],
         'qualifies': bool(lo >= 0.05)}
    out[r['dataset']] = e
    rows.append((diff.mean(), r['dataset'], e))
(D / 'summary.json').write_text(json.dumps(out, indent=2))
print('| Dataset | d | dropped cat. | Logistic | HGB | RF | TabPFN-3.5 | TabPFN − logistic [95% CI] | Qualifies |')
print('|---|---:|---:|---:|---:|---:|---:|---|---|')
for _, n, e in sorted(rows, reverse=True):
    a, d = e['acc'], e['diff_vs_logistic']
    print(f"| {n} | {e['d']} | {e['categorical_dropped']} | {a['logistic']:.3f} | {a['hgb']:.3f} | {a['rf']:.3f} | {a['tabpfn_standard']:.3f} | "
          f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {'yes' if e['qualifies'] else 'no'} |")
print('qualifying:', [n for _, n, e in rows if e['qualifies']])
