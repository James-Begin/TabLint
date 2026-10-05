"""Summarize D (detection head-to-head) and E (receiver-targeted fragility). Writes comparisons_summary.json.

Two-level bootstrap intervals (resample seeds, then audits within seed); descriptive only.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/confirmation')
rng = np.random.default_rng(0)


LEVEL = float(sys.argv[2]) if len(sys.argv) > 2 else 95.0


def boot(rows, fn, n=4000):
    """rows: list of (seed, value-dict). fn(list of dicts)->float."""
    by = defaultdict(list)
    for s, v in rows:
        by[s].append(v)
    seeds = list(by)
    est = fn([v for s in seeds for v in by[s]])
    draws = []
    for _ in range(n):
        chosen = rng.choice(len(seeds), len(seeds))
        pool = []
        for c in chosen:
            g = by[seeds[c]]
            pool += [g[i] for i in rng.integers(0, len(g), len(g))]
        draws.append(fn(pool))
    a = (100 - LEVEL) / 2
    return est, float(np.percentile(draws, a)), float(np.percentile(draws, 100 - a))


def load(sub):
    return [json.loads(p.read_text()) for p in sorted((D / sub).glob('*.json'))]


out = {'detection': {}, 'targeted': {}}
det = {r['report']: r for r in load('detection') if 'report' in r}
mod = {r['report']: r for r in load('detection_models')}
for ds in sorted({r['dataset'] for r in mod.values()}):
    out['detection'][ds] = {}
    reps = [n for n in mod if mod[n]['dataset'] == ds and n in det]
    sets = {'tabpfn_loo': lambda n: (det[n]['loo_influence_auroc'], det[n]['loo_influence_hits_in_topk'] / det[n]['k'], det[n]['seconds']),
            'label_suspicion_tabpfn': lambda n: (det[n]['label_suspicion_auroc'], det[n]['label_suspicion_hits_in_topk'] / det[n]['k'], None)}
    for m in ('hgb', 'rf', 'logistic'):
        sets[f'{m}_loo'] = (lambda m: lambda n: (mod[n][f'{m}_auroc'], mod[n][f'{m}_hits_in_topk'] / mod[n]['k'], mod[n][f'{m}_seconds']))(m)
    diffs = [(mod[n]['seed'], det[n]['loo_influence_auroc'] - mod[n]['logistic_auroc']) for n in reps]
    out['detection'][ds]['_paired_tabpfn_minus_logistic_auroc'] = {'n': len(diffs), 'ci_level': LEVEL,
        'diff': boot(diffs, lambda v: float(np.mean(v)))}
    for name, get in sets.items():
        rows = [(mod[n]['seed'], get(n)) for n in reps]
        auc = boot(rows, lambda v: float(np.mean([x[0] for x in v])))
        rec = boot(rows, lambda v: float(np.mean([x[1] for x in v])))
        secs = [x[2] for _, x in rows if x[2] is not None]
        out['detection'][ds][name] = {'n': len(rows), 'auroc': auc, 'topk_recall': rec,
                                      'mean_seconds': float(np.mean(secs)) if secs else None}
tg = load('targeted')
for ds in sorted({r['dataset'] for r in tg}):
    out['targeted'][ds] = {}
    reps = [r for r in tg if r['dataset'] == ds]
    for m in ('tabpfn_standard', 'hgb', 'rf', 'logistic'):
        rows = [(r['seed'], r['receivers'][m]) for r in reps]
        rate = lambda key: (lambda v: float(np.mean([key(x) for x in v])))
        entry = {'n': len(rows)}
        for label, key in (('random_search', lambda x: x['random_search']['flipped']),
                           ('coordinate_search', lambda x: x['coordinate_search']['flipped']),
                           ('either_search', lambda x: x['random_search']['flipped'] or x['coordinate_search']['flipped']),
                           ('gradient_rows_transfer', lambda x: bool(x['transfer_gradient_flipped']))):
            entry[label] = boot(rows, rate(key))
        agree = [x for x in rows if x[1]['clean_agrees_with_surrogate']]
        entry['clean_agrees_with_surrogate'] = len(agree) / len(rows)
        entry['either_search_given_agree'] = boot(agree, rate(lambda x: x['random_search']['flipped'] or x['coordinate_search']['flipped'])) if agree else None
        out['targeted'][ds][m] = entry
(D / 'comparisons_summary.json').write_text(json.dumps(out, indent=2))
f = lambda t: f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'
print('== Detection (AUROC, top-k recall)')
for ds, d in out['detection'].items():
    print(ds)
    for n, v in d.items():
        if n.startswith('_'):
            print(f"  PAIRED TabPFN-logistic AUROC {f(v['diff'])} @ {v['ci_level']}%"); continue
        print(f"  {n:24s} AUROC {f(v['auroc'])}  recall {f(v['topk_recall'])}  s/audit {v['mean_seconds'] and round(v['mean_seconds'],1)}")
print('== Targeted fragility (flip rate, 24-query search aimed at each receiver)')
for ds, d in out['targeted'].items():
    print(ds)
    for m, v in d.items():
        print(f"  {m:16s} search-either {f(v['either_search'])}  gradient-rows-transfer {f(v['gradient_rows_transfer'])}  agree {v['clean_agrees_with_surrogate']:.2f}")
