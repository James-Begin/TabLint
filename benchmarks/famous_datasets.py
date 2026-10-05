"""Exploratory scan of famous datasets (protocol fixed in docs/FAMOUS_DATASETS_PREREG.md).
Writes results/famous/<name>.proofread.json and results/famous/top10.json (top-10 flags with full rows + auto checks)."""
import io, json, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from proofread import Proofreader

OUT = Path('results/famous'); OUT.mkdir(parents=True, exist_ok=True)
SB = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{}.csv'


def seaborn(name):
    return pd.read_csv(io.StringIO(urllib.request.urlopen(SB.format(name), timeout=60).read().decode()))


def uci_iris():
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://archive.ics.uci.edu/static/public/53/iris.zip', timeout=60).read()))
    df = pd.read_csv(io.StringIO(z.read('iris.data').decode()), header=None,
                     names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']).dropna()
    return df.reset_index(drop=True)


def california():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(as_frame=True)
    return d.frame


def sample_rule(df):
    """>10,000 rows: fixed 10k sample (seed 0) plus every row with an 'impossible zero' (0 in a column whose
    non-zero minimum is > 0). Returns the sampled frame with original row ids kept in column '_orig_row'."""
    if len(df) <= 10_000:
        return df.reset_index(drop=True).assign(_orig_row=np.arange(len(df)))
    num = df.select_dtypes('number')
    zero_cols = [c for c in num if (num[c] == 0).any() and num.loc[num[c] != 0, c].min() > 0]
    zero_rows = set(np.where((num[zero_cols] == 0).any(axis=1))[0]) if zero_cols else set()
    rows = sorted(set(np.random.default_rng(0).choice(len(df), 10_000, replace=False)) | zero_rows)
    return df.iloc[rows].reset_index(drop=True).assign(_orig_row=rows)


def checks(name, row):
    """Automatic, definitional checks used to support a 'verified error' label (documented in the gallery)."""
    out = []
    if name == 'diamonds':
        if min(row['x'], row['y'], row['z']) == 0:
            out.append('a physical dimension (x, y or z) is 0 mm')
        if row['x'] > 0 and row['y'] > 0:
            implied = 2 * row['z'] / (row['x'] + row['y']) * 100
            if abs(implied - row['depth']) > 5:
                out.append(f"depth {row['depth']} ≠ 2z/(x+y) = {implied:.1f} (definition of depth)")
    if name == 'taxis':
        parts = row['fare'] + row['tip'] + row['tolls']
        if abs(row['total'] - parts) > 5:
            out.append(f"total {row['total']} vs fare+tip+tolls = {parts:.2f} (surcharges are a few dollars)")
    return out


DATASETS = {
    'penguins': (lambda: seaborn('penguins'), 'species'),
    'tips': (lambda: seaborn('tips'), None),
    'titanic': (lambda: seaborn('titanic')[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']], 'survived'),
    'taxis': (lambda: seaborn('taxis'), None),
    'planets': (lambda: seaborn('planets'), 'method'),
    'diamonds': (lambda: seaborn('diamonds'), None),
    'california_housing': (california, None),
    'iris_uci': (uci_iris, 'species'),
}

if __name__ == '__main__':
    import sys
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else list(DATASETS)
    pr = Proofreader(device='cuda:0')
    summary = json.loads((OUT / 'top10.json').read_text()) if (OUT / 'top10.json').exists() else {}
    for name in names:
        load, label = DATASETS[name]
        full = load(); df = sample_rule(full)
        rep = pr.check(df.drop(columns=['_orig_row']), label=label, threshold=2.0, max_issues=60)
        rep.meta.update(title=f'{name} (scan)', source=name, n_full=len(full), n_scanned=len(df))
        rep.save(OUT / f'{name}.proofread.json')
        top = []
        for r in rep.issues.head(10).itertuples():
            row = df.iloc[int(r.row)].drop(labels=['_orig_row']).to_dict()
            top.append({'rank': len(top) + 1, 'orig_row': int(df.at[int(r.row), '_orig_row']), 'kind': r.kind, 'column': r.column,
                        'value': r.value, 'suggested': r.suggested, 'low': r.low, 'high': r.high, 'surprise': round(float(r.surprise), 2),
                        'cause': r.cause, 'auto_checks': checks(name, row), 'row': {k: (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in row.items()}})
        extra = {}
        if name == 'iris_uci':    # pre-specified known-answer check (0-based rows 34 and 37)
            order = [(int(r.row), str(r.column)) for r in rep.issues.itertuples()]
            for rr, c in ((34, 'petal_width'), (37, 'sepal_width'), (37, 'petal_length')):
                extra[f'sample {rr + 1} {c}'] = {'rank': order.index((rr, c)) + 1 if (rr, c) in order else None,
                                                 'surprise': round(float(rep.cell_surprise.at[rr, c]), 2), 'value': float(df.at[rr, c])}
        summary[name] = {'n_full': len(full), 'n_scanned': len(df), 'issues': len(rep.issues), 'seconds': round(rep.meta['seconds']),
                         'patterns': rep.meta.get('patterns', []), 'top10': top, 'known_answer': extra}
        (OUT / 'top10.json').write_text(json.dumps(summary, indent=1, default=str))
        print(f"== {name}: {len(full)} rows (scanned {len(df)}), {len(rep.issues)} issues, {rep.meta['seconds']:.0f}s, patterns {[p['column'] for p in rep.meta.get('patterns', [])]}", flush=True)
        for t in top:
            print(f"  {t['rank']:2d} {t['surprise']:4.1f} row {t['orig_row']:6d} {t['column']:14s} {str(t['value'])[:10]:>10} -> {str(t['suggested'])[:9]:>9} | {t['cause'][:60]} | {t['auto_checks']}", flush=True)
        if extra:
            print('  known-answer check:', extra, flush=True)
