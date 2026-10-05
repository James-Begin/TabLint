"""Real-data check: raw Pima diabetes (OpenML 37) has known impossible zeros (plas, pres, skin, insu, mass)."""
import json
from pathlib import Path
import openml
import numpy as np
from proofread import Proofreader

ds = openml.datasets.get_dataset(37, download_data=True, download_qualities=False, download_features_meta_data=False)
df, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format='dataframe')
df['class'] = y.astype(str)
rep = Proofreader(device='cuda:0').check(df, label='class', max_issues=200, threshold=2.0)
iss = rep.issues
cells = iss[iss.kind == 'cell']
impossible = {c: set(np.where(df[c].to_numpy() == 0)[0]) for c in ('plas', 'pres', 'skin', 'insu', 'mass')}
n_imp = sum(len(v) for v in impossible.values())
def is_known(r):
    return r.column in impossible and r.row in impossible[r.column]
for k in (10, 25, 50):
    top = cells.head(k)
    print(f'top-{k} flagged cells that are known impossible zeros: {sum(is_known(r) for r in top.itertuples())}/{len(top)}')
print('known impossible zeros in table:', n_imp, {c: len(v) for c, v in impossible.items()})
print(rep.to_markdown(15))
out = Path('results/famous/pima_zero_checks.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'n_known': n_imp, 'top': [dict(r._asdict()) for r in cells.head(50).itertuples()]}, default=str, indent=2) + '\n')
