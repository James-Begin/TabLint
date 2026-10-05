import numpy as np


def load_any(ds):
    from auditkit.cli import dataset
    if ds in ('breast-cancer', 'credit-g', 'taiwan'):
        X, y, _ = dataset(ds)
    else:
        from experiments.screen_regimes import load
        X, y, _ = load(ds)
    return X.astype(np.float32), np.asarray(y).astype(int)
