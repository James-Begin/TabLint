"""Dataset loaders. All return float32 numpy X (one-hot categoricals), int y in {0,1}."""
from __future__ import annotations

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

# OpenML dataset ids
DATASETS = {
    "credit-g": 31,        # German credit (Statlog), 1000 rows; positive=bad
    "taiwan": 42477,       # default of credit card clients, 30000 rows
    "diabetes": 37,        # Pima Indians diabetes, 768 rows
}


def load(name: str, n_max: int | None = None, seed: int = 0):
    X, y, _ = load_meta(name, n_max=n_max, seed=seed)
    return X, y


def load_meta(name: str, n_max: int | None = None, seed: int = 0):
    """Like `load`, plus metadata: {'names': column names after one-hot,
    'groups': list of column-index arrays, one per original categorical feature,
    'numeric': indices of numeric columns}. Row order and values match `load`."""
    ds = openml.datasets.get_dataset(DATASETS[name], download_data=True,
                                     download_qualities=False, download_features_meta_data=False)
    X, y, cat, cols = ds.get_data(target=ds.default_target_attribute)
    X = pd.DataFrame(X, columns=cols)
    if name == "credit-g":
        y = (y.astype(str) == "bad").astype(int).values
    elif name == "diabetes":
        y = (y.astype(str) == "tested_positive").astype(int).values
    else:
        y = pd.to_numeric(y).astype(int).values
    catcols = [c for c, iscat in zip(cols, cat) if iscat]
    if catcols:
        X = pd.get_dummies(X, columns=catcols, drop_first=False, dtype=float)
    names = list(X.columns)
    groups = []
    for c in catcols:
        idx = [i for i, n in enumerate(names) if n.startswith(f"{c}_")]
        groups.append(np.array(idx))
    in_group = set(i for g in groups for i in g)
    numeric = np.array([i for i in range(len(names)) if i not in in_group])
    X = X.astype(float).values.astype(np.float32)
    if n_max is not None and len(y) > n_max:
        X, _, y, _ = train_test_split(X, y, train_size=n_max, stratify=y, random_state=seed)
    return X, y, {"names": names, "groups": groups, "numeric": numeric}


def split(X, y, test_size=0.3, seed=0):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
