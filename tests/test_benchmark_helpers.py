"""Keep the confirmation preprocessing and model settings stable after cleanup."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.common import factory, load_any


def test_builtin_dataset_keeps_original_class_coding():
    from sklearn.datasets import load_breast_cancer
    original = load_breast_cancer()
    X, y = load_any("breast-cancer")
    np.testing.assert_array_equal(X, original.data.astype(np.float32))
    np.testing.assert_array_equal(y, 1 - original.target)


def test_openml_preprocessing_and_cap(monkeypatch):
    n = 3010
    frame = pd.DataFrame({"numeric": np.arange(n, dtype=float), "category": ["a"] * n})
    frame.loc[2, "numeric"] = np.inf
    target = pd.Series(["negative", "positive"] * (n // 2))
    dataset = SimpleNamespace(default_target_attribute="target",
        get_data=lambda **kwargs: (frame, target, [False, True], list(frame.columns)))
    requests = []
    def get_dataset(identifier, **kwargs):
        requests.append((identifier, kwargs))
        return dataset
    monkeypatch.setitem(sys.modules, "openml", SimpleNamespace(datasets=SimpleNamespace(get_dataset=get_dataset)))
    X, y = load_any("diabetes")
    finite = np.isfinite(frame.numeric.to_numpy())
    sample = np.random.default_rng(0).choice(n - 1, 3000, replace=False)
    np.testing.assert_array_equal(X[:, 0], frame.numeric.to_numpy(np.float32)[finite][sample])
    np.testing.assert_array_equal(y, target.astype("category").cat.codes.to_numpy()[finite][sample])
    assert X.shape == (3000, 1)
    assert X.dtype == np.float32
    assert requests == [(37, dict(download_data=True, download_qualities=False, download_features_meta_data=False))]


def test_baseline_classifier_settings():
    assert factory("rf", 701, "cpu").get_params()["n_estimators"] == 100
    assert factory("rf", 701, "cpu").get_params()["n_jobs"] == 1
    assert factory("hgb", 701, "cpu").get_params()["random_state"] == 701
    assert factory("logistic", 701, "cpu")[-1].get_params()["max_iter"] == 2000


def test_tabpfn_classifier_settings_without_weights(monkeypatch):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    calls = []
    monkeypatch.setattr(TabPFNClassifier, "create_default_for_version", lambda version, **kwargs: calls.append((version, kwargs)))
    factory("tabpfn_standard", 701, "cuda:0")
    assert calls == [(ModelVersion.V3_5, dict(device="cuda:0", n_estimators=4,
                                           random_state=701, ignore_pretraining_limits=True))]


def test_unknown_dataset_fails_before_network_access():
    with pytest.raises(ValueError, match="Unknown numerical"):
        load_any("unknown")
