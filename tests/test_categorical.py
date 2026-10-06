from tabpfn.model_loading import ModelSource, get_cache_dir
import numpy as np
import pandas as pd
import pytest
from proofread.core import Proofreader

WEIGHTS = (get_cache_dir() / ModelSource.get_v3_5().default_filename).is_file()


def test_categorical_column_rule():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"name": [f"car {i}" for i in range(60)],                 # unique text -> not categorical
                       "origin": rng.choice(["USA", "Japan", "Europe"], 60),      # text, 3 levels -> categorical
                       "cyl": rng.choice([4, 6, 8], 60),                         # numeric, 3 distinct -> categorical
                       "weight": rng.normal(3000, 500, 60),                      # continuous -> numeric check, not categorical
                       "y": rng.choice(["a", "b"], 60)})
    assert Proofreader(device="cpu").categorical_columns(df, label="y") == ["origin", "cyl"]
    assert Proofreader.categorical_context_columns(df, label="y") == ["name", "origin"]


def test_sorted_category_codes_preserve_missing_and_ignore_declared_order():
    values = pd.Series(pd.Categorical(["sku-z", "sku-a", None, "sku-z"], categories=["sku-z", "sku-a"]))
    actual = Proofreader._categorical_codes(values)
    np.testing.assert_equal(actual, [1.0, 0.0, np.nan, 1.0])
    shuffled = values.iloc[[1, 3, 0, 2]]
    np.testing.assert_equal(Proofreader._categorical_codes(shuffled), actual[[1, 3, 0, 2]])


class RecordingClassifier:
    def __init__(self, calls, categorical_indices):
        self.calls, self.categorical_indices = calls, list(categorical_indices)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.calls.append((self.categorical_indices, X.copy(), y.copy()))
        return self

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


def test_high_cardinality_context_reaches_categorical_classifier(monkeypatch):
    n = 80
    df = pd.DataFrame({"price": np.arange(n, dtype=float), "origin": ["Japan", "USA"] * (n // 2),
                       "sku": [f"SKU-{n - i:03}" for i in range(n)],
                       "description": [f"description {i}" for i in range(n)]})
    df.loc[5, "sku"] = None
    reader = Proofreader(device="cpu", folds=2)
    calls = []
    monkeypatch.setattr(reader, "_clf", lambda idx, **kw: RecordingClassifier(calls, idx))
    assert reader.categorical_columns(df) == ["origin"]
    scores = reader.categorical_scores(df, ["origin"])
    assert np.isfinite(scores["origin"]["surprise"]).all()
    assert len(calls) == 2
    for indices, X, _ in calls:
        assert indices == [1, 2]  # price remains numerical; both unique string inputs are declared categorical
        rows = X[:, 0].astype(int)
        np.testing.assert_equal(X[:, 1], reader._categorical_codes(df.sku)[rows])
        np.testing.assert_equal(X[:, 2], reader._categorical_codes(df.description)[rows])


def test_confirmation_retains_published_context_policy(monkeypatch):
    from benchmarks.categorical_reference import CategoricalConfirmation
    from tabpfn import TabPFNClassifier
    calls = []
    monkeypatch.setattr(TabPFNClassifier, "create_default_for_version",
                        lambda *args, **kw: RecordingClassifier(calls, kw.get("categorical_features_indices") or []))
    df = pd.DataFrame({"price": np.arange(80, dtype=float), "origin": ["Japan", "USA"] * 40,
                       "sku": [f"unique-{i}" for i in range(80)]})
    CategoricalConfirmation(device="cpu", folds=2).categorical_scores(df, ["origin"])
    assert len(calls) == 2
    assert all(indices == [] and X.shape[1] == 1 for indices, X, _ in calls)


def test_high_cardinality_context_reaches_numeric_and_label_models(monkeypatch):
    import torch
    n = 80
    df = pd.DataFrame({"price": np.arange(n, dtype=float) + 10,
                       "quantity": np.arange(n, dtype=float) + 100,
                       "sku": [f"SKU-{n - i:03}" for i in range(n)],
                       "customer": pd.Categorical(np.arange(n)),
                       "active": [True, False] * (n // 2), "label": ["a", "b"] * (n // 2)})
    df.loc[5, "sku"] = None
    reader = Proofreader(device="cpu", folds=2)
    regressions, classifications = [], []

    class Criterion:
        def cdf(self, logits, values):
            return torch.full_like(values, 0.5)

    class Regressor:
        def __init__(self, idx, **kw):
            self.idx = list(idx)

        def fit(self, X, y):
            regressions.append((self.idx, X.copy(), y.copy()))
            return self

        def predict(self, X, output_type):
            assert output_type == "full"
            return {"logits": torch.zeros(len(X), 1), "criterion": Criterion(),
                    "median": np.full(len(X), 42.0), "quantiles": [np.full(len(X), i) for i in range(9)]}

    monkeypatch.setattr(reader, "_reg", Regressor)
    monkeypatch.setattr(reader, "_clf", lambda idx, **kw: RecordingClassifier(classifications, idx))
    report = reader.check(df, columns=["price"], label="label", categorical=False, threshold=99)
    assert report.meta["columns_checked"] == 1
    assert report.meta["categorical_columns_checked"] == 0
    assert report.meta["categorical_context_columns"] == ["sku", "customer", "active"]
    for indices, X, y in regressions:
        assert indices == [1, 2, 3, 4]  # quantity, then SKU/customer/bool/label; checked price is held out
        rows = (y - 10).astype(int)
        np.testing.assert_equal(X[:, 0], df.quantity.to_numpy()[rows])
        for i, col in enumerate(["sku", "customer", "active", "label"], 1):
            np.testing.assert_equal(X[:, i], reader._categorical_codes(df[col])[rows])
    for indices, X, _ in classifications:
        assert indices == [2, 3, 4]  # price and quantity are numerical; label itself is absent
        rows = (X[:, 0] - 10).astype(int)
        for i, col in enumerate(["sku", "customer", "active"], 2):
            np.testing.assert_equal(X[:, i], reader._categorical_codes(df[col])[rows])


def test_typo_and_rare_hints():
    s = pd.Series(["Japan"] * 20 + ["USA"] * 20 + ["Japna", "Mars"])
    c = Proofreader._category_cause
    assert "possible typo of 'Japan'" in c(s, "Japna", 0.0, "Japan", 0.9)
    assert "rare category" in c(s, "Mars", 0.0, "USA", 0.9)
    assert "TabPFN gives 'USA'" in c(s, "USA", 0.004, "Japan", 0.95)


@pytest.mark.skipif(not WEIGHTS, reason="TabPFN-3.5 weights not downloaded")
def test_finds_planted_category_cpu(cpu_inference_threads):
    rng = np.random.default_rng(1)
    size = rng.uniform(1, 10, 90)
    df = pd.DataFrame({"size": size, "kind": np.where(size > 5.5, "large", "small")})
    df.loc[10, "kind"] = "large" if df.loc[10, "kind"] == "small" else "small"   # contradicts its size
    rep = Proofreader(device="cpu", folds=3).check(df, threshold=1.0)
    cat = rep.issues[rep.issues.kind == "category"]
    assert not cat.empty and (int(cat.iloc[0].row), cat.iloc[0].column) == (10, "kind")


@pytest.mark.skipif(not WEIGHTS, reason="TabPFN-3.5 weights not downloaded")
def test_real_high_cardinality_context_stays_categorical(cpu_inference_threads, monkeypatch):
    from tabpfn.preprocessing.datamodel import FeatureModality
    rng = np.random.default_rng(123)
    n = 240
    prices = np.tile(rng.uniform(10, 50, 40), 6) + rng.normal(0, .05, n)
    df = pd.DataFrame({"sku": np.tile([f"SKU-{i:03}" for i in range(40)], 6), "price": prices,
                       "label": np.tile(["a", "b"] * 20, 6)})
    df.loc[17, "price"] *= 10
    reader = Proofreader(device="cpu", folds=2)
    fitted = []

    def capture(factory, kind):
        def create(indices, **kw):
            model = factory(indices, **kw)
            fitted.append((kind, list(indices), model))
            return model
        return create

    monkeypatch.setattr(reader, "_reg", capture(reader._reg, "regression"))
    monkeypatch.setattr(reader, "_clf", capture(reader._clf, "classification"))
    report = reader.check(df, columns=["price"], label="label", categorical=False)
    assert report.meta["columns_checked"] == 1
    assert ((report.issues.row == 17) & (report.issues.column == "price")).any()
    assert {kind for kind, _, _ in fitted} == {"regression", "classification"}
    for kind, indices, model in fitted:
        inferred = model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL)
        assert inferred == indices
        assert model.inference_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES >= n
        assert indices == ([0, 1] if kind == "regression" else [1])
