"""Dataset loading and model settings used by the numerical confirmation runs.

Preserves the original preprocessing, class coding and deterministic sample cap.
Weights and OpenML data are downloaded only when a benchmark runs.
"""
import numpy as np


OPENML_IDS = {
    "MagicTelescope": 1120, "phoneme": 1489,
    "wilt": 40983, "electricity": 151, "diabetes": 37, "banknote": 1462,
    "spambase": 44, "kc1": 1067,
    "steel-plates": 1504, "ilpd": 1480, "blood-transfusion": 1464,
}


def load_any(name):
    """Load a confirmed numerical dataset as float32 features and binary codes."""
    if name == "breast-cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data.data.astype(np.float32), (1 - data.target).astype(int)
    if name not in OPENML_IDS:
        raise ValueError(f"Unknown numerical benchmark dataset: {name}")

    import openml
    dataset = openml.datasets.get_dataset(
        OPENML_IDS[name], download_data=True, download_qualities=False,
        download_features_meta_data=False,
    )
    frame, target, categorical, names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe",
    )
    numeric = [column for column, is_category in zip(names, categorical) if not is_category]
    X = frame[numeric].apply(lambda column: column.astype(float)).to_numpy(np.float32)
    target = target.astype("category")
    if target.nunique() != 2:
        raise ValueError(f"{name}: not binary")
    y = target.cat.codes.to_numpy().astype(int)
    finite = np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]
    if len(y) > 3000:
        sample = np.random.default_rng(0).choice(len(y), 3000, replace=False)
        X, y = X[sample], y[sample]
    return X, y


def factory(name, seed, device):
    """Construct the classifier settings frozen for the label benchmark."""
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "tabpfn_standard":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, device=device, n_estimators=4,
            random_state=seed, ignore_pretraining_limits=True,
        )
    raise ValueError(f"Unknown label benchmark model: {name}")
