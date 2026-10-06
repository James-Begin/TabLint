"""End-to-end: real TabPFN-3.5 on CPU finds a planted decimal slip. Skipped if weights are not cached."""
from tabpfn.model_loading import ModelSource, get_cache_dir
import numpy as np
import pandas as pd
import pytest

WEIGHTS = (get_cache_dir() / ModelSource.get_v3_5().default_filename).is_file()


@pytest.mark.skipif(not WEIGHTS, reason="TabPFN-3.5 weights not downloaded (accept the license once to enable)")
def test_finds_planted_decimal_slip(cpu_inference_threads):
    from proofread import Proofreader
    rng = np.random.default_rng(0)
    a = rng.uniform(10, 20, 80)
    df = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(0, 0.3, 80), "c": rng.normal(5, 1, 80)})
    df.loc[17, "b"] = df.loc[17, "b"] * 10            # 2a ≈ 30 → ≈ 300; ordinary for nothing, so easy
    df.loc[41, "a"] = df.loc[41, "a"] * 10            # a ≈ 15 → 150
    rep = Proofreader(device="cpu", folds=3).check(df, threshold=2.0)
    top = {(r.row, r.column) for r in rep.issues.head(2).itertuples()}
    assert top == {(17, "b"), (41, "a")}
    assert "decimal slip" in rep.issues.iloc[0]["cause"]
