from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from proofread.core import Proofreader

WEIGHTS = list((Path.home() / ".cache" / "tabpfn").glob("tabpfn-v3.5-*.safetensors"))


def test_categorical_column_rule():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"name": [f"car {i}" for i in range(60)],                 # unique text -> not categorical
                       "origin": rng.choice(["USA", "Japan", "Europe"], 60),      # text, 3 levels -> categorical
                       "cyl": rng.choice([4, 6, 8], 60),                         # numeric, 3 distinct -> categorical
                       "weight": rng.normal(3000, 500, 60),                      # continuous -> numeric check, not categorical
                       "y": rng.choice(["a", "b"], 60)})
    assert Proofreader(device="cpu").categorical_columns(df, label="y") == ["origin", "cyl"]


def test_typo_and_rare_hints():
    s = pd.Series(["Japan"] * 20 + ["USA"] * 20 + ["Japna", "Mars"])
    c = Proofreader._category_cause
    assert "possible typo of 'Japan'" in c(s, "Japna", 0.0, "Japan", 0.9)
    assert "rare category" in c(s, "Mars", 0.0, "USA", 0.9)
    assert "TabPFN gives 'USA'" in c(s, "USA", 0.004, "Japan", 0.95)


@pytest.mark.skipif(not WEIGHTS, reason="TabPFN-3.5 weights not downloaded")
def test_finds_planted_category_cpu():
    rng = np.random.default_rng(1)
    size = rng.uniform(1, 10, 90)
    df = pd.DataFrame({"size": size, "kind": np.where(size > 5.5, "large", "small")})
    df.loc[10, "kind"] = "large" if df.loc[10, "kind"] == "small" else "small"   # contradicts its size
    rep = Proofreader(device="cpu", folds=3).check(df, threshold=1.0)
    cat = rep.issues[rep.issues.kind == "category"]
    assert not cat.empty and (int(cat.iloc[0].row), cat.iloc[0].column) == (10, "kind")
