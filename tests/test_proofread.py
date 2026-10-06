import numpy as np
import pandas as pd
import pytest

from proofread.core import Proofreader, Report, _transpose


def test_cause_names_common_slips():
    c = Proofreader._cause
    assert "×10" in c(1830.0, 150, 220)
    assert "÷10" in c(18.3, 150, 220)
    assert "0" in c(0.0, 20, 40)
    assert "swapped leading digits" in c(81.0, 15, 25)
    assert c(500.0, 150, 220).startswith("implausible")
    # zero-inflated column: 12.2 is inside a wide band but not close to the median -> no slip claimed
    assert c(122.0, 5, 80, 64).startswith("implausible")
    assert "placeholder" in c(99.0, 0, 35, 0.05, placeholder_column=True)


def test_transpose():
    assert _transpose(183.0) == pytest.approx(813.0)
    assert _transpose(11.0) is None


def test_patterns_flag_placeholder_codes():
    rng = np.random.default_rng(0)
    v = rng.normal(100, 10, 200); v[:40] = 0
    pats = Proofreader._patterns(pd.DataFrame({"insulin": v, "age": rng.normal(40, 5, 200)}), ["insulin", "age"])
    assert [p["column"] for p in pats] == ["insulin"]


def test_report_roundtrip_and_markdown(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 30.0], "y": ["x", "y", "x"]})
    iss = pd.DataFrame([dict(kind="cell", row=2, column="a", value=30.0, suggested=3.0, low=1.0, high=4.0, surprise=4.2,
                             cause="possible decimal slip", evidence="b")])
    rep = Report(df, iss, pd.DataFrame({"a": [0.1, 0.2, 4.2]}), "y", {"columns_checked": 1, "folds": 5, "patterns": []})
    p = rep.save(tmp_path / "r.json")
    back = Report.load(p)
    assert back.issues.iloc[0]["column"] == "a" and "decimal slip" in back.to_markdown()
