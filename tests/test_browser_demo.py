"""Exercise the saved-report browser workflow without model weights."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from proofread.core import Report

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("kind,value,suggested", [("category", "USA", "Japan"), ("cell", 0.0, 70.0)])
def test_saved_browser_suggestion(tmp_path, monkeypatch, kind, value, suggested):
    data = pd.DataFrame({"value": [value], "context": [4]})
    issues = pd.DataFrame([dict(kind=kind, row=0, column="value", value=value,
                               suggested=suggested, low=69.0, high=75.0,
                               surprise=3.0, cause="Review the source record", evidence="context = 4")])
    report = Report(data, issues, pd.DataFrame(np.zeros(data.shape), columns=data.columns),
                    None, {"title": "Saved fixture", "columns_checked": 1})
    report.save(tmp_path / "fixture.json")
    monkeypatch.setenv("PROOFREAD_DEMO_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "demo/proofread_app.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "TabLint"
    assert any(str(suggested if kind == "category" else "70") in item.value for item in app.markdown)
