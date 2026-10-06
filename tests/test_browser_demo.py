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
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "proofread/browser.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "TabLint"
    assert any(str(suggested if kind == "category" else "70") in item.value for item in app.markdown)


def test_browser_without_repository_fixtures(tmp_path, monkeypatch):
    """An installed wheel must expose upload even when no demo directory exists."""
    monkeypatch.setenv("PROOFREAD_DEMO_DIR", str(tmp_path / "missing"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "proofread/browser.py")).run(timeout=30)
    assert not app.exception
    assert any(element.type == "file_uploader" for element in app.sidebar)
    assert any("Upload a CSV" in element.value for element in app.info)
    assert len(app.tabs) == 3


def test_browser_with_no_flagged_cells(tmp_path, monkeypatch):
    """A valid report with no issues should render without indexing its first row."""
    data = pd.DataFrame({"value": [1.0], "context": [4.0]})
    issues = pd.DataFrame(columns=["kind", "row", "column", "value", "suggested",
                                   "low", "high", "surprise", "cause", "evidence"])
    report = Report(data, issues, pd.DataFrame(np.zeros(data.shape), columns=data.columns),
                    None, {"title": "No flags", "columns_checked": 1})
    report.save(tmp_path / "fixture.json")
    monkeypatch.setenv("PROOFREAD_DEMO_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "proofread/browser.py")).run(timeout=30)
    assert not app.exception
    assert any("No issues meet" in element.value for element in app.info)


def test_cli_app_uses_installed_browser(monkeypatch):
    from proofread import cli
    commands = []
    monkeypatch.setattr(cli.subprocess, "call", lambda command: commands.append(command) or 0)
    with pytest.raises(SystemExit) as result:
        cli.main(["app"])
    assert result.value.code == 0
    target = Path(commands[0][-1])
    assert target == Path(cli.__file__).with_name("browser.py")
    assert target.is_file()
