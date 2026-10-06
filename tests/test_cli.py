import pandas as pd

from proofread import Proofreader, Report
from proofread.cli import main
from proofread.core import ISSUE_COLUMNS


def test_custom_json_path_and_nested_exports(tmp_path, monkeypatch, capsys):
    data = pd.DataFrame({"value": [1.0, 2.0]})
    csv = tmp_path / "input.csv"
    data.to_csv(csv, index=False)
    report = Report(data, pd.DataFrame(columns=ISSUE_COLUMNS), pd.DataFrame(), None,
                    {"seconds": 0, "columns_checked": 1, "folds": 2})
    monkeypatch.setattr(Proofreader, "check", lambda self, df, **kw: report)
    out = tmp_path / "exports" / "table"
    custom = tmp_path / "analysis" / "report.json"
    main(["check", str(csv), "--out", str(out), "--report", str(custom)])
    assert Report.load(custom).issues.empty
    assert out.with_suffix(".md").exists()
    assert pd.read_csv(out.with_suffix(".issues.csv")).empty
    assert "<table" in out.with_suffix(".html").read_text()
    assert str(custom) in capsys.readouterr().out
