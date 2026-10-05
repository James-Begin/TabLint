import pandas as pd
import pytest

from proofread.core import Report
from proofread.notebook import report_html, load_ipython_extension


def make_report():
    df = pd.DataFrame({"a": [1.0, 2.0, 30.0, 4.0], "b": [10.0, 0.0, 30.0, 40.0], "y": ["x", "y", "x", "x"]})
    iss = pd.DataFrame([
        dict(kind="cell", row=2, column="a", value=30.0, suggested=3.0, low=2.5, high=3.5, surprise=4.2, cause="possible decimal slip (×10)", evidence="b = 30"),
        dict(kind="cell", row=1, column="b", value=0.0, suggested=20.0, low=15.0, high=25.0, surprise=3.1, cause="possible missing value recorded as 0", evidence=""),
        dict(kind="label", row=3, column="y", value="x", suggested="y", low=float("nan"), high=float("nan"), surprise=2.0, cause="model gives 1%", evidence=""),
    ])
    return Report(df, iss, pd.DataFrame(), "y", {"model": "TabPFN-3.5", "patterns": []})


def test_static_html_highlights_flagged_cells():
    h = report_html(make_report())
    assert "underline wavy" in h and "TabPFN-3.5 expects" in h and "row 2 · a" in h
    assert make_report()._repr_html_() == h


def test_accessor_is_registered():
    import proofread  # noqa: F401
    assert hasattr(pd.DataFrame({"a": [1]}), "proofread")
    assert hasattr(pd.DataFrame({"a": [1]}), "tablint")


def test_widget_state_cleaned_and_decisions():
    pytest.importorskip("anywidget")
    w = make_report().widget()
    assert w.columns == ["a", "b", "y"] and len(w.issues) == 3 and w.status == ["open"] * 3
    assert [r["i"] for r in w.rows] == [0, 1, 2, 3]
    w.status = ["accepted", "dismissed", "accepted"]          # what the front end sends after clicks
    c = w.cleaned
    assert c.at[2, "a"] == 3.0 and c.at[1, "b"] == 0.0 and c.at[3, "y"] == "y"
    assert list(w.decisions.decision) == ["accepted", "dismissed", "accepted"]
    w.only_flagged = True
    assert [r["i"] for r in w.rows] == [1, 2, 3]
    assert all(v is None or isinstance(v, (int, float, str, bool)) for r in w.issues for v in r.values())   # JSON-safe (NaN -> None)


def test_magic_registers_and_runs(monkeypatch):
    pytest.importorskip("anywidget"); pytest.importorskip("IPython")
    rep = make_report()
    monkeypatch.setattr("proofread.notebook.ProofreadAccessor.check", lambda self, label=None, **kw: rep)
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda w: shown.append(w))

    class FakeIPython:
        user_ns = {"df": rep.data}
        magics = {}
        def register_magic_function(self, fn, magic_kind, magic_name):
            self.fn, self.name = fn, magic_name
            self.magics[magic_name] = fn
    ip = FakeIPython(); load_ipython_extension(ip)
    assert ip.name == "proofread"
    assert ip.magics["tablint"] is ip.magics["proofread"]
    w = ip.fn("df --label y --threshold 3")
    assert shown and len(w.issues) == 3


def test_cleaned_handles_integer_columns():
    pytest.importorskip("anywidget")
    df = pd.DataFrame({"year": [1970, 1971, 1917, 1975], "w": [1.0, 2.0, 3.0, 4.0]})
    iss = pd.DataFrame([dict(kind="cell", row=2, column="year", value=1917, suggested=1971.9887, low=1969.0, high=1975.0,
                             surprise=5.0, cause="implausible", evidence="")])
    w = Report(df, iss, pd.DataFrame(), None, {"patterns": []}).widget()
    w.status = ["accepted"]
    assert w.cleaned.at[2, "year"] == 1972 and pd.api.types.is_integer_dtype(w.cleaned["year"])


def test_notebook_categorical_accept_and_undo():
    pytest.importorskip("anywidget")
    report = make_report()
    report.issues.loc[2, "kind"] = "category"
    w = report.widget()
    w.status = ["open", "open", "accepted"]
    assert w.cleaned.at[3, "y"] == "y"
    assert w.decisions.iloc[2].cause == "model gives 1%"
    w.status = ["open"] * 3
    assert w.cleaned.at[3, "y"] == "x"
    assert report.data.at[3, "y"] == "x"
