"""Headless test of the terminal viewer: navigate, accept, dismiss, undo, save."""
import asyncio
import json

import pandas as pd
import pytest

pytest.importorskip("textual")
from proofread.core import Report
from proofread.tui import ProofreadViewer, range_bar


def make_report():
    df = pd.DataFrame({"a": [1.0, 2.0, 30.0, 4.0], "b": [10.0, 0.0, 30.0, 40.0], "y": ["x", "y", "x", "x"]})
    iss = pd.DataFrame([
        dict(kind="cell", row=2, column="a", value=30.0, suggested=3.0, low=2.5, high=3.5, surprise=4.2, cause="possible decimal slip (×10)", evidence="b = 30"),
        dict(kind="cell", row=1, column="b", value=0.0, suggested=20.0, low=15.0, high=25.0, surprise=3.1, cause="possible missing value recorded as 0", evidence=""),
        dict(kind="label", row=3, column="y", value="x", suggested="y", low=float("nan"), high=float("nan"), surprise=2.0, cause="model gives 1%", evidence=""),
    ])
    return Report(df, iss, pd.DataFrame(), "y", {"model": "TabPFN-3.5", "patterns": []})


def test_range_bar_marks_value_and_suggestion():
    t = range_bar(30.0, 2.5, 3.5, 3.0).plain
    assert "▲" in t and "◆" in t and "█" in t


def test_viewer_accept_dismiss_undo_save(tmp_path):
    async def go():
        app = ProofreadViewer(make_report(), source="t.csv", out_prefix=str(tmp_path / "t"))
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            assert app.current == 0
            assert app.query_one("DataTable").fixed_columns == 1          # numeric first column: pin only "row"
            await pilot.press("a")                 # accept issue 1: a[2] 30 -> 3
            assert app.data.at[2, "a"] == 3.0 and app.status[0] == "accepted"
            assert app.current == 1                # jumped to the next open issue
            await pilot.press("d")                 # dismiss issue 2
            assert app.status[1] == "dismissed" and app.data.at[1, "b"] == 0.0
            await pilot.press("u")                 # undo dismissal
            assert app.status[1] == "open"
            await pilot.press("n"); await pilot.press("a")   # label issue: x -> y
            assert app.data.at[3, "y"] == "y"
            await pilot.press("f")                 # only flagged rows
            assert app.row_order == [1, 2, 3]
            await pilot.press("s")
        return app
    asyncio.run(go())
    cleaned = pd.read_csv(tmp_path / "t.cleaned.csv")
    assert cleaned.loc[2, "a"] == 3.0 and cleaned.loc[3, "y"] == "y" and cleaned.loc[1, "b"] == 0.0
    log = json.loads((tmp_path / "t.decisions.json").read_text())
    assert [d["decision"] for d in log["decisions"]] == ["accepted", "open", "accepted"]
    assert log["decisions"][0]["recorded"] == "30"


def test_accept_on_integer_column(tmp_path):
    df = pd.DataFrame({"year": [1970, 1971, 1917, 1975], "w": [1.0, 2.0, 3.0, 4.0]})
    iss = pd.DataFrame([dict(kind="cell", row=2, column="year", value=1917, suggested=1971.9887, low=1969.0, high=1975.0,
                             surprise=5.0, cause="implausible", evidence="")])
    rep = Report(df, iss, pd.DataFrame(), None, {"patterns": []})

    async def go():
        app = ProofreadViewer(rep, source="t.csv", out_prefix=str(tmp_path / "t"))
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(); await pilot.press("a")
            assert app.data.at[2, "year"] == 1972
            await pilot.press("u")
            assert app.data.at[2, "year"] == 1917
    asyncio.run(go())
