"""`proofread view` — a terminal spreadsheet with TabPFN-3.5 proofreading on top.

    proofread view data.csv --label outcome     # runs the check (GPU if available), then opens the viewer
    proofread view report.json                  # opens a saved report instantly (no model, no GPU)

Keys: n / p next / previous issue · a accept suggestion · d dismiss · u undo · f only rows with issues
      s save cleaned CSV + decision log · ? help · q quit
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.widgets import DataTable, Footer, Header, Static

from .core import Report, set_cell

STYLE = {"open_cell": "bold #ffffff on #a40000", "open_label": "bold #000000 on #ffb000",
         "accepted": "bold #000000 on #3fb950", "dismissed": "dim strike"}


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(v):
            return "NaN"
        return f"{v:.6g}"
    return str(v)


def range_bar(value: float, low: float, high: float, suggested: float, width: int = 36) -> Text:
    """ASCII strip: TabPFN's 80% plausible range as a band, the suggestion (◆) and the recorded value (▲)."""
    span_lo, span_hi = min(low, value, suggested), max(high, value, suggested)
    pad = (span_hi - span_lo) * 0.05 or 1.0
    span_lo, span_hi = span_lo - pad, span_hi + pad
    pos = lambda x: int(round((x - span_lo) / (span_hi - span_lo) * (width - 1)))
    cells = [" "] * width
    lo_i, hi_i = pos(low), pos(high)
    if hi_i - lo_i < 2:                      # keep a narrow range visible
        lo_i, hi_i = max(0, lo_i - 1), min(width - 1, hi_i + 1)
    for i in range(lo_i, hi_i + 1):
        cells[i] = "█"
    t = Text()
    for i, ch in enumerate(cells):
        if i == pos(value):
            t.append("▲", style="bold red")
        elif i == pos(suggested):
            t.append("◆", style="bold green")
        else:
            t.append(ch, style="#4c78a8")
    return t


class ProofreadViewer(App):
    """Spreadsheet view of a table with Proofread issues overlaid."""

    TITLE = "Proofread"
    CSS = """
    #main { height: 1fr; }
    DataTable { width: 3fr; }
    #side { width: 50; padding: 0 1; border-left: tall $accent; }
    #detail { height: auto; }
    #summary { height: auto; color: $text-muted; padding-top: 1; }
    """
    BINDINGS = [
        Binding("n", "next_issue", "next issue"), Binding("p", "prev_issue", "prev issue"),
        Binding("a", "accept", "accept fix"), Binding("d", "dismiss", "dismiss"), Binding("u", "undo", "undo"),
        Binding("f", "toggle_filter", "only flagged rows"), Binding("s", "save", "save"),
        Binding("question_mark", "help", "help"), Binding("q", "quit", "quit"),
    ]

    def __init__(self, report: Report, source: str = "table", out_prefix: str | None = None):
        super().__init__()
        self.report = report
        self.source = source
        self.out_prefix = out_prefix or str(Path(source).with_suffix(""))
        self.data = report.data.copy()
        self.columns = list(self.data.columns)
        iss = report.issues.reset_index(drop=True)
        self.issues = [dict(r) for _, r in iss.iterrows()]
        self.status = ["open"] * len(self.issues)
        self.by_cell = {(int(r["row"]), str(r["column"])): k for k, r in enumerate(self.issues)}
        self.history: list[tuple[int, str, object]] = []   # (issue index, previous status, previous value)
        self.only_flagged = False
        self.row_order: list[int] = []
        self.current = 0

    # -- layout ---------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            yield DataTable(id="grid", zebra_stripes=True, cursor_type="cell")
            with Vertical(id="side"):
                yield Static(id="detail")
                yield Static(id="summary")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self.source} · {len(self.data)} rows · {len(self.issues)} issues · " \
                         f"{self.report.meta.get('model', 'TabPFN-3.5')}"
        grid = self.query_one(DataTable)
        # Pin the row number, plus the first column if it is text (e.g. a name), so context stays visible when scrolling.
        first_is_text = bool(self.columns) and not pd.api.types.is_numeric_dtype(self.data[self.columns[0]])
        grid.fixed_columns = 2 if first_is_text else 1
        grid.add_column("row", key="__row__")
        for c in self.columns:
            grid.add_column(str(c), key=str(c))
        self._fill()
        if self.issues:
            self._goto(0)
        self._refresh_side()

    def _cell(self, i: int, c: str):
        k = self.by_cell.get((i, c))
        txt = _fmt(self.data.at[i, c])
        if k is None:
            return txt
        st = self.status[k]
        style = STYLE["accepted"] if st == "accepted" else STYLE["dismissed"] if st == "dismissed" else \
            STYLE["open_label"] if self.issues[k]["kind"] != "cell" else STYLE["open_cell"]
        return Text(txt, style=style)

    def _fill(self) -> None:
        grid = self.query_one(DataTable)
        grid.clear()
        flagged = sorted({int(r["row"]) for r in self.issues})
        self.row_order = flagged if self.only_flagged else list(range(len(self.data)))
        for i in self.row_order:
            grid.add_row(Text(str(i), style="dim"), *[self._cell(i, str(c)) for c in self.columns], key=str(i))

    def _redraw_cell(self, i: int, c: str) -> None:
        grid = self.query_one(DataTable)
        if i in self.row_order:
            grid.update_cell(str(i), c, self._cell(i, c))

    # -- navigation -----------------------------------------------------------
    def _goto(self, k: int) -> None:
        if not self.issues:
            return
        self.current = k % len(self.issues)
        r = self.issues[self.current]
        i, c = int(r["row"]), str(r["column"])
        if i not in self.row_order:
            self.only_flagged = False
            self._fill()
        grid = self.query_one(DataTable)
        grid.move_cursor(row=self.row_order.index(i), column=self.columns.index(c) + 1, animate=False)
        self._refresh_side()

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        row_i = self.row_order[event.coordinate.row] if event.coordinate.row < len(self.row_order) else None
        col = self.columns[event.coordinate.column - 1] if event.coordinate.column >= 1 else None
        k = self.by_cell.get((row_i, str(col))) if row_i is not None and col is not None else None
        if k is not None:
            self.current = k
        self._refresh_side(cursor=(row_i, col), issue=k)

    def action_next_issue(self) -> None:
        self._goto(self.current + 1)

    def action_prev_issue(self) -> None:
        self._goto(self.current - 1)

    def action_toggle_filter(self) -> None:
        self.only_flagged = not self.only_flagged
        self._fill()
        if self.issues:
            self._goto(self.current)

    # -- decisions ------------------------------------------------------------
    def _issue_under_cursor(self):
        grid = self.query_one(DataTable)
        co = grid.cursor_coordinate
        if co.column < 1 or co.row >= len(self.row_order):
            return None
        return self.by_cell.get((self.row_order[co.row], str(self.columns[co.column - 1])))

    def action_accept(self) -> None:
        k = self._issue_under_cursor()
        if k is None or self.status[k] == "accepted":
            return
        r = self.issues[k]; i, c = int(r["row"]), str(r["column"])
        self.history.append((k, self.status[k], self.data.at[i, c]))
        new = float(r["suggested"]) if r["kind"] == "cell" else r["suggested"]
        set_cell(self.data, i, c, new)
        self.status[k] = "accepted"
        self._redraw_cell(i, c); self._next_open()

    def action_dismiss(self) -> None:
        k = self._issue_under_cursor()
        if k is None or self.status[k] == "dismissed":
            return
        r = self.issues[k]; i, c = int(r["row"]), str(r["column"])
        self.history.append((k, self.status[k], self.data.at[i, c]))
        self.status[k] = "dismissed"
        self._redraw_cell(i, c); self._next_open()

    def action_undo(self) -> None:
        if not self.history:
            return
        k, st, val = self.history.pop()
        r = self.issues[k]; i, c = int(r["row"]), str(r["column"])
        set_cell(self.data, i, c, val); self.status[k] = st
        self._redraw_cell(i, c); self._goto(k)

    def _next_open(self) -> None:
        for step in range(1, len(self.issues) + 1):
            k = (self.current + step) % len(self.issues)
            if self.status[k] == "open":
                self._goto(k); return
        self._refresh_side()

    def action_save(self) -> None:
        cleaned = Path(f"{self.out_prefix}.cleaned.csv")
        log = Path(f"{self.out_prefix}.decisions.json")
        self.data.to_csv(cleaned, index=False)
        decisions = [{"row": int(r["row"]), "column": str(r["column"]), "kind": r["kind"], "recorded": _fmt(self.report.data.at[int(r["row"]), r["column"]]),
                      "suggested": _fmt(r["suggested"]), "surprise": round(float(r["surprise"]), 3), "cause": r["cause"], "decision": s}
                     for r, s in zip(self.issues, self.status)]
        log.write_text(json.dumps({"source": self.source, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "model": self.report.meta.get("model", "TabPFN-3.5"), "decisions": decisions}, indent=2))
        self.notify(f"saved {cleaned.name} and {log.name}", timeout=4)

    def action_help(self) -> None:
        self.notify("n/p next/prev issue · a accept · d dismiss · u undo · f flagged rows only · s save · q quit", timeout=8)

    # -- side panel -----------------------------------------------------------
    def _refresh_side(self, cursor=None, issue=None) -> None:
        detail = self.query_one("#detail", Static)
        k = issue if cursor is not None else (self.current if self.issues else None)
        if k is None:
            i, c = cursor if cursor else (None, None)
            body = Text("No issue in this cell.\n", style="dim")
            if i is not None and c is not None:
                body.append(f"\nrow {i} · {c} = {_fmt(self.data.at[i, c])}")
            detail.update(body)
        else:
            r = self.issues[k]; st = self.status[k]
            t = Text()
            t.append(f"Issue {k + 1}/{len(self.issues)}  ", style="bold")
            t.append(f"[{st}]\n", style={"open": "bold red", "accepted": "bold green", "dismissed": "dim"}[st])
            t.append(f"row {int(r['row'])} · {r['column']}\n\n", style="bold")
            t.append("recorded   "); t.append(f"{_fmt(self.report.data.at[int(r['row']), r['column']])}\n", style="bold red")
            if r["kind"] == "cell":
                t.append("TabPFN     "); t.append(f"{float(r['suggested']):.4g}", style="bold green")
                t.append(f"  (80%: {float(r['low']):.4g}–{float(r['high']):.4g})\n\n")
                t.append_text(range_bar(float(r["value"]), float(r["low"]), float(r["high"]), float(r["suggested"])))
                t.append("\n▲ recorded  ◆ suggested  █ plausible\n\n", style="dim")
            else:
                t.append("TabPFN     "); t.append(f"{r['suggested']}\n\n", style="bold green")
            t.append("surprise   "); t.append(f"{float(r['surprise']):.1f}", style="bold")
            t.append("  (−log₁₀ tail probability)\n", style="dim")
            t.append("cause      "); t.append(f"{r['cause']}\n")
            if r.get("evidence"):
                t.append("\nrelated in this row:\n", style="dim"); t.append(f"{r['evidence']}\n")
            detail.update(t)
        n_open = self.status.count("open"); n_acc = self.status.count("accepted"); n_dis = self.status.count("dismissed")
        summ = Text(f"{n_open} open · {n_acc} accepted · {n_dis} dismissed\n", style="bold")
        for p in self.report.meta.get("patterns", []) or []:
            summ.append(f"\n⚑ {p['column']}: {p['value']:g} in {p['share']:.0%} of rows (likely placeholder code)", style="#ffb000")
        summ.append("\n\nA flag is a prompt to check the source, not proof of an error.", style="dim")
        self.query_one("#summary", Static).update(summ)


def run(path: str, label: str | None = None, fast: bool = False, threshold: float = 2.0, max_issues: int = 200,
        device: str = "auto") -> None:
    p = Path(path)
    if p.suffix == ".json":
        rep = Report.load(p)
        source = rep.meta.get("source_file", p.name)
        prefix = str(p.with_suffix(""))
    else:
        from .core import Proofreader
        df = pd.read_csv(p)
        print(f"Proofreading {len(df)} rows × {df.shape[1]} columns with TabPFN-3.5 ... (saved reports open instantly)", flush=True)
        rep = Proofreader(device=device, fast=fast).check(df, label=label, threshold=threshold, max_issues=max_issues)
        rep.meta["source_file"] = p.name
        rep.save(p.with_suffix(".proofread.json"))
        print(f"report saved to {p.with_suffix('.proofread.json')}", flush=True)
        source, prefix = p.name, str(p.with_suffix(""))
    ProofreadViewer(rep, source=source, out_prefix=prefix).run()
