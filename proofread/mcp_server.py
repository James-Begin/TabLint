"""MCP tools for an LLM data-cleaning agent.

Replay tools read stored reports (no model needed). `proofread_csv` runs TabPFN-3.5 live and is disabled unless
PROOFREAD_ALLOW_INFERENCE=1 (explicit opt-in; serialized; CSV size-capped).
"""
import json
import os
import threading
from pathlib import Path

REPORTS = Path(os.environ.get("PROOFREAD_REPORT_DIR", "results/proofread_demos")).resolve()
BENCH = Path(os.environ.get("PROOFREAD_BENCHMARK", "results/proofread/summary.json")).resolve()
_LOCK = threading.Lock()


def _path(report_id: str) -> Path:
    p = (REPORTS / Path(report_id).name).resolve()
    if not p.is_relative_to(REPORTS) or p.suffix != ".json" or not p.is_file():
        raise ValueError("unknown report")
    return p


def list_reports() -> list:
    """Stored Proofread reports with titles and issue counts."""
    from .core import Report
    out = []
    for p in sorted(REPORTS.glob("*.json")):
        try:
            r = Report.load(p)
        except Exception:
            continue
        out.append({"report_id": p.name, "title": r.meta.get("title", p.stem), "rows": len(r.data),
                    "cell_issues": int((r.issues.kind == "cell").sum()), "category_issues": int((r.issues.kind == "category").sum()),
                    "label_issues": int((r.issues.kind == "label").sum())})
    return out


def get_issues(report_id: str, top: int = 20) -> dict:
    """Most suspicious cells/labels: recorded value, suggestion, 80% range, surprise, likely cause, evidence."""
    from .core import Report
    r = Report.load(_path(report_id))
    return {"report_id": report_id, "label": r.label, "column_patterns": r.meta.get("patterns", []),
            "issues": json.loads(r.issues.head(max(1, min(int(top), 200))).to_json(orient="records")),
            "note": "Flags are prompts to check the source record, not proof of error."}


def get_row(report_id: str, row: int) -> dict:
    """The full recorded row, so the agent can reason about the flagged cell in context."""
    from .core import Report
    r = Report.load(_path(report_id))
    return {"row": int(row), "values": json.loads(r.data.iloc[[int(row)]].to_json(orient="records"))[0]}


def report_markdown(report_id: str, top: int = 15) -> str:
    """Human-readable Proofread report."""
    from .core import Report
    return Report.load(_path(report_id)).to_markdown(int(top))


def benchmark_summary() -> dict:
    """Pre-registered benchmark results (cells: H6, labels: H7) as stored."""
    return json.loads(BENCH.read_text()) if BENCH.is_file() else {"error": "benchmark summary not found"}


def proofread_csv(csv_path: str, label: str = "", report_name: str = "agent_check") -> dict:
    """Run TabPFN-3.5 on a CSV (opt-in). Max 2000 rows × 60 columns."""
    if os.environ.get("PROOFREAD_ALLOW_INFERENCE") != "1":
        raise ValueError("live inference disabled; set PROOFREAD_ALLOW_INFERENCE=1")
    import pandas as pd
    from .core import Proofreader
    df = pd.read_csv(csv_path)
    if len(df) > 2000 or df.shape[1] > 60:
        raise ValueError("table too large for the agent tool")
    with _LOCK:
        rep = Proofreader(device=os.environ.get("PROOFREAD_DEVICE", "auto")).check(df, label=label or None)
    name = Path(report_name).name.removesuffix(".json") + ".json"
    rep.save(REPORTS / name)
    return {"report_id": name, "cell_issues": int((rep.issues.kind == "cell").sum()), "label_issues": int((rep.issues.kind == "label").sum())}


def main():
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("Proofread")
    for tool in (list_reports, get_issues, get_row, report_markdown, benchmark_summary, proofread_csv):
        mcp.tool()(tool)
    mcp.run()


if __name__ == "__main__":
    main()
