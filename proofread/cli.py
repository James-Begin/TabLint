"""`proofread` command line.

  proofread check data.csv --label outcome --out reports/data      # GPU if available, else CPU
  proofread view data.csv  # terminal spreadsheet with issues highlighted (accept / dismiss / save)
  proofread app            # browser viewer
  proofread mcp            # tools for an LLM agent (stdio)
"""
import argparse
import subprocess
import sys
from pathlib import Path


def _check(a):
    import pandas as pd
    from . import Proofreader
    df = pd.read_csv(a.csv)
    if a.label and a.label not in df.columns:
        sys.exit(f"--label {a.label!r} is not a column")
    print(f"proofreading {len(df)} rows × {df.shape[1]} columns with TabPFN-3.5 ...", flush=True)
    rep = Proofreader(device=a.device, folds=a.folds, fast=a.fast).check(df, label=a.label, threshold=a.threshold, max_issues=a.max_issues,
                                                                         categorical=not a.no_categorical)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(a.report) if a.report else out.with_suffix(".json")
    rep.meta["source_file"] = Path(a.csv).name
    rep.save(report_path)
    out.with_suffix(".md").write_text(rep.to_markdown(a.show))
    rep.issues.to_csv(out.with_suffix(".issues.csv"), index=False)
    try:
        out.with_suffix(".html").write_text(rep.highlight(a.show).to_html())
    except Exception as e:  # jinja2 missing etc.
        print("html export skipped:", e)
    print(rep.to_markdown(a.show))
    print(f"\nwrote {report_path}, {out.with_suffix('.md')}, {out.with_suffix('.issues.csv')}, {out.with_suffix('.html')}  ({rep.meta['seconds']:.0f}s)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tablint" if Path(sys.argv[0]).name == "tablint" else "proofread", description="Spellcheck for tables with TabPFN-3.5")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="flag suspicious cells and labels in a CSV")
    c.add_argument("csv"); c.add_argument("--label"); c.add_argument("--device", default="auto")
    c.add_argument("--fast", action="store_true", help="use TabPFN-3.5-Fast (speed/precision trade-off in docs/VERSION_COMPARISON.md)")
    c.add_argument("--no-categorical", action="store_true", help="skip categorical-cell checks (numeric cells and labels only)")
    c.add_argument("--folds", type=int, default=5); c.add_argument("--threshold", type=float, default=2.0, help="2: ~87%% precision / 57%% recall; 3: ~94%% / 20%% (retained benchmark)")
    c.add_argument("--max-issues", type=int, default=50); c.add_argument("--show", type=int, default=15)
    c.add_argument("--out", default="reports/proofread")
    c.add_argument("--report", help="write the JSON report to this exact path (used by the VS Code extension)")
    c.set_defaults(fn=_check)
    v = sub.add_parser("view", help="terminal spreadsheet with TabLint on top (CSV runs the check; .json opens a saved report)")
    v.add_argument("path"); v.add_argument("--label"); v.add_argument("--fast", action="store_true")
    v.add_argument("--threshold", type=float, default=2.0); v.add_argument("--max-issues", type=int, default=200)
    v.add_argument("--device", default="auto")
    v.set_defaults(fn=lambda a: __import__("proofread.tui", fromlist=["run"]).run(
        a.path, label=a.label, fast=a.fast, threshold=a.threshold, max_issues=a.max_issues, device=a.device))
    s = sub.add_parser("app", help="open the browser viewer (Streamlit)")
    s.set_defaults(fn=lambda a: sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run",
                   str(Path(__file__).with_name("browser.py"))])))
    m = sub.add_parser("mcp", help="serve Proofread tools to an LLM agent (stdio)")
    m.set_defaults(fn=lambda a: __import__("proofread.mcp_server", fromlist=["main"]).main())
    a = p.parse_args(argv); a.fn(a)


if __name__ == "__main__":
    main()
