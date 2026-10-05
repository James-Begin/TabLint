"""`custody` command line: investigate | app | mcp."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _investigate(a):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from . import Case
    if a.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("Set CUDA_VISIBLE_DEVICES explicitly before CUDA inference.")
    if a.csv:
        import pandas as pd
        df = pd.read_csv(a.csv)
        if a.label not in df.columns:
            sys.exit(f"--label {a.label!r} not in CSV columns")
        y = df[a.label]
        if y.nunique() != 2:
            sys.exit("label must be binary")
        X = df.drop(columns=[a.label]).select_dtypes("number")
        X, names, y = X.to_numpy(np.float32), list(X.columns), (y == sorted(y.unique())[1]).astype(int).to_numpy()
        groups = ()
    else:
        from auditkit.cli import dataset
        X, y, meta = dataset(a.dataset)
        names, groups = meta["names"], meta["groups"]
    ids = np.arange(len(y))
    ic, it = train_test_split(ids, test_size=0.3, stratify=y, random_state=a.seed)
    ic = np.random.default_rng(a.seed).choice(ic, min(a.context, len(ic)), replace=False)
    row = int(it[a.target % len(it)])
    case = Case(X[ic], y[ic], names, device=a.device, groups=groups, seed=a.seed)
    print(f"[1/3] suspect: gradient search through TabPFN (k={a.k}, steps={a.steps}) ...", flush=True)
    finding = case.suspect(X[row:row + 1], k=a.k, steps=a.steps)
    print(f"      surrogate P {finding.p_before:.3f} -> {finding.p_after:.3f}  flipped={finding.surrogate_flipped}", flush=True)
    print("[2/3] prove: refit standard TabPFN-3.5 and baselines ...", flush=True)
    proof = case.prove(finding)
    print(f"      verified on deployed TabPFN-3.5: {proof.verified}", flush=True)
    suspects = None
    if not a.skip_catch:
        print("[3/3] catch: exact leave-one-out over all rows ...", flush=True)
        suspects = case.catch(finding)
        print(f"      appended rows in top-{suspects.k}: {suspects.planted_in_top_k}/{suspects.k}", flush=True)
    out = Path(a.out)
    case.save(out.with_suffix(".json"), finding, proof, suspects)
    out.with_suffix(".md").write_text(case.report(finding, proof, suspects))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


def _app(a):
    env = dict(os.environ, DECISION_STRESS_REPORTS_DIR=str(Path(a.reports).resolve()))
    app = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)], env=env))


def _mcp(a):
    os.environ.setdefault("AUDIT_REPORT_DIR", str(Path(a.reports).resolve()))
    os.environ.setdefault("AUDIT_BENCHMARK_DIR", str(Path(a.benchmark).resolve()))
    from auditkit import mcp_server
    mcp_server.main()


def main(argv=None):
    p = argparse.ArgumentParser(prog="custody", description="Chain of Custody: forensics for tabular AI (TabPFN-3.5)")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("investigate", help="suspect -> prove -> catch on one prediction")
    i.add_argument("--dataset", default="breast-cancer", choices=["breast-cancer", "credit-g", "taiwan"])
    i.add_argument("--csv"); i.add_argument("--label", help="binary label column (with --csv)")
    i.add_argument("--seed", type=int, default=101); i.add_argument("--context", type=int, default=200)
    i.add_argument("--target", type=int, default=0, help="index into held-out rows")
    i.add_argument("--k", type=int, default=3); i.add_argument("--steps", type=int, default=12)
    i.add_argument("--device", default="cpu"); i.add_argument("--skip-catch", action="store_true")
    i.add_argument("--out", default="results/custody/case")
    i.set_defaults(fn=_investigate)
    s = sub.add_parser("app", help="open the replay app"); s.add_argument("--reports", default="results/reports")
    s.set_defaults(fn=_app)
    m = sub.add_parser("mcp", help="serve evidence tools to an LLM agent (stdio)")
    m.add_argument("--reports", default="results/reports"); m.add_argument("--benchmark", default="results/confirmation")
    m.set_defaults(fn=_mcp)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
