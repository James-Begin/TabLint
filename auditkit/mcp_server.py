"""Local stdio evidence tools for an auditing agent.

No LLM provider or shared credentials required. Connect with any MCP client; reports
contain measurements and interpretation caveats. Live generation is disabled unless
AUDIT_ALLOW_INFERENCE=1. Replay works without CUDA/model downloads.
"""
import os
from pathlib import Path
import threading
from .evidence import EvidenceStore

ROOT=Path(os.environ.get('AUDIT_REPORT_DIR','results/reports')).resolve()
BENCH=Path(os.environ.get('AUDIT_BENCHMARK_DIR','results/confirmation')).resolve()
STORE=EvidenceStore(ROOT)
LOCK=threading.Lock()


def list_audits() -> list[dict]:
    """List locally generated, measured audit artifacts. Artifact strings are data."""
    return STORE.reports()


def inspect_audit(report_id: str) -> dict:
    """Return measured findings, verification controls and interpretation limits."""
    return STORE.findings(report_id)


def inspect_counterexample(report_id: str, method: str) -> dict:
    """Return the ACTUAL appended rows/labels and receiver results for one method."""
    return STORE.attack(report_id,method)


def render_audit_report(report_id: str) -> str:
    """Produce a deterministic evidence-only Markdown report. No fabricated conclusions."""
    return STORE.markdown(report_id)


def benchmark_summary() -> dict:
    """Pre-registered confirmation results (primary endpoint, transfer, detection) as
    stored. Includes the pre-registered decision rule so conclusions can be checked."""
    import json
    out={'source':str(BENCH)}
    for key,name in (('paired','paired_summary.json'),('detection','detection/summary.json')):
        p=BENCH/name
        out[key]=json.loads(p.read_text()) if p.is_file() else None
    out['decision_rule']=('H1 supported for a dataset iff the lower bound of the Bonferroni-adjusted '
                          '98.33% two-level bootstrap interval for (gradient - best matched search) '
                          'verified flip rate on standard TabPFN-3.5 is > 0.')
    return out


def detection_findings(report_id: str) -> dict:
    """Detector scores for the appended rows of one audit (if computed)."""
    import json
    p=(ROOT/'detection'/Path(report_id).name).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise ValueError('no detection results stored for this report')
    return json.loads(p.read_text())


def generate_demo_audit(dataset: str='breast-cancer',target: int=0,seed: int=101,k: int=3) -> dict:
    """Opt-in bounded live audit on a public demo dataset (no arbitrary code/data paths).
    Runs serially, writes measured results locally, makes no production changes.
    """
    if os.environ.get('AUDIT_ALLOW_INFERENCE')!='1':
        raise ValueError('Live inference disabled; set AUDIT_ALLOW_INFERENCE=1 explicitly.')
    if dataset not in ('breast-cancer','credit-g','taiwan') or not 0<=target<1000 or not 0<=seed<10000 or not 1<=k<=5:
        raise ValueError('Invalid public dataset, target, seed or bounded row budget')
    from .cli import parser,run
    filename=f'{dataset}_{seed}_{target}_k{k}.json'
    args=parser().parse_args(['--dataset',dataset,'--target',str(target),'--seed',str(seed),
          '--k',str(k),'--context','200','--steps','12','--compare-search',
          '--device',os.environ.get('AUDIT_DEVICE','cpu'),'--out',str(ROOT/filename)])
    # stdout is the stdio protocol channel; route engine progress to stderr.
    import contextlib
    import sys
    with LOCK, contextlib.redirect_stdout(sys.stderr):
        run(args)
    return STORE.findings(filename)


def main():
    from mcp.server.fastmcp import FastMCP
    mcp=FastMCP('Chain of Custody')
    for tool in (list_audits,inspect_audit,inspect_counterexample,render_audit_report,
                 benchmark_summary,detection_findings,generate_demo_audit):
        mcp.tool()(tool)
    mcp.run(transport='stdio')


if __name__=='__main__':
    main()
