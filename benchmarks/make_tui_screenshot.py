"""Render the terminal viewer on a stored report to docs/figures/proofread_tui.svg (headless)."""
import asyncio, sys
from proofread.core import Report
from proofread.tui import ProofreadViewer

path = sys.argv[1] if len(sys.argv) > 1 else 'results/proofread_demos/breast_injected.json'
out = sys.argv[2] if len(sys.argv) > 2 else 'docs/figures/proofread_tui.svg'
cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'worst radius', 'worst texture', 'worst area', 'diagnosis']


async def main():
    rep = Report.load(path)
    rep.data = rep.data[[c for c in cols if c in rep.data.columns]] if 'breast' in path else rep.data
    rep.issues = rep.issues[rep.issues.column.isin(rep.data.columns)].reset_index(drop=True)
    app = ProofreadViewer(rep, source='breast_dirty.csv', out_prefix='/tmp/shot')
    async with app.run_test(size=(150, 34)) as pilot:
        await pilot.pause(); await pilot.press('f'); await pilot.pause()
        app.save_screenshot(filename=out.split('/')[-1], path='/'.join(out.split('/')[:-1]))
    print('wrote', out)

asyncio.run(main())
