"""Terminal-viewer screenshots for the demo video (headless; SVG) from the stored demo/video reports."""
import asyncio
from proofread.core import Report
from proofread.tui import ProofreadViewer

SHOTS = [("demo/video/auto_mpg.proofread.json", "auto_mpg.csv", 13, "weight", "docs/figures/video_tui_real_buick.svg"),
         ("demo/video/auto_mpg_dirty.proofread.json", "auto_mpg_dirty.csv", 181, "weight", "docs/figures/video_tui_civic.svg")]


async def shot(path, name, row, col, out):
    rep = Report.load(path)
    app = ProofreadViewer(rep, source=name, out_prefix="/tmp/shot")
    async with app.run_test(size=(160, 36)) as pilot:
        await pilot.pause(); await pilot.press("f"); await pilot.pause()
        k = next(k for k, r in enumerate(app.issues) if int(r["row"]) == row and r["column"] == col)
        app._goto(k); await pilot.pause()
        app.save_screenshot(filename=out.split("/")[-1], path="/".join(out.split("/")[:-1]))
    print("wrote", out)

for s in SHOTS:
    asyncio.run(shot(*s))
