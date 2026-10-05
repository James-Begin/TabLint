# Showcase production assets

The data, stored predictions and rehearsal notebooks here support the product video and development checks. User installation and analysis instructions are in [Getting started](../docs/GETTING_STARTED.md).

- [Auto MPG data and saved reports](video/README.md) — video examples and attribution.
- [Approved product video](showcase/README.md) — 1:28 animation and provenance.
- [Animation source](../showcase/) — editable production project.

For video production, `uv run --extra notebook python scripts/rehearse_notebook.py` opens the saved notebook rehearsal. `npm run rehearse:vscode` opens the editor rehearsal. These commands are for preparing footage, rather than setting up the product for a user's own data.

The browser integration is part of the Python package: [proofread/browser.py](../proofread/browser.py). Run `uv run --extra demo tablint app` to check your own CSV.
