"""Write examples/widget_preview.html: the real notebook widget (proofread/static/widget.js + .css) on a stored report,
driven by a stand-in model (no Jupyter needed). Open it in a browser, or screenshot it headlessly:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --allow-file-access-from-files \
        --window-size=1220,560 --screenshot=docs/figures/proofread_notebook_widget.png "file://$PWD/examples/widget_preview.html"
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
o = json.loads((ROOT / 'results/proofread_demos/breast_injected.json').read_text())
cols, data = o['data']['columns'], o['data']['data']
keep = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'worst radius', 'worst texture', 'diagnosis']
ci = [cols.index(c) for c in keep]
iss = [i for i in o['issues'] if i['column'] in keep][:20]
rows = [{'i': r, 'cells': [data[r][j] for j in ci]} for r in sorted({i['row'] for i in iss})]
state = {'columns': keep, 'rows': rows, 'issues': iss, 'status': ['accepted'] + ['open'] * (len(iss) - 1), 'selected': 1,
         'only_flagged': True, 'meta': {'n': len(data), 'model': 'TabPFN-3.5', 'patterns': []}}
html = f"""<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="../proofread/static/widget.css">
<style>body{{margin:16px;background:#fff}} .cell{{border:1px solid #e1e4e8;border-radius:6px;padding:10px;max-width:1180px}}
.in{{font:12px Menlo,monospace;color:#57606a;margin-bottom:6px}}</style></head><body>
<div class="cell"><div class="in">[3]: w = report.widget(); w</div><div id="el"></div></div>
<script type="module">
import widget from "../proofread/static/widget.js";
const state = {json.dumps(state)};
const h = {{}};
const model = {{ get: k => state[k], set: (k, v) => {{ state[k] = v; (h['change:' + k] || []).forEach(f => f()); }},
  on: (e, f) => (h[e] ||= []).push(f), off: () => {{}}, save_changes: () => {{}} }};
widget.render({{ model, el: document.getElementById('el') }});
</script></body></html>"""
(ROOT / 'examples/widget_preview.html').write_text(html)
print('wrote examples/widget_preview.html')
