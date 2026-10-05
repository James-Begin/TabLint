"""Runtime smoke test of the Proofread viewer on stored reports (each example, issue selection, tabs)."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / 'demo' / 'proofread_app.py')).run(timeout=60)
assert not app.exception, [str(e) for e in app.exception]
print('viewer passed; examples:', len(app.radio[0].options) if app.radio else 0, 'dataframes:', len(app.dataframe))
for opt in list(app.radio[0].options):
    a = AppTest.from_file(str(ROOT / 'demo' / 'proofread_app.py')).run(timeout=60)
    a.radio[0].set_value(opt).run(timeout=60)
    assert not a.exception, (opt, [str(e) for e in a.exception])
    print('example passed:', Path(str(opt)).name, '| success boxes:', len(a.success), '| info boxes:', len(a.info))
