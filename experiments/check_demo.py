"""Runtime smoke: viewer and benchmark page render measured artifacts without exceptions."""
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
os.environ['DECISION_STRESS_REPORTS_DIR']=str(ROOT/'results'/'reports')
app=AppTest.from_file(str(ROOT/'demo'/'app.py')).run(timeout=30)
assert not app.exception,[str(e) for e in app.exception]
print('viewer runtime passed; selectboxes:',len(app.selectbox),'dataframes:',len(app.dataframe))
for s in app.selectbox:
    if 'gradient' in str(s.options):
        s.select(next(o for o in s.options if 'gradient' in str(o))).run(timeout=30)
        assert not app.exception,[str(e) for e in app.exception]
        print('gradient selection passed')
        break
if app.radio:
    for option in list(app.radio[0].options):
        fresh=AppTest.from_file(str(ROOT/'demo'/'app.py')).run(timeout=30)
        fresh.radio[0].set_value(option).run(timeout=30)
        app=fresh
        assert not app.exception,(option,[str(e) for e in app.exception])
        print('artifact set passed:',option,'dataframes',len(app.dataframe))
bench=AppTest.from_file(str(ROOT/'demo'/'pages'/'1_Benchmark.py')).run(timeout=30)
assert not bench.exception,[str(e) for e in bench.exception]
print('benchmark page passed; dataframes:',len(bench.dataframe))
