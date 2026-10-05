"""Auto MPG (UCI, CC BY 4.0, https://archive.ics.uci.edu/dataset/9/auto+mpg) loader for the demo video."""
import io, shlex, urllib.request, zipfile
from pathlib import Path
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip"
CACHE = Path(__file__).resolve().parents[1] / "data_cache"


def load_auto_mpg() -> pd.DataFrame:
    raw = CACHE / "auto-mpg.data"
    if not raw.exists():
        CACHE.mkdir(exist_ok=True)
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read())).extractall(CACHE)
    rows = []
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        nums, name = line.split("\t")
        v = nums.split()
        rows.append(dict(car=name.strip().strip('"'), mpg=float(v[0]), cylinders=int(v[1]), displacement=float(v[2]),
                         horsepower=float("nan") if v[3] == "?" else float(v[3]), weight=float(v[4]),
                         acceleration=float(v[5]), model_year=1900 + int(v[6]), origin={"1": "USA", "2": "Europe", "3": "Japan"}[v[7]]))
    return pd.DataFrame(rows)
