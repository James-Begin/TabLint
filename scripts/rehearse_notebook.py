"""Video-production rehearsal; user setup opens a normal JupyterLab workspace."""
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess
import sys


def prepare(root: Path) -> Path:
    source = root / "examples" / "tablint_demo.ipynb"
    if not source.is_file():
        raise FileNotFoundError("Demo assets are missing. Clone https://github.com/James-Begin/TabLint and run from that checkout.")
    target = root / "demo" / "notebook-workspace" / "TabLint-demo.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    return target


def launch(args):
    root = Path(__file__).resolve().parents[1]
    try:
        notebook = prepare(root)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    print(f"Notebook ready: {notebook}\nSaved TabPFN results; no model download, API key or GPU needed.", flush=True)
    if args.prepare_only:
        return
    if find_spec("jupyterlab") is None:
        sys.exit("Install the notebook extra: uv run --extra notebook python scripts/rehearse_notebook.py")
    command = [sys.executable, "-m", "jupyterlab",
               f"--LabApp.default_url=/lab/tree/{notebook.relative_to(root).as_posix()}",
               f"--ServerApp.root_dir={root}", "--ServerApp.ip=127.0.0.1", f"--ServerApp.port={args.port}"]
    if args.no_browser:
        command.append("--no-browser")
    try:
        sys.exit(subprocess.call(command, cwd=root))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Open the saved notebook rehearsal for video production")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8888)
    launch(parser.parse_args())
