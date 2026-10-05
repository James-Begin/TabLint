"""Verify the notebook launch path and execute the saved demo without model weights."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from proofread import demo


def test_prepare_preserves_working_notebook(tmp_path):
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/tablint_demo.ipynb").write_text("original")
    target = demo.prepare(tmp_path)
    assert target.read_text() == "original"
    target.write_text("my saved review")
    assert demo.prepare(tmp_path).read_text() == "my saved review"


def test_launch_uses_current_python_and_local_auth_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "prepare", lambda root: root / "demo/notebook-workspace/TabLint-demo.ipynb")
    monkeypatch.setattr(demo, "find_spec", lambda name: object())
    monkeypatch.setattr(demo.subprocess, "call", lambda command, **kw: calls.append((command, kw)) or 0)
    with pytest.raises(SystemExit) as exc:
        demo.launch(SimpleNamespace(prepare_only=False, no_browser=True, port=8890))
    assert exc.value.code == 0
    command, kw = calls[0]
    assert command[:3] == [sys.executable, "-m", "jupyterlab"]
    assert "--LabApp.default_url=/lab/tree/demo/notebook-workspace/TabLint-demo.ipynb" in command
    assert "--ServerApp.ip=127.0.0.1" in command and "--ServerApp.port=8890" in command
    assert "--no-browser" in command
    assert not any("token" in arg or "password" in arg for arg in command)
    assert kw["cwd"] == Path(demo.__file__).resolve().parents[1]


def test_saved_notebook_executes_without_weights():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "examples/tablint_demo.ipynb", as_version=4)
    nbformat.validate(notebook)
    executed = nbclient.NotebookClient(notebook, timeout=60, kernel_name="python3",
                                      resources={"metadata": {"path": str(root / "examples")}}).execute()
    outputs = [out for cell in executed.cells if cell.cell_type == "code" for out in cell.outputs]
    assert not any(out.output_type == "error" for out in outputs)
    assert any("application/vnd.jupyter.widget-view+json" in out.get("data", {}) for out in outputs)
    assert any("1877" in str(out.get("data", {})) for out in outputs)
