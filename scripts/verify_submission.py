"""Check submission links, stored run inventories and distributable artifacts without inference."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode().split("\0")
    files = sorted({ROOT / name for name in names if name and (ROOT / name).is_file()})
    errors = []
    for path in files:
        try:
            if path.suffix in {".json", ".ipynb"}:
                json.loads(path.read_text())
            elif path.suffix == ".py":
                ast.parse(path.read_text())
            elif path.suffix == ".svg":
                ET.parse(path)
            elif path.suffix == ".md":
                # Check both Markdown links and the HTML images used by GitHub's README.
                text = path.read_text()
                targets = re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", text)
                targets += re.findall(r'(?:href|src)="([^"]+)"', text)
                for target in targets:
                    target = unquote(target.strip("<>"))
                    parsed = urlsplit(target)
                    if parsed.scheme or parsed.netloc or not parsed.path:
                        continue
                    resolved = (path.parent / parsed.path).resolve()
                    if not resolved.is_relative_to(ROOT) or not resolved.exists():
                        errors.append(f"{path.relative_to(ROOT)}: broken link {target}")
        except (ValueError, SyntaxError, ET.ParseError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")

    suites = {
        "proofread": (12, set(range(701, 706))),
        "proofread_addendum": (12, set(range(701, 706))),
        "version_compare": (12, set(range(801, 806))),
        "categorical": (10, set(range(901, 906))),
    }
    numerical_datasets = None
    for suite, (count, seeds) in suites.items():
        grouped = {}
        for path in (ROOT / "results" / suite).glob("*.json"):
            record = json.loads(path.read_text())
            if "dataset" in record and "seed" in record:
                grouped.setdefault(record["dataset"], []).append(record["seed"])
        if len(grouped) != count or any(len(v) != 5 or set(v) != seeds for v in grouped.values()):
            errors.append(f"{suite}: expected {count} datasets with exactly five frozen seeds each")
        if suite == "proofread":
            numerical_datasets = set(grouped)
        elif suite != "categorical" and set(grouped) != numerical_datasets:
            errors.append(f"{suite}: numerical dataset scope differs")

    sys.path.insert(0, str(ROOT))
    from benchmarks.summarize_high_cardinality import load_records, summarize
    try:
        records = load_records(ROOT / "results/high_cardinality", verify_hashes=True)
        if summarize(records) != json.loads((ROOT / "results/high_cardinality/summary.json").read_text()):
            errors.append("High-cardinality summary differs from its 15 original result files")
        protocol = ROOT / "benchmarks/high_cardinality_protocol"
        provenance = json.loads((protocol / "provenance.json").read_text())
        for filename, key in (("PROTOCOL.md", "protocol_sha256"), ("requirements.txt", "requirements_sha256")):
            if hashlib.sha256((protocol / filename).read_bytes()).hexdigest() != provenance[key]:
                errors.append(f"Original high-cardinality {filename} checksum differs")
    except (ValueError, KeyError) as exc:
        errors.append(f"High-cardinality evidence: {exc}")

    export = json.loads((ROOT / "demo/showcase/export.json").read_text())
    video = ROOT / "demo/showcase" / export["file"]
    if hashlib.sha256(video.read_bytes()).hexdigest() != export["sha256"]:
        errors.append("Approved MP4 checksum differs from its export metadata")

    source = json.loads((ROOT / "vscode-proofread/package.json").read_text())
    with zipfile.ZipFile(ROOT / "demo/showcase/tablint.vsix") as archive:
        if archive.testzip() is not None:
            errors.append("VSIX contains a corrupt member")
        bundled = json.loads(archive.read("extension/package.json"))
        if bundled != source:
            errors.append("Downloadable VSIX manifest differs from extension source; rebuild it")
        for name in archive.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                errors.append(f"Unsafe VSIX member: {name}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(files)} files, local documentation links, 230 original benchmark runs, "
          "15 high-cardinality tables (60 arms), MP4 checksum and VSIX manifest.")


if __name__ == "__main__":
    main()
