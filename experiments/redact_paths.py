"""Redact machine-specific absolute paths from stored artifacts (cosmetic only).

Replaces exact path prefixes in string values; verifies every JSON artifact is
otherwise identical (all non-string leaves unchanged) before writing.
"""
import json
import sys
from pathlib import Path

PREFIXES = [('/home/coder/forensics/', ''), ('/home/coder/.cache/', '~/.cache/'),
            ('/home/coder/', '~/')]


def scrub(text):
    for a, b in PREFIXES:
        text = text.replace(a, b)
    return text


def leaves(o):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from leaves(o[k])
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    elif not isinstance(o, str):
        yield o


changed = 0
for p in Path(sys.argv[1] if len(sys.argv) > 1 else 'results').rglob('*'):
    if not p.is_file() or p.suffix not in ('.json', '.log', '.txt', '.jsonl'):
        continue
    raw = p.read_text(errors='replace')
    new = scrub(raw)
    if new == raw:
        continue
    if p.suffix == '.json':
        a, b = json.loads(raw), json.loads(new)
        assert list(leaves(a)) == list(leaves(b)), p
    p.write_text(new)
    changed += 1
print('redacted files:', changed)
