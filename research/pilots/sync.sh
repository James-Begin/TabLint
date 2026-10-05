#!/bin/bash
# Developer helper: push code to a remote GPU host. Set FORENSICS_REMOTE=user@host.
set -euo pipefail
cd "$(dirname "$0")"
: "${FORENSICS_REMOTE:?set FORENSICS_REMOTE to your GPU host}"
tar cf - fz auditkit experiments tests demo docs *.py pyproject.toml 2>/dev/null \
  | ssh -o BatchMode=yes "$FORENSICS_REMOTE" 'mkdir -p ~/forensics && cd ~/forensics && tar xf -'
