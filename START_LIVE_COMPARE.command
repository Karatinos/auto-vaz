#!/usr/bin/env bash
#
# START_LIVE_COMPARE.command
#
# Double-click this in Terminal.app to run the LIVE, PAID comparison from a
# machine that can reach OpenRouter and your benchmark endpoint. This cannot
# run from the Claude cloud sandbox (egress is allowlisted to pip/npm/GitHub).
#
# Reads all secrets from .env (never committed). Writes metrics_report.json.

set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill it in." >&2
  exit 1
fi

# Set up the venv on first run.
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

# Confirm before spending money.
echo "This will make LIVE, PAID model calls and submit to EVAL_TARGET_URL."
read -r -p "Type 'run' to continue: " CONFIRM
[[ "${CONFIRM}" == "run" ]] || { echo "Aborted."; exit 0; }

LABEL="compare-$(date +%Y%m%d-%H%M%S)"
echo "Running live comparison (label: ${LABEL})..."
.venv/bin/python main.py --source live --provider deepseek --label "${LABEL}" --out metrics_report.json

echo
echo "Done. See metrics_report.json"
read -r -p "Press Enter to close."
