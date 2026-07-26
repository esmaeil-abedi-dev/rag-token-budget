#!/usr/bin/env bash
# End-to-end pipeline. Each stage caches to data/ and skips if done (--force).
# API-spending stages (03 embeddings, 04 compliance, 04b, 04c, 05) project
# their cost and 05 requires explicit confirmation (--yes to skip the prompt).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

# PYTHONHASHSEED pinned as defense-in-depth for reproducibility (the code also
# sorts all order-sensitive set iterations)
export PYTHONHASHSEED=0

$PY src/01_acquire.py "$@"          # datasets + sampling (no API)
$PY src/02_clean.py "$@"            # cleaning + chunking (no API)
$PY src/03_index.py "$@"            # embeddings via OpenRouter + validation gate
$PY src/04_arms.py "$@"             # graph build + budget compliance (paid; gated)
$PY src/05_run.py "$@"              # THE SWEEPS (prints cost, asks confirmation)
$PY src/04b_baselines.py "$@"       # floor/ceiling/random/full-context (paid; gated)
$PY src/04c_position.py "$@"        # lost-in-the-middle ablation (paid; gated)
$PY src/06_analyze.py               # stats + figures (no API)
$PY src/07_power.py                 # observed-effect power refresh (no API)
$PY src/08_summary.py               # RESULTS_SUMMARY.md (no API)
echo "Pipeline complete. See outputs/RESULTS_SUMMARY.md"
