#!/usr/bin/env bash
# End-to-end pipeline: 01 -> 07. Each stage caches to data/ and skips if done.
# Usage: ./run_all.sh [--force] [--limit N]   (flags passed through where relevant)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

$PY src/01_acquire.py "$@"
$PY src/02_clean.py "$@"
$PY src/03_index.py "$@"
$PY src/04_arms.py "$@"
$PY src/05_run.py "$@"
$PY src/06_analyze.py
$PY src/07_power.py
echo "Pipeline complete. See outputs/RESULTS_SUMMARY.md"
