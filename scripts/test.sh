#!/usr/bin/env bash
# CI-style: run all tests and write dated evidence. Usage: bash scripts/test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/record_evidence.py
