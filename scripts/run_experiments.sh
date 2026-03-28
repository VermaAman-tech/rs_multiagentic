#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

python evaluation/benchmarks/openearth_eval.py
python evaluation/benchmarks/thinkgeo_eval.py

echo "Wrote results/e2_oea.json and results/e5_thinkgeo.json"
