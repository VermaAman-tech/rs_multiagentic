#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

mkdir -p logs
LOG_FILE="logs/smoke_full_pipeline_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== Full Pipeline Smoke Test ==="
  echo "Start: $(date -Iseconds)"
  echo "CWD: $PWD"

  echo "\n[1/3] Health checks"
  for u in \
    "http://127.0.0.1:8001/v1/models" \
    "http://127.0.0.1:8002/v1/models" \
    "http://127.0.0.1:9000/health"; do
    echo "Checking $u"
    curl -fsS --max-time 5 "$u" >/dev/null
  done
  echo "Health checks passed"

  echo "\n[2/3] Running one-by-one eval smoke (1 OEA + 1 ThinkGeo sample)"
  python evaluation/benchmarks/run_full_framework_eval.py \
    --oea-limit 1 \
    --thinkgeo-limit 1 \
    --max-turns 4

  echo "\n[3/3] Latest eval summary"
  latest_dir=$(ls -dt results/full_framework_eval/eval_* | head -n 1)
  echo "Latest eval dir: $latest_dir"
  cat "$latest_dir/summary.json"

  echo "\nSmoke test completed: $(date -Iseconds)"
} | tee "$LOG_FILE"

echo "Smoke log saved to $LOG_FILE"
