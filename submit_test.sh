#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
job_id=$(sbatch jobs/test_pipeline.sh | awk '{print $4}')
echo "Submitted job: $job_id"
echo "Track: squeue -j $job_id"
echo "Logs:  tail -f logs/test_${job_id}.out"
