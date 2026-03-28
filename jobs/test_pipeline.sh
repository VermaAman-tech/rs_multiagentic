#!/bin/bash
#SBATCH --job-name=lcot-test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

export HF_HOME="$SLURM_SUBMIT_DIR/models/weights"
export TRANSFORMERS_OFFLINE=0

bash scripts/setup_env.sh
source .venv/bin/activate

echo "=== MAGRF GPU Pipeline Test ==="
echo "Job ID: ${SLURM_JOB_ID:-NA} | Node: ${SLURM_NODELIST:-NA} | Start: $(date)"
echo "Python: $(which python)"
python - << 'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

bash scripts/smoke_test.sh

echo "=== Test Complete | End: $(date) ==="
