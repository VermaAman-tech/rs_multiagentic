#!/bin/bash
#SBATCH --job-name=magrf-infer
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "================================================================"
echo "MAGRF Real Inference Evaluation — Job ${SLURM_JOB_ID:-local}"
echo "================================================================"

python evaluation/benchmarks/run_real_inference.py --limit 636

echo "Evaluation Complete."
