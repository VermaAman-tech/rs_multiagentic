#!/bin/bash
#SBATCH --job-name=magrf-infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
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
export VLLM_USE_V1=0

MODEL="Qwen/Qwen3-4B-Instruct-2507"

echo "================================================================"
echo "MAGRF Verbose Inference Evaluation — ${MODEL}"
echo "Job: ${SLURM_JOB_ID:-local}  |  $(date)"
echo "================================================================"

# ── OEA Eval — text-only, conversation-prefix, full verbose trace ─────────
echo ">>> [1/2] OEA Inference (3330 tool-call steps, text-only)..."
python evaluation/benchmarks/run_deepseek_inference.py \
    --model "${MODEL}" \
    --oea-data data/openearth_agent/test.json \
    --text-only \
    --output results/e2_oea_verbose.json

echo ""

# ── ThinkGeo Eval — text-only 14B (VL-3B crashed on multi-modal tokens) ──
echo ">>> [2/2] ThinkGeo Inference (436 tasks, text-only 14B)..."
python evaluation/benchmarks/run_thinkgeo_inference.py \
    --model "${MODEL}" \
    --tg-data data/thinkgeo/ThinkGeoBench.json \
    --output results/e5_thinkgeo_verbose.json

echo ""
echo "================================================================"
echo "ALL DONE — $(date)"
echo "  OEA verbose results:      results/e2_oea_verbose.json"
echo "  ThinkGeo verbose results: results/e5_thinkgeo_verbose.json"
echo "================================================================"
