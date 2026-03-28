#!/bin/bash
#SBATCH --job-name=magrf-infer-vlm
#SBATCH --output=logs/infer_vlm_%j.out
#SBATCH --error=logs/infer_vlm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0

echo "================================================================"
echo "MAGRF Native Inference Evaluation (Qwen2.5-VL-3B-Instruct) — Job ${SLURM_JOB_ID:-local}"
echo "================================================================"

echo ">>> Running OEA Inference (636 instances)..."
python evaluation/benchmarks/run_real_inference.py --model Qwen/Qwen2.5-VL-3B-Instruct

echo ">>> Running ThinkGeo Inference (436 instances)..."
python evaluation/benchmarks/run_thinkgeo_inference.py --model Qwen/Qwen2.5-VL-3B-Instruct

echo "All native VLM evaluations complete."
