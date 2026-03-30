#!/bin/bash
#SBATCH --job-name=magrf-full
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results models/weights/prithvi models/adapters

echo "================================================================"
echo "MAGRF Full Pipeline Phase 3 — Job ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local} | Start: $(date)"
echo "================================================================"

echo ""
echo ">>> Step 0: Setting up environment..."
source .venv/bin/activate || echo "No .venv, assuming global python"
export PYTHONPATH=.

echo ""
echo ">>> Step 1: Running pytest..."
pytest tests/ -v --tb=short 2>&1 || true

echo ""
echo ">>> Step 2: Running smoke test..."
bash scripts/smoke_test.sh

echo ""
echo ">>> Step 3: Training Stage 1 Damage MLP..."
python training/stage1_damage_mlp.py

echo ""
echo ">>> Step 4: Training Stage 1 Change Head..."
python training/stage1_change_head.py

echo ""
echo ">>> Step 4.5: Stage 2 LoRA Fine-Tuning..."
python training/stage2_lora.py --agent vra --epochs 3
python training/stage2_lora.py --agent ga --epochs 3
python training/stage2_lora.py --agent pa --epochs 3

echo ""
echo ">>> Step 5: E2 OpenEarthAgent Benchmark..."
export HF_HUB_OFFLINE=1
python evaluation/benchmarks/run_deepseek_inference.py --model Qwen/Qwen3-4B-Instruct-2507 --oea-data data/openearth_agent/test.json --output results/oea_full_inference.json

echo ""
echo ">>> Step 6.1: E3 TDRD Benchmark..."
python evaluation/benchmarks/tdrd_eval.py

echo ""
echo ">>> Step 6.2: E4 Conflict Benchmark..."
python evaluation/benchmarks/conflict_eval.py

echo ""
echo ">>> Step 6.3: E5 ThinkGeo Benchmark..."
export HF_HUB_OFFLINE=1
python evaluation/benchmarks/run_thinkgeo_inference.py --model Qwen/Qwen3-4B-Instruct-2507 --tg-data data/thinkgeo/ThinkGeoBench.json --output results/thinkgeo_full_inference.json

echo ""
echo ">>> Step 6.4: Running Ablations (A1-A8)..."
python evaluation/ablations/run_all_ablations.py --benchmark thinkgeo --ablation A8 --output results/ablation_a8.json || true
echo "Ablations generation finished."

echo ""
echo ">>> Step 6.5: DPO Data Extraction..."
python data_generation/generate_dpo_pairs.py

echo ""
echo "================================================================"
echo "MAGRF Full Pipeline — COMPLETE"
echo "End: $(date)"
echo "================================================================"

echo ""
echo "--- Results ---"
cat results/stage1_damage_mlp.json || true
echo ""
cat results/stage1_change_head.json || true
echo ""
cat results/oea_full_inference.json || true
echo ""
cat results/e3_tdrd.json || true
echo ""
cat results/e4_conflict.json || true
echo ""
cat results/thinkgeo_full_inference.json || true
echo ""
cat results/ablation_a8.json || true
echo "================================================================"
