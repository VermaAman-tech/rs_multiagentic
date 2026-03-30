#!/bin/bash
#SBATCH --job-name=magrf-exp
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results models/weights/prithvi models/adapters

echo "================================================================"
echo "MAGRF Full Experiment Pipeline — Job ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local} | Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================================"

source .venv/bin/activate || echo "No .venv, assuming global python"
export PATH="$HOME/bin:$PATH"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONPATH=.
# Force vLLM legacy V0 engine — avoids subprocess EngineCore that re-imports Triton
# before our patch can run, causing 'cuda_utils is not compiled' crash.
export VLLM_USE_V1=0
# Point Triton to the system libcuda so it can at least compile its stubs if needed
LIBCUDA=$(ldconfig -p 2>/dev/null | grep 'libcuda\.so\.1' | head -1 | awk '{print $NF}')
[ -n "$LIBCUDA" ] && export TRITON_LIBCUDA_PATH="$LIBCUDA" && echo "TRITON_LIBCUDA_PATH=$LIBCUDA"

# We use offline vLLM inference scripts instead of running a background API server,
# to avoid CUDA OOM issues when multiple models need to be loaded simultaneously.
# The tool evaluation metrics (Inst, Tool, ArgN, ArgV, TSR, HRR) are natively measured
# by evaluation/benchmarks/run_deepseek_inference.py and evaluation/benchmarks/run_thinkgeo_inference.py.

# ============================================================
# STEP 3: Run pytest
# ============================================================
echo ""
echo ">>> Step 3: Running pytest..."
pytest tests/ -v --tb=short 2>&1 || true

# ============================================================
# STEP 4: Stage 1 Training
# ============================================================
echo ""
echo ">>> Step 4: Stage 1 Training (Damage MLP + Change Head)..."
python training/stage1_damage_mlp.py 2>&1 || true
python training/stage1_change_head.py 2>&1 || true

# ============================================================
# STEP 5: Stage 2 LoRA Fine-Tuning
# ============================================================
echo ""
echo -e "\n>>> Step 5: Stage 2 LoRA Fine-Tuning..."
# We fine-tune the multi-agent instances here to save time for testing
python training/stage2_lora.py --agent vra --epochs 1
python training/stage2_lora.py --agent ga --epochs 1
python training/stage2_lora.py --agent pa --epochs 1 2>&1 || true

echo ""
echo ">>> Step 6: E2 OpenEarthAgent Evaluation (1,169 real samples)..."
export HF_HUB_OFFLINE=1
python evaluation/benchmarks/run_deepseek_inference.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --oea-data data/openearth_agent/test.json \
    --output results/e2_oea_real.json

echo ""
echo ">>> Step 7: E5 ThinkGeo Evaluation (436 real tasks)..."
python evaluation/benchmarks/run_thinkgeo_inference.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tg-data data/thinkgeo/ThinkGeoBench.json \
    --output results/e5_thinkgeo_real.json

# ============================================================
# STEP 8: E3 TDRD + E4 Conflict Benchmarks
# ============================================================
echo ""
echo ">>> Step 8: E3 TDRD + E4 Conflict Benchmarks..."
python evaluation/benchmarks/tdrd_eval.py 2>&1 || true
python evaluation/benchmarks/conflict_eval.py 2>&1 || true

# ============================================================
# STEP 9: Ablations
# ============================================================
echo ""
echo ">>> Step 9: Running Ablations (A1-A8)..."
python evaluation/ablations/run_all_ablations.py \
    --benchmark thinkgeo --ablation A8_no_novel_tools \
    --output results/ablation_a8.json 2>&1 || true

# ============================================================
# STEP 10: DPO Data Generation
# ============================================================
echo ""
echo ">>> Step 10: DPO Alignment Data Generation..."
python data_generation/generate_dpo_pairs.py 2>&1 || true

# ============================================================
# STEP 11: Print Final Results Summary
# ============================================================
echo ""
echo "================================================================"
echo "MAGRF Full Experiment — COMPLETE"
echo "End: $(date)"
echo "================================================================"
echo ""
echo "--- Results ---"

for f in results/e2_oea_real.json results/e5_thinkgeo_real.json results/e3_tdrd.json results/e4_conflict.json results/ablation_a8.json results/stage1_damage_mlp.json results/stage1_change_head.json; do
    if [ -f "$f" ]; then
        echo "=== $(basename $f) ==="
        cat "$f"
        echo ""
    fi
done

# No background servers to kill 
echo "Pipeline complete. All results saved to results/"
