#!/bin/bash
#SBATCH --job-name=agenttrace-real
#SBATCH --output=logs/agenttrace_real_%j.out
#SBATCH --error=logs/agenttrace_real_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
mkdir -p logs

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

cleanup() {
  kill "${VLLM_PID:-}" "${TOOL_PID:-}" >/dev/null 2>&1 || true
  wait "${VLLM_PID:-}" >/dev/null 2>&1 || true
  wait "${TOOL_PID:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

uvicorn tools.server:app --host 127.0.0.1 --port 9000 > logs/tool_server_agenttrace.log 2>&1 &
TOOL_PID=$!

vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > logs/vllm_agenttrace.log 2>&1 &
VLLM_PID=$!

for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 5
done

python scripts/run_detailed_agent_trace.py \
  --query "Plan safest evacuation route to nearest shelter after flood with damaged roads and blocked segments" \
  --model-id "Qwen/Qwen2.5-14B-Instruct" \
  --base-url "http://127.0.0.1:8000/v1" \
  --tool-server "http://127.0.0.1:9000" \
  --strict-no-mock-fallback

