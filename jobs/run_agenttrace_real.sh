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
  kill "${VLLM_SHARED_PID:-}" "${VLLM_VRA_PID:-}" "${TOOL_PID:-}" >/dev/null 2>&1 || true
  wait "${VLLM_SHARED_PID:-}" >/dev/null 2>&1 || true
  wait "${VLLM_VRA_PID:-}" >/dev/null 2>&1 || true
  wait "${TOOL_PID:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

uvicorn tools.server:app --host 127.0.0.1 --port 9000 > logs/tool_server_agenttrace.log 2>&1 &
TOOL_PID=$!

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > logs/vllm_shared_agenttrace.log 2>&1 &
VLLM_SHARED_PID=$!

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > logs/vllm_vra_agenttrace.log 2>&1 &
VLLM_VRA_PID=$!

for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8002/v1/models >/dev/null && curl -sf http://127.0.0.1:8001/v1/models >/dev/null; then
    break
  fi
  sleep 5
done

python scripts/run_detailed_agent_trace.py \
  --query "Plan safest evacuation route to nearest shelter after flood with damaged roads and blocked segments" \
  --model-id "Qwen/Qwen3-4B-Instruct-2507" \
  --base-url "http://127.0.0.1:8002/v1" \
  --orc-model-id "Qwen/Qwen3-4B-Instruct-2507" \
  --ga-model-id "Qwen/Qwen3-4B-Instruct-2507" \
  --pa-model-id "Qwen/Qwen3-4B-Instruct-2507" \
  --vra-model-id "Qwen/Qwen3-VL-4B-Instruct" \
  --orc-base-url "http://127.0.0.1:8002/v1" \
  --ga-base-url "http://127.0.0.1:8002/v1" \
  --pa-base-url "http://127.0.0.1:8002/v1" \
  --vra-base-url "http://127.0.0.1:8001/v1" \
  --tool-server "http://127.0.0.1:9000" \
  --strict-no-mock-fallback

