#!/bin/bash
#SBATCH --job-name=agent-obs
#SBATCH --output=logs/run_obs_%j.out
#SBATCH --error=logs/run_obs_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }
mkdir -p logs

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

cleanup() {
  echo "Cleaning up background processes..."
  kill "${VLLM_PID:-}" "${TOOL_PID:-}" >/dev/null 2>&1 || true
  wait "${VLLM_PID:-}" >/dev/null 2>&1 || true
  wait "${TOOL_PID:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting tool server..."
uvicorn tools.server:app --host 127.0.0.1 --port 9000 > logs/tool_server_obs.log 2>&1 &
TOOL_PID=$!

echo "Starting vLLM server..."
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > logs/vllm_obs.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
VLLM_READY=0
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    VLLM_READY=1
    break
  fi
  sleep 5
done

if [ "$VLLM_READY" -eq 0 ]; then
  echo "Error: vLLM failed to start within 10 minutes. Check logs/vllm_obs.log"
  exit 1
fi
echo "vLLM is ready. Starting full observability pipeline run..."

export PYTHONUNBUFFERED=1
python scripts/run_obs_pipeline.py \
  --oea-path data/openearth_agent/test.json \
  --thinkgeo-path data/thinkgeo/ThinkGeoBench.json \
  --limit 200 \
  --model-id Qwen/Qwen2.5-14B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --tool-server http://127.0.0.1:9000 \
  --out-json results/agent_benchmark_runs/full_obs_trace_${SLURM_JOB_ID:-local}.json

echo "Pipeline run completed successfully."
