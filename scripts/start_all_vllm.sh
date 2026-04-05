#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate
mkdir -p logs

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM CLI not found in .venv. Install with: pip install vllm"
  exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
VRA_GPU=0
SHARED_GPU=1
VRA_MODEL=${VRA_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
SHARED_MODEL=${SHARED_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
VRA_PORT=${VRA_PORT:-8001}
SHARED_PORT=${SHARED_PORT:-8002}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
VRA_GPU_UTIL=${VRA_GPU_UTIL:-0.85}
SHARED_GPU_UTIL=${SHARED_GPU_UTIL:-0.85}
START_VRA_SERVER=${START_VRA_SERVER:-1}
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -lt 2 ]]; then
  echo "Detected <2 GPUs; assigning both model servers to GPU 0."
  SHARED_GPU=0
  START_VRA_SERVER=${START_VRA_SERVER_SINGLE_GPU:-0}
  # When sharing one GPU, run a single server by default to avoid OOM and startup thrash.
  if [[ "$START_VRA_SERVER" == "1" ]]; then
    VRA_GPU_UTIL=${VRA_GPU_UTIL_SINGLE_GPU:-0.42}
    SHARED_GPU_UTIL=${SHARED_GPU_UTIL_SINGLE_GPU:-0.42}
  else
    SHARED_GPU_UTIL=${SHARED_GPU_UTIL_SINGLE_GPU:-0.85}
  fi
fi

if [[ "$START_VRA_SERVER" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$VRA_GPU" nohup vllm serve "$VRA_MODEL" \
    --host 0.0.0.0 --port "$VRA_PORT" \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --trust-remote-code \
    --gpu-memory-utilization "$VRA_GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    > logs/vllm_vra.out 2>&1 &
else
  echo "Skipping dedicated VRA server (single-server mode)."
fi

CUDA_VISIBLE_DEVICES="$SHARED_GPU" nohup vllm serve "$SHARED_MODEL" \
  --host 0.0.0.0 --port "$SHARED_PORT" \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code \
  --gpu-memory-utilization "$SHARED_GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  > logs/vllm_orc_ga_pa.out 2>&1 &

if [[ "$START_VRA_SERVER" == "1" ]]; then
  echo "Started vLLM servers with max_model_len=$MAX_MODEL_LEN (VRA util=$VRA_GPU_UTIL, shared util=$SHARED_GPU_UTIL)."
else
  echo "Started single vLLM server on port $SHARED_PORT with max_model_len=$MAX_MODEL_LEN (util=$SHARED_GPU_UTIL)."
fi
echo "Check logs/vllm_*.out"
