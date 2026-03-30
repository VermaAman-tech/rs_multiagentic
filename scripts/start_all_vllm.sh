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
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
VRA_GPU_UTIL=${VRA_GPU_UTIL:-0.85}
SHARED_GPU_UTIL=${SHARED_GPU_UTIL:-0.85}
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -lt 2 ]]; then
  echo "Detected <2 GPUs; assigning both model servers to GPU 0."
  SHARED_GPU=0
  # When sharing one GPU, cap per-process utilization to avoid OOM contention.
  VRA_GPU_UTIL=${VRA_GPU_UTIL_SINGLE_GPU:-0.42}
  SHARED_GPU_UTIL=${SHARED_GPU_UTIL_SINGLE_GPU:-0.42}
fi

CUDA_VISIBLE_DEVICES="$VRA_GPU" nohup vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code \
  --gpu-memory-utilization "$VRA_GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  > logs/vllm_vra.out 2>&1 &

CUDA_VISIBLE_DEVICES="$SHARED_GPU" nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8002 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code \
  --gpu-memory-utilization "$SHARED_GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  > logs/vllm_orc_ga_pa.out 2>&1 &

echo "Started vLLM servers with max_model_len=$MAX_MODEL_LEN (VRA util=$VRA_GPU_UTIL, shared util=$SHARED_GPU_UTIL)."
echo "Check logs/vllm_*.out"
