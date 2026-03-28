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

nohup vllm serve Qwen/Qwen3-235B-A22B --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 --enable-auto-tool-choice --tool-call-parser hermes > logs/vllm_orc.out 2>&1 &
nohup vllm serve Qwen/Qwen2.5-VL-72B-Instruct --host 0.0.0.0 --port 8001 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes > logs/vllm_vra.out 2>&1 &
nohup vllm serve Qwen/Qwen3-32B --host 0.0.0.0 --port 8002 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes > logs/vllm_ga.out 2>&1 &
nohup vllm serve Qwen/Qwen3-32B --host 0.0.0.0 --port 8003 --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser hermes > logs/vllm_pa.out 2>&1 &

echo "Started vLLM servers. Check logs/vllm_*.out"
