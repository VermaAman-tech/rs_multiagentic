#!/usr/bin/env bash
set -euo pipefail

cd /mnt/media1/maram/rs_multiagentic_project/rs_multiagentic
source .venv/bin/activate

python scripts/chat_oea_case_live.py \
  --case-id 2 \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8002/v1 \
  --orc-model-id Qwen/Qwen3-4B-Instruct-2507 \
  --vra-model-id Qwen/Qwen3-4B-Instruct-2507 \
  --ga-model-id Qwen/Qwen3-4B-Instruct-2507 \
  --pa-model-id Qwen/Qwen3-4B-Instruct-2507 \
  --orc-base-url http://127.0.0.1:8002/v1 \
  --vra-base-url http://127.0.0.1:8002/v1 \
  --ga-base-url http://127.0.0.1:8002/v1 \
  --pa-base-url http://127.0.0.1:8002/v1 \
  --tool-server http://127.0.0.1:9000 \
  --max-turns 4
