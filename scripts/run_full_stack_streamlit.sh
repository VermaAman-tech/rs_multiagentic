#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv in $ROOT_DIR"
  echo "Create it first, then rerun this script."
  exit 1
fi

source .venv/bin/activate
mkdir -p logs

STREAMLIT_HOST="${STREAMLIT_HOST:-127.0.0.1}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
TOOL_HOST="${TOOL_HOST:-127.0.0.1}"
TOOL_PORT="${TOOL_PORT:-9000}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
MODEL_PORT="${MODEL_PORT:-8002}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
SMOKE_CHAT_CHECK="${SMOKE_CHAT_CHECK:-1}"

TOOL_HEALTH_URL="http://${TOOL_HOST}:${TOOL_PORT}/health"
MODEL_MODELS_URL="http://${MODEL_HOST}:${MODEL_PORT}/v1/models"

wait_http() {
  local name="$1"
  local url="$2"
  local retries="${3:-90}"
  local delay="${4:-2}"

  for _ in $(seq 1 "$retries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[ok] ${name} is ready at ${url}"
      return 0
    fi
    sleep "$delay"
  done

  echo "[error] ${name} did not become ready: ${url}"
  return 1
}

start_tool_server() {
  if curl -fsS "$TOOL_HEALTH_URL" >/dev/null 2>&1; then
    echo "[skip] Tool server already running at ${TOOL_HEALTH_URL}"
  else
    echo "[start] Starting tool server on port ${TOOL_PORT}"
    nohup uvicorn tools.server:app --host 0.0.0.0 --port "$TOOL_PORT" \
      > logs/tool_server.out 2>&1 &
    echo $! > logs/tool_server.pid
  fi

  wait_http "tool server" "$TOOL_HEALTH_URL"
}

start_vllm() {
  if curl -fsS "$MODEL_MODELS_URL" >/dev/null 2>&1; then
    echo "[skip] vLLM already running at ${MODEL_MODELS_URL}"
  else
    echo "[start] Starting vLLM servers"
    bash scripts/start_all_vllm.sh
  fi

  wait_http "vLLM ORC/GA/PA" "$MODEL_MODELS_URL" 120 3

  # Optional: ensure chat completions are responsive before launching Streamlit.
  if [[ "$SMOKE_CHAT_CHECK" == "1" ]]; then
    echo "[check] Running vLLM chat completion smoke test"
    export MODEL_ID MODEL_HOST MODEL_PORT
    .venv/bin/python - <<'PY'
import os
import sys
import httpx

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
model_host = os.environ.get("MODEL_HOST", "127.0.0.1")
model_port = os.environ.get("MODEL_PORT", "8002")
url = f"http://{model_host}:{model_port}/v1/chat/completions"
payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": "Reply exactly with: ok"}],
    "temperature": 0.0,
    "max_tokens": 8,
}

try:
    r = httpx.post(url, json=payload, timeout=120.0)
    r.raise_for_status()
    print("[ok] vLLM chat completion smoke test passed")
except Exception as exc:
    print(f"[error] vLLM chat completion smoke test failed: {exc}")
    sys.exit(2)
PY
  fi
}

launch_streamlit() {
  echo "[start] Launching Streamlit Agent Studio"
  echo "        URL: http://${STREAMLIT_HOST}:${STREAMLIT_PORT}"
  export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
  exec streamlit run scripts/streamlit_agent_studio.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address "$STREAMLIT_HOST"
}

start_tool_server
start_vllm
launch_streamlit
