#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: Python 3.10+ is required."
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

# On clusters, compute nodes are often offline. If deps are already present,
# reuse the venv as-is and avoid hitting package indexes.
if python - << 'PY'
import importlib
mods = [
  "fastapi",
  "uvicorn",
  "pydantic",
  "httpx",
  "yaml",
  "torch",
  "numpy",
  "huggingface_hub",
  "datasets",
  "tqdm",
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  echo "Existing .venv has required packages; skipping pip installs."
else
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  echo "Installing Python dependencies into .venv..."
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  pip install -e .
fi

mkdir -p logs results data models/weights

echo "Environment is ready in $ROOT_DIR/.venv"
