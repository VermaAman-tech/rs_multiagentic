#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

MODE="${1:-all}"
export MODE

python - << 'PY'
from pathlib import Path
import os

from huggingface_hub import snapshot_download

mode = os.environ.get("MODE", "all")
root = Path("models/weights")
root.mkdir(parents=True, exist_ok=True)

jobs = {
    "vra": ("Qwen/Qwen3-VL-4B-Instruct", root / "qwen3-vl-4b"),
    "shared": ("Qwen/Qwen3-4B-Instruct-2507", root / "qwen3-4b-2507"),
}

prithvi_candidates = [
    "ibm-nasa-geospatial/Prithvi-300M",
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
]

selected = []
if mode == "all":
    selected = ["vra", "shared", "prithvi"]
elif mode in jobs:
    selected = [mode]
elif mode == "prithvi":
    selected = [mode]
else:
    raise SystemExit(f"Unknown mode '{mode}'. Use: all|vra|shared|prithvi")

for key in selected:
    if key == "prithvi":
        out_dir = root / "prithvi-300m"
        success = False
        for repo_id in prithvi_candidates:
            print(f"\n>>> Downloading {repo_id} -> {out_dir}")
            try:
                snapshot_download(repo_id=repo_id, local_dir=str(out_dir), local_dir_use_symlinks=False)
                success = True
                break
            except Exception as exc:
                print(f"Failed {repo_id}: {exc}")
        if not success:
            raise SystemExit("Unable to download any Prithvi model candidate.")
    else:
        repo_id, out_dir = jobs[key]
        print(f"\n>>> Downloading {repo_id} -> {out_dir}")
        snapshot_download(repo_id=repo_id, local_dir=str(out_dir), local_dir_use_symlinks=False)

print("\nModel download step complete.")
PY
