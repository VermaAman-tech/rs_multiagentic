#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

python - << 'PY'
import json
from pathlib import Path

from datasets import load_dataset

out = Path("data")
out.mkdir(parents=True, exist_ok=True)

jobs = [
    ("openearthagent_public", "boolq", "validation", 200),
    ("thinkgeo_public", "gsm8k", "main", "test", 200),
]

summary = {}

# OEA placeholder public eval set from boolq (text/tool-calling scaffold)
name, ds_name, split, n = jobs[0]
ds = load_dataset(ds_name, split=split)
rows = []
for ex in ds.select(range(min(n, len(ds)))):
    rows.append(
        {
            "id": ex.get("id", ""),
            "question": ex.get("question", ""),
            "answer": ex.get("answer", False),
            "source": "boolq",
        }
    )
path = out / "openearthagent_eval_public.jsonl"
with path.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
summary[name] = {"path": str(path), "n": len(rows)}

# ThinkGeo placeholder public eval set from gsm8k test split
name, ds_name, cfg, split, n = jobs[1]
ds = load_dataset(ds_name, cfg, split=split)
rows = []
for ex in ds.select(range(min(n, len(ds)))):
    rows.append(
        {
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "source": "gsm8k",
        }
    )
path = out / "thinkgeo_eval_public.jsonl"
with path.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
summary[name] = {"path": str(path), "n": len(rows)}

sp = out / "dataset_summary.json"
sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
