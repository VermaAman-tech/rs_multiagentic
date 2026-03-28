#!/bin/bash
set -ex
export HF_TOKEN="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"
source /home/vis-comp/22b3929/rs_multiagentic/.venv/bin/activate

echo ">>> Downloading FloodNet..."
python -c '
from huggingface_hub import snapshot_download
snapshot_download("torchgeo/floodnet", repo_type="dataset", local_dir="data/floodnet", token="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")
print("FloodNet done")
' || echo "FloodNet failed, trying alternate..."
python -c '
from huggingface_hub import snapshot_download
snapshot_download("takara-ai/FloodNet_2021-Track_2_Dataset_HF", repo_type="dataset", local_dir="data/floodnet", token="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")
print("FloodNet alternate done")
' || echo "FloodNet alternate also failed"

echo ">>> Downloading LEVIR-CD..."
python -c '
from huggingface_hub import snapshot_download
snapshot_download("sy2002123/levir-cd", repo_type="dataset", local_dir="data/levircd", token="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")
print("LEVIR-CD done")
' || echo "LEVIR-CD failed"

echo ">>> Downloading Sen1Floods11 (Prithvi version)..."
python -c '
from huggingface_hub import snapshot_download
snapshot_download("ibm-nasa-geospatial/Prithvi-100M-sen1floods11", repo_type="model", local_dir="data/sen1floods11_prithvi", token="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")
print("Sen1Floods11 Prithvi done")
' || echo "Sen1Floods11 failed"

echo ">>> All remaining dataset downloads attempted."
