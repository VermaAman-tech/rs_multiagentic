#!/bin/bash
set -ex

# Load environment and dependencies
export HF_TOKEN="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"
source .venv/bin/activate
pip install huggingface_hub awscli gdown datasets

mkdir -p data/

echo ">>> 2. Downloading FloodNet (3.6 GB)"
python -c 'from huggingface_hub import snapshot_download; snapshot_download("ker0sene/FloodNet-Dataset", repo_type="dataset", local_dir="data/floodnet")'

echo ">>> 3. Downloading Sen1Floods11 (13 GB)"
# Try HuggingFace mirror first as it's often faster and cleaner than S3 syncing 13GB of tiny files
python -c 'from huggingface_hub import snapshot_download; snapshot_download("isp-uv-es/Sen1Floods11", repo_type="dataset", local_dir="data/sen1floods11")' || aws s3 sync s3://sen1floods11 data/sen1floods11 --no-sign-request

echo ">>> 4. Downloading LEVIR-CD (1.5 GB)"
python -c 'from huggingface_hub import snapshot_download; snapshot_download("qingwangcs/LEVIR-CD", repo_type="dataset", local_dir="data/levircd")'

echo ">>> 5. Downloading OpenEarthAgent Eval Images (~6 GB)"
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="mbzuai-oryx/OpenEarthAgent", repo_type="dataset", local_dir="data/openearth_agent_raw", ignore_patterns=["train*"])'

echo ">>> 6. Downloading ThinkGeo / GeoBenchX (~500 MB)"
if [ ! -d "data/thinkgeo_repo" ]; then
    git clone https://github.com/MBZUAI-Oryx/GeoAgent.git data/thinkgeo_repo || true
fi
cp -r data/thinkgeo_repo/benchmark/* data/thinkgeo/ || true
python -c 'from huggingface_hub import snapshot_download; snapshot_download("MBZUAI-Oryx/GeoBenchX", repo_type="dataset", local_dir="data/geobenchx")'

echo ">>> 7/8. Downloading DIOR (DOTA Alternative) for E1 RescueADI Testing (~11 GB)"
# Pulling just the DIOR dataset from EarthNets compilation
python -c 'from huggingface_hub import snapshot_download; snapshot_download("EarthNets/Dataset4EO", repo_type="dataset", local_dir="data/dota_v2", allow_patterns=["*DIOR*"])'

echo ">>> Massive Dataset Transfers Complete."
