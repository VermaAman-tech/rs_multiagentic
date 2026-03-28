#!/bin/bash
set -ex

source .venv/bin/activate

echo "Installing Vision Tool dependencies via Login Node internet..."
pip install groundingdino-py easyocr
pip install git+https://github.com/facebookresearch/sam2.git

echo "Downloading HuggingFace checkpoints locally..."
python - << 'PYEOF'
from huggingface_hub import hf_hub_download
import os

os.makedirs("models/weights/groundingdino", exist_ok=True)
if not os.path.exists("models/weights/groundingdino/pytorch_model.bin"):
    print("Downloading GroundingDINO...")
    hf_hub_download("IDEA-Research/grounding-dino-tiny", filename="pytorch_model.bin", local_dir="models/weights/groundingdino")

os.makedirs("models/weights/sam2", exist_ok=True)
if not os.path.exists("models/weights/sam2/sam2.1_hiera_large.pt"):
    print("Downloading SAM2...")
    hf_hub_download("facebook/sam2.1-hiera-large", filename="sam2.1_hiera_large.pt", local_dir="models/weights/sam2")
    
# Download the main Qwen models if they were blocked earlier as well!
os.makedirs("models/weights/qwen", exist_ok=True)
print("Downloading Qwen models (this may take a while or fail if >100GB without streaming)")
# Note: typically Qwen downloads happen via vLLM upon server start. 
# We just enable vLLM to download them. We will run the vLLM download offline trick.
PYEOF

echo "Done fetching components."
