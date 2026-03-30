#!/bin/bash
set -euo pipefail

# Download only train-related datasets (xBD/FloodNet/Sen1Floods11/LEVIR-CD).
bash scripts/download_datasets.sh train
