#!/bin/bash
set -euo pipefail

# Download only public benchmark datasets via plan-aligned workflow.
bash scripts/download_datasets.sh public
