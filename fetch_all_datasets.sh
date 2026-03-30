#!/bin/bash
set -euo pipefail

# Canonical entrypoint for full dataset download, aligned with plan/MAGRF_MASTER_README.md
bash scripts/download_datasets.sh all
