#!/usr/bin/env bash
set -euo pipefail

cd /mnt/media1/maram/rs_multiagentic_project/rs_multiagentic
source .venv/bin/activate

streamlit run scripts/streamlit_agent_studio.py --server.port 8501 --server.address 127.0.0.1
