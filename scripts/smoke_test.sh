#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

python - << 'PY'
import os
import sys

sys.path.insert(0, ".")
from tools.novel.temporal_stack_loader import temporal_stack_loader
from tools.novel.road_damage_scorer import road_damage_scorer
from tools.novel.evacuation_route_planner import evacuation_route_planner
from tools.novel.prithvi_embed import prithvi_embed

print("TemporalStackLoader:", temporal_stack_loader([0, 0, 1, 1], ["2024-01-01", "2024-02-01"]))
print("RoadDamageScorer:", road_damage_scorer("roads.gpkg", "roads", "damage.tif"))
print("EvacuationRoutePlanner:", evacuation_route_planner("graph.gpkg", ["A"], ["B"]))
print("PrithviEmbed:", prithvi_embed("scene.tif"))
print("Smoke test passed")
PY
