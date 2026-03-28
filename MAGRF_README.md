# Multi-Agent Geospatial Reasoning Framework (MAGRF)
## Complete Build Guide for Copilot — Framework-First (No Datasets Required)

> **Context for Copilot**: This is a research framework for multi-agent geospatial intelligence.
> Four AI agents coordinate to analyse satellite images, score road damage, and plan evacuation routes.
> This README describes everything that must be built, in order, without needing the two research
> datasets (TDRD and Toolchain). Once datasets arrive (~3 weeks), only Stage 2 LoRA fine-tuning
> and the TDRD-specific experiments are unlocked. Everything else is buildable right now.

---

## What you are building

A 4-agent system:
- **VRA** (Visual Reasoning Agent) — satellite image analysis using Qwen3-VL-72B + Prithvi-300M
- **GA** (Geospatial Agent) — GIS reasoning using Qwen3-32B
- **PA** (Planning Agent) — route planning using Qwen3-32B
- **ORC** (Orchestrator) — task decomposition, message routing, conflict resolution using Qwen3-235B-A22B

All agents use the **ReAct loop** (Reason → Act → Observe) with Hermes function calling via vLLM.
All inter-agent communication goes through a typed **MPC protocol**.
All intermediate data lives in a **versioned Episodic Memory** store.

---

## What is doable RIGHT NOW (no datasets needed) — ~80% of the full project

| Category | Status | Notes |
|---|---|---|
| Full tool server (all 28 tools) | BUILD NOW | No data needed for the server itself |
| N1 TemporalStackLoader | BUILD NOW | Calls Copernicus API live |
| N2 RoadDamageScorer | BUILD NOW | Uses OSMnx + rasterio deterministically |
| N3 EvacuationRoutePlanner | BUILD NOW | Uses NetworkX + OR-Tools |
| N4 PrithviEmbed | BUILD NOW | Download Prithvi weights from HuggingFace |
| All 4 agent implementations | BUILD NOW | ReAct loop + Hermes tool calling |
| MPC protocol + message broker | BUILD NOW | Pure Python |
| Episodic Memory + Trajectory Log | BUILD NOW | Pure Python |
| Deadlock detection (DFS) | BUILD NOW | Pure Python, O(V+E) per message |
| 2-tier conflict resolution | BUILD NOW | Confidence judge + ORC LLM judge |
| Safety-aware context compression | BUILD NOW | CCQ >= 0.98 guarantee |
| Model download scripts | BUILD NOW | HuggingFace + vLLM |
| vLLM serving configs | BUILD NOW | Once models downloaded |
| Stage 1: Prithvi damage MLP | BUILD + TRAIN NOW | xBD dataset is publicly downloadable |
| Stage 1: change detection head | BUILD + TRAIN NOW | Sen1Floods11 is public |
| All 19 metric implementations | BUILD NOW | Pure Python, no data |
| E2: OpenEarthAgent benchmark | RUN NOW | OEA eval split is public |
| E5: ThinkGeo / GeoBenchX | RUN NOW | ThinkGeo dataset is public |
| Unit + integration tests | BUILD NOW | Use mock/fixture data |
| Full smoke test on Harvey AOI | RUN NOW | Uses live Copernicus API |
| Stage 2 LoRA training scripts | SCRIPT NOW, RUN LATER | Needs Toolchain Dataset |
| Stage 3 DPO scripts | SCRIPT NOW, RUN LATER | Needs Alignment Dataset |
| E1: RescueADI | SCRIPT NOW, RUN LATER | Needs RescueADI dataset |
| E3/E4: TDRD benchmark | SCRIPT NOW, RUN LATER | Needs TDRD Dataset |

---

## Complete File Structure

```
magrf/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
│
├── configs/
│   ├── agents.yaml          # Agent configs: model, port, tools, LoRA path
│   ├── models.yaml          # Model IDs, HuggingFace paths, VRAM requirements
│   ├── tools.yaml           # Tool server host, port, timeout per tool
│   └── experiments.yaml     # Experiment hyperparameters and paths
│
├── tools/
│   ├── __init__.py
│   ├── server.py            # FastAPI tool server — all 28 tools as HTTP endpoints
│   ├── schemas.py           # Pydantic input/output schemas for every tool
│   ├── vision/
│   │   ├── object_detection.py      # ObjectDetection
│   │   ├── segmentation.py          # SegmentObjectPixels
│   │   ├── description.py           # ImageDescription
│   │   ├── text_to_bbox.py          # TextToBbox
│   │   ├── region_attribute.py      # RegionAttributeDescription
│   │   ├── counting.py              # CountGivenObject
│   │   ├── ocr.py                   # OCR
│   │   ├── change_detection.py      # ChangeDetection
│   │   ├── draw_box.py              # DrawBox
│   │   └── add_text.py              # AddText
│   ├── gis/
│   │   ├── area_boundary.py         # GetAreaBoundary
│   │   ├── pois_layer.py            # AddPoisLayer
│   │   ├── compute_distance.py      # ComputeDistance
│   │   ├── display_map.py           # DisplayOnMap
│   │   ├── bbox_from_geotiff.py     # GetBboxFromGeotiff
│   │   └── display_geotiff.py       # DisplayOnGeotiff
│   ├── spectral/
│   │   ├── add_index_layer.py       # AddIndexLayer (NDVI/NDWI/NBR/NDBI/dNBR)
│   │   ├── compute_index_change.py  # ComputeIndexChange
│   │   └── show_index_layer.py      # ShowIndexLayer
│   ├── math_tools/
│   │   ├── calculator.py            # Calculator
│   │   ├── solver.py                # Solver (sympy)
│   │   └── plot.py                  # Plot
│   ├── utility/
│   │   ├── google_search.py         # GoogleSearch
│   │   └── terminate.py             # Terminate
│   └── novel/
│       ├── temporal_stack_loader.py # N1: multi-epoch Sentinel stacks
│       ├── road_damage_scorer.py    # N2: per-segment road traversability
│       ├── evacuation_route_planner.py  # N3: A* + VRP constrained routing
│       └── prithvi_embed.py         # N4: Prithvi-300M spectral embeddings
│
├── agents/
│   ├── base_agent.py        # Abstract ReAct loop + Hermes tool calling
│   ├── vra.py               # Visual Reasoning Agent (Qwen3-VL-72B)
│   ├── ga.py                # Geospatial Agent (Qwen3-32B)
│   ├── pa.py                # Planning Agent (Qwen3-32B)
│   └── orc.py               # Orchestrator (Qwen3-235B-A22B)
│
├── framework/
│   ├── mpc/
│   │   ├── message.py        # Message dataclass: all 7 types, all fields
│   │   ├── broker.py         # Routes all messages through ORC
│   │   └── subscriptions.py  # Per-agent subscription map
│   ├── memory/
│   │   ├── episodic_memory.py   # Shared versioned KV store, CCQ tracking
│   │   ├── working_memory.py    # Per-agent 4K token scratchpad, auto-flush
│   │   └── trajectory_log.py   # Append-only, ORC-only write
│   ├── protocols/
│   │   ├── deadlock.py       # DFS wait-for graph + Break-and-Replay
│   │   ├── conflict.py       # Tier-1 confidence + Tier-2 LLM judge
│   │   ├── compression.py    # Safety-aware EM compression (CCQ >= 0.98)
│   │   └── safety.py         # Route safety check (RSS), forces re-route if < 1.0
│   └── episode_runner.py    # Top-level: orchestrates one full 4-agent episode
│
├── models/
│   ├── download.py          # Download all weights to models/weights/
│   ├── prithvi/
│   │   ├── encoder.py       # Load frozen Prithvi-300M
│   │   ├── damage_mlp.py    # 2-layer MLP: 768 -> 256 -> 4 (damage classes)
│   │   └── phase_lstm.py    # LSTM: change scores over epochs -> phase label
│   └── serving/
│       ├── vra_vllm.yaml    # Qwen3-VL-72B serving config
│       ├── ga_vllm.yaml     # Qwen3-32B (GA) serving config
│       ├── pa_vllm.yaml     # Qwen3-32B (PA) serving config
│       ├── orc_vllm.yaml    # Qwen3-235B-A22B serving config
│       └── start_servers.sh # Launch all 4 vLLM servers
│
├── training/
│   ├── stage1_damage_mlp.py     # Prithvi MLP on xBD — RUN NOW
│   ├── stage1_change_head.py    # SAM2-CD head on Sen1Floods11 — RUN NOW
│   ├── stage2_lora_vra.py       # LoRA VRA on Toolchain — NEEDS DATASET
│   ├── stage2_lora_ga.py        # LoRA GA — NEEDS DATASET
│   ├── stage2_lora_pa.py        # LoRA PA — NEEDS DATASET
│   ├── stage3_reward_model.py   # Bradley-Terry on Alignment — NEEDS DATASET
│   ├── stage3_dpo_orc.py        # DPO for ORC — NEEDS DATASET
│   └── utils/
│       ├── data_loaders.py      # DataLoaders for all training datasets
│       └── metrics_callback.py  # Log metrics during training
│
├── evaluation/
│   ├── metrics.py               # All 19 metrics with formulas (see below)
│   ├── benchmarks/
│   │   ├── openearth_eval.py    # E2: OEA — RUN NOW
│   │   ├── thinkgeo_eval.py     # E5: ThinkGeo — RUN NOW
│   │   ├── rescueadi_eval.py    # E1: RescueADI — NEEDS DATASET
│   │   ├── tdrd_eval.py         # E3: TDRD — NEEDS DATASET
│   │   └── conflict_eval.py     # E4: Conflict re-planning — NEEDS DATASET
│   └── ablations/
│       ├── ablation_configs.py  # Config overrides for A1-A8
│       └── run_all_ablations.py # Runs all 8 ablations sequentially
│
├── scripts/
│   ├── download_models.sh       # Download Prithvi + Qwen3 family
│   ├── download_datasets.sh     # Download OEA, ThinkGeo, xBD, Sen1Floods11
│   ├── setup_env.sh             # Full environment from scratch
│   ├── start_tool_server.sh     # Launch FastAPI tool server on port 9000
│   ├── start_all_vllm.sh        # Launch all 4 vLLM servers
│   ├── smoke_test.sh            # Full episode on Harvey AOI
│   └── run_experiments.sh       # Run E2 + E5 (available now)
│
└── tests/
    ├── conftest.py              # Pytest fixtures: mock tool server, fixture data
    ├── test_tools/
    │   ├── test_vision_tools.py
    │   ├── test_gis_tools.py
    │   ├── test_spectral_tools.py
    │   ├── test_novel_tools.py   # N1/N2/N3/N4
    │   └── test_math_tools.py
    ├── test_agents/
    │   ├── test_vra.py
    │   ├── test_ga.py
    │   ├── test_pa.py
    │   └── test_orc.py
    ├── test_framework/
    │   ├── test_mpc.py
    │   ├── test_memory.py
    │   ├── test_deadlock.py
    │   └── test_conflict.py
    └── test_integration/
        ├── test_single_epoch.py  # 2-agent, 1 image
        └── test_multi_epoch.py   # 4-agent, 3 epochs
```

---

## Build Priority Order

Build in this exact sequence. Each group is a testable milestone.

### GROUP 1 — Foundation [Days 1-2] — Goal: tool server starts, unit tests pass
1. `requirements.txt` and `setup.py`
2. `configs/` — all four YAML files
3. `tools/schemas.py` — Pydantic schemas for ALL 28 tools
4. `tools/server.py` — FastAPI server with health check + all 28 tool endpoints
5. All 24 existing tools (full implementation, not stubs — use GroundingDINO+SAM2 for vision tools)
6. `tests/conftest.py` — mock tool server fixture
7. `tests/test_tools/` — unit tests for all 28 tools
8. **Gate**: `pytest tests/test_tools/ -v` must pass 100%

### GROUP 2 — Novel Tools [Days 2-4] — Goal: N1/N2/N3/N4 fully working
9. `tools/novel/temporal_stack_loader.py` — N1 (full Copernicus download + ECC align)
10. `tools/novel/road_damage_scorer.py` — N2 (OSMnx + rasterio spatial intersection)
11. `tools/novel/evacuation_route_planner.py` — N3 (NetworkX A* + OR-Tools VRP)
12. `tools/novel/prithvi_embed.py` — N4 (Prithvi forward pass + spectral indices)
13. `tests/test_tools/test_novel_tools.py`
14. **Gate**: `pytest tests/test_tools/test_novel_tools.py -v`

### GROUP 3 — Memory and MPC [Days 3-5] — Goal: agents can share state
15. `framework/memory/episodic_memory.py` (versioning + CCQ tracking)
16. `framework/memory/working_memory.py` (4K limit + auto-flush)
17. `framework/memory/trajectory_log.py` (append-only)
18. `framework/mpc/message.py` (all 7 message types)
19. `framework/mpc/broker.py` (priority queue + TTL + ORC-only routing)
20. `framework/mpc/subscriptions.py`
21. `tests/test_framework/test_mpc.py` + `test_memory.py`
22. **Gate**: all MPC tests pass

### GROUP 4 — Protocols [Days 4-6] — Goal: deadlock, conflict, compression working
23. `framework/protocols/deadlock.py` (DFS wait-for graph + Break-and-Replay)
24. `framework/protocols/conflict.py` (Tier-1 confidence gap + Tier-2 LLM judge)
25. `framework/protocols/compression.py` (safety-aware eviction + CCQ >= 0.98 check)
26. `framework/protocols/safety.py` (RSS check, force re-route if < 1.0)
27. `tests/test_framework/test_deadlock.py` + `test_conflict.py`
28. **Gate**: inject artificial deadlock → break detected in < 1ms

### GROUP 5 — Agents [Days 5-8] — Goal: ReAct loops work with mock LLM
29. `agents/base_agent.py` (ReAct loop, Hermes parsing, tool dispatcher)
30. `agents/vra.py` (VRA system prompt, thinking mode selective)
31. `agents/ga.py` (GA system prompt, pure text reasoning)
32. `agents/pa.py` (PA system prompt, thinking mode always)
33. `agents/orc.py` (ORC decompose_task, assign_subtask, conflict resolution, Terminate)
34. `tests/test_agents/` — use mock vLLM responses (pre-scripted JSON tool calls)
35. **Gate**: VRA processes a single test image end-to-end with mock LLM

### GROUP 6 — Episode Runner + Integration [Days 7-9]
36. `framework/episode_runner.py` (parallel VRA+GA, SYNC→PA, ORC orchestration)
37. `tests/test_integration/test_single_epoch.py`
38. `tests/test_integration/test_multi_epoch.py`
39. **Gate**: full mock episode completes, all fields in episode dict populated

### GROUP 7 — Models + vLLM Serving [Days 8-10]
40. `models/download.py` + `scripts/download_models.sh`
41. `models/serving/` — all 4 vLLM YAML configs
42. `scripts/start_all_vllm.sh`
43. `scripts/setup_env.sh`
44. Replace mock LLM calls in agents with real vLLM OpenAI-compatible client
45. **Gate**: `curl http://localhost:8001/v1/models` returns Qwen3-VL-72B

### GROUP 8 — Prithvi Heads + Stage 1 Training [Days 9-12]
46. `models/prithvi/encoder.py` (load frozen Prithvi-300M from HuggingFace)
47. `models/prithvi/damage_mlp.py` (2-layer MLP: 768→256→4)
48. `models/prithvi/phase_lstm.py` (2-layer LSTM: T×1→4-class phase)
49. `training/stage1_damage_mlp.py` — **TRAIN THIS NOW on xBD**
50. `training/stage1_change_head.py` — **TRAIN THIS NOW on Sen1Floods11**
51. `scripts/download_datasets.sh` (OEA, ThinkGeo, xBD, Sen1Floods11)
52. **Gate**: damage MLP weighted F1 >= 0.71 on xBD test split

### GROUP 9 — Metrics + Benchmarks [Days 11-14]
53. `evaluation/metrics.py` — ALL 19 metrics (see list below)
54. `evaluation/benchmarks/openearth_eval.py` — **RUN E2 NOW**
55. `evaluation/benchmarks/thinkgeo_eval.py` — **RUN E5 NOW**
56. **Gate**: E2 runs to completion, results logged to `results/e2_oea.json`

### GROUP 10 — Ablations + Smoke Test [Days 13-16]
57. `evaluation/ablations/ablation_configs.py` (8 ablation config overrides)
58. `evaluation/ablations/run_all_ablations.py`
59. `scripts/smoke_test.sh`
60. **Gate**: smoke test on Harvey AOI — RSS >= 0.90, episode completes

### GROUP 11 — Stage 2+3 Scripts, ready to run when data arrives [Days 14-16]
61. `training/stage2_lora_vra.py`
62. `training/stage2_lora_ga.py`
63. `training/stage2_lora_pa.py`
64. `training/stage3_reward_model.py`
65. `training/stage3_dpo_orc.py`
66. `evaluation/benchmarks/rescueadi_eval.py`
67. `evaluation/benchmarks/tdrd_eval.py`
68. `evaluation/benchmarks/conflict_eval.py`

---

## All 19 Metrics to Implement in `evaluation/metrics.py`

### Group A — Inherited RS (comparison with prior work)
| Metric | Formula | Benchmark |
|---|---|---|
| `IoU` | TP / (TP+FP+FN) pixel-level per class | xBD, FloodNet |
| `mIoU` | Mean IoU over C classes | OEA, RescueADI |
| `weighted_f1` | Per-class F1 weighted by class frequency | xBD primary metric |
| `Inst` | Correct answers / total × 100 (±5% numeric, exact categorical) | OEA baseline 99.51 |
| `Tool` | Correct tool names / total calls × 100 | OEA baseline 97.18 |
| `ArgN` | Correct arg names / total arg slots × 100 | OEA baseline 96.08 |
| `ArgV` | Correct arg values / total arg slots × 100 | OEA baseline 62.10 — main target |
| `Summ` | ROUGE-L F1 vs GT summary or LLM-as-Judge 0-100 | OEA baseline 83.64 |
| `TSR` | Correct final answers / total episodes × 100 | RescueADI, ThinkGeo |
| `HRR` | Correctly rejected unsolvable / all unsolvable × 100 | ThinkGeo, GeoBenchX |

### Group B — Novel Coordination and Safety (new contributions)
| Metric | Formula | Notes |
|---|---|---|
| `MGAR` | Episodes where ALL L2 subtasks complete AND answer correct / total × 100 | Main TDRD metric; weight 0.30 |
| `RSS` | Route segments with score < 0.6 / total segments × 100 | Safety metric; weight 0.30; target >= 99.0 |
| `MTCS` | Epoch pairs with consistent phase / total pairs × 100 | Temporal reasoning; monotone progression |
| `TLS` | GT tool calls / actual tool calls (> 1.0 = more efficient) | Trajectory efficiency; weight 0.10 |
| `CRR` | ORC resolutions matching GT winner / total conflicts × 100 | Conflict resolution quality; weight 0.10 |
| `DDF1` | F1 of deadlock detection on injected deadlock test set | Target >= 0.90; weight 0.10 |
| `CCQ` | Safety facts retained after compression / total safety facts | Target >= 0.98; weight 0.10 |
| `CDF1` | F1 of route invalidation detection on conflict subset (468 ep) | TDRD conflict evaluation |
| `ReSR` | Re-plans with RSS=1.0 / conflicts detected × 100 | Re-plan success rate |

---

## `configs/models.yaml` — exact content

```yaml
vra:
  model_id: "Qwen/Qwen3-VL-72B-Instruct"
  vllm_port: 8001
  tensor_parallel: 1
  max_model_len: 32768
  thinking_mode: selective
  lora_path: null

ga:
  model_id: "Qwen/Qwen3-32B"
  vllm_port: 8002
  tensor_parallel: 1
  max_model_len: 32768
  thinking_mode: selective
  lora_path: null

pa:
  model_id: "Qwen/Qwen3-32B"
  vllm_port: 8003
  tensor_parallel: 1
  max_model_len: 32768
  thinking_mode: always
  lora_path: null

orc:
  model_id: "Qwen/Qwen3-235B-A22B"
  vllm_port: 8000
  tensor_parallel: 2
  max_model_len: 65536
  thinking_mode: always
  fp8_quantization: true
  lora_path: null

prithvi:
  model_id: "ibm-nasa-geospatial/Prithvi-300M"
  damage_mlp_path: null
  phase_lstm_path: null
```

---

## `configs/agents.yaml` — exact content

```yaml
vra:
  tool_server: "http://localhost:9000"
  max_react_turns: 12
  working_memory_tokens: 4096
  tools:
    - GetBboxFromGeotiff
    - ObjectDetection
    - SegmentObjectPixels
    - ImageDescription
    - TextToBbox
    - RegionAttributeDescription
    - CountGivenObject
    - OCR
    - ChangeDetection
    - DrawBox
    - AddText
    - AddIndexLayer
    - ComputeIndexChange
    - ShowIndexLayer
    - TemporalStackLoader
    - PrithviEmbed

ga:
  tool_server: "http://localhost:9000"
  max_react_turns: 8
  working_memory_tokens: 4096
  tools:
    - GetAreaBoundary
    - AddPoisLayer
    - GetBboxFromGeotiff
    - ComputeDistance
    - DisplayOnMap
    - DisplayOnGeotiff
    - RoadDamageScorer

pa:
  tool_server: "http://localhost:9000"
  max_react_turns: 10
  working_memory_tokens: 4096
  tools:
    - EvacuationRoutePlanner
    - ComputeDistance
    - Calculator
    - Solver
    - Plot

orc:
  tool_server: "http://localhost:9000"
  max_react_turns: 20
  episodic_memory_soft_limit_tokens: 32000
  episodic_memory_hard_limit_tokens: 64000
  compression_trigger: 0.80
  deadlock_timeout_seconds: 30
  conflict_confidence_gap: 0.20
  tools:
    - GoogleSearch
    - Calculator
    - Terminate
```

---

## `tools/server.py` — FastAPI tool server skeleton

```python
from fastapi import FastAPI, HTTPException
from tools.schemas import *
from tools.vision import object_detection, segmentation, description  # etc.
from tools.gis import area_boundary, pois_layer  # etc.
from tools.novel import temporal_stack_loader, road_damage_scorer, evacuation_route_planner, prithvi_embed

app = FastAPI(title="MAGRF Tool Server", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok", "n_tools": 28}

@app.post("/tools/ObjectDetection", response_model=ObjectDetectionOutput)
def run_object_detection(req: ObjectDetectionInput):
    return object_detection.run(req)

@app.post("/tools/SegmentObjectPixels", response_model=SegmentObjectPixelsOutput)
def run_segment(req: SegmentObjectPixelsInput):
    return segmentation.run(req)

# ... one endpoint per tool, all following the same pattern ...

@app.post("/tools/TemporalStackLoader", response_model=TemporalStackLoaderOutput)
def run_temporal_stack(req: TemporalStackLoaderInput):
    result = temporal_stack_loader.temporal_stack_loader(
        aoi_bbox=req.aoi_bbox,
        date_range=req.date_range,
        sensor=req.sensor,
        max_cloud_pct=req.max_cloud_pct,
    )
    return result

@app.post("/tools/RoadDamageScorer", response_model=RoadDamageScorerOutput)
def run_road_damage(req: RoadDamageScorerInput):
    return road_damage_scorer.road_damage_scorer(
        gpkg_path=req.gpkg_path,
        road_layer_name=req.road_layer_name,
        damage_raster_layer=req.damage_raster_layer,
        buffer_meters=req.buffer_meters,
    )

@app.post("/tools/EvacuationRoutePlanner", response_model=EvacuationRoutePlannerOutput)
def run_evacuation(req: EvacuationRoutePlannerInput):
    return evacuation_route_planner.evacuation_route_planner(**req.dict())

@app.post("/tools/PrithviEmbed", response_model=PrithviEmbedOutput)
def run_prithvi(req: PrithviEmbedInput):
    return prithvi_embed.prithvi_embed(**req.dict())
```

---

## `agents/base_agent.py` — ReAct loop (Copilot: implement the full class)

The base agent must:
1. Accept a task message and EM context dict
2. Build a system prompt + initial user message
3. Call vLLM OpenAI-compatible API with tool schemas
4. Parse tool calls from response (Hermes format)
5. POST each tool call to `http://localhost:9000/tools/{tool_name}`
6. Append result to messages and continue the loop
7. Stop when `finish_reason == "stop"` or `max_turns` reached
8. Log every tool call (tool_name, args, output, latency_ms, thinking_trace) for Trajectory Log
9. Return dict with: `{agent, output, tool_calls, confidence, n_turns}`

Key detail: Qwen3 returns `reasoning_content` field alongside `content` when thinking mode is on.
Extract this as `thinking_trace` and include it in every tool call log entry.

---

## `framework/episode_runner.py` — Orchestration sequence

The episode runner must execute this exact sequence:

```
1. ORC.decompose_task(query) → L2 plan (which agent handles which tools)
2. ORC sends TASK message to VRA (priority=1)
3. ORC sends TASK message to GA (priority=1) — PARALLEL with step 2
4. VRA.run() → executes full ReAct loop → writes to EM → sends RESULT
5. GA.run() → executes full ReAct loop → writes to EM → sends RESULT
6. ORC receives both RESULTs
7. ORC checks consistency (if VRA and GA outputs conflict → 2-tier resolution)
8. ORC checks EM budget → if > 80%, run compress_context()
9. ORC sends SYNC + TASK to PA
10. PA.run() → executes full ReAct loop → writes routes to EM → sends RESULT
11. ORC verifies RSS >= 1.0 on all routes (if not → PA.re_route())
12. ORC.log_trajectory() → writes full episode to Trajectory Log
13. ORC.Terminate(final_answer)
14. Return episode dict with all fields
```

After every RESULT message: run deadlock detector. If cycle found → break_cycle().
After every EM write: check EM token count. If > 80% soft limit → compress_context().

---

## `framework/protocols/compression.py` — Safety-aware compression

Rules that MUST be hardcoded:
- Keys containing "damage_polygons", "flood_extent", "impassable_roads", "conflict_events": NEVER compress
- Keys containing "prithvi_embed_": ALWAYS evict first (largest, least important)
- Keys containing "spectral_": replace with per-epoch scalar statistics (mean, std, max of index values)
- Keys containing "change_map_": replace with top-5 anomaly centroids only
- After any compression: compute CCQ. If CCQ < 0.98 → abort and log error. Never proceed with CCQ < 0.98.

---

## `training/stage2_lora_vra.py` — Script now, run when Toolchain arrives

```python
# Script is complete. Just needs --data_path pointing at toolchain_train.jsonl
# The DataLoader filters for episodes where 'VRA' appears in tool_calls
# Fine-tune Qwen3-VL-72B with LoRA r=16, alpha=32 on VRA subsequences
# Expected: ArgV 59% → 67%+, tool accuracy 94% → 97%+
# Config: 3 epochs, AdamW lr=2e-4, 1x A100 80GB, ~12 hours
```

---

## Key implementation references

| Component | Reference |
|---|---|
| Qwen3 function calling | qwen.readthedocs.io/en/latest/framework/function_call.html |
| vLLM serving Qwen3 | `vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes` |
| Prithvi-300M weights | huggingface.co/ibm-nasa-geospatial/Prithvi-300M |
| OpenEarthAgent tool server | github.com/mbzuai-oryx/OpenEarthAgent |
| Qwen-Agent framework | github.com/QwenLM/Qwen-Agent |
| OSMnx | osmnx.readthedocs.io |
| OR-Tools VRP | developers.google.com/optimization/routing |
| Copernicus STAC | catalogue.dataspace.copernicus.eu/stac |
| SAM2 | github.com/facebookresearch/sam2 |
| GroundingDINO | github.com/IDEA-Research/GroundingDINO |
| PEFT (LoRA) | huggingface.co/docs/peft |
| TRL (DPO) | huggingface.co/docs/trl |

---

## `.env.example`

```bash
# Copernicus satellite data (register free at dataspace.copernicus.eu)
CDSE_USERNAME=your_email
CDSE_PASSWORD=your_password

# If using GPT-4o for ORC instead of local Qwen3-235B
ORC_USE_API=false
OPENAI_API_KEY=sk-...

# Tool server
TOOL_SERVER_HOST=localhost
TOOL_SERVER_PORT=9000

# vLLM model server ports
ORC_PORT=8000
VRA_PORT=8001
GA_PORT=8002
PA_PORT=8003

# Directories
DATA_DIR=data
MODELS_DIR=models/weights
RESULTS_DIR=results
```

---

## How to run

```bash
# 1. Install
pip install -e .

# 2. Download models (long, run once — ~600GB total)
bash scripts/download_models.sh

# 3. Download public datasets (OEA, ThinkGeo, xBD, Sen1Floods11)
bash scripts/download_datasets.sh

# 4. Start tool server (port 9000)
bash scripts/start_tool_server.sh

# 5. Start vLLM servers (ports 8000-8003)
bash scripts/start_all_vllm.sh

# 6. Train Stage 1 (run once — ~12h total)
python training/stage1_damage_mlp.py
python training/stage1_change_head.py

# 7. Run unit tests
pytest tests/ -v

# 8. Run smoke test on Harvey AOI
bash scripts/smoke_test.sh

# 9. Run available benchmarks NOW
python evaluation/benchmarks/openearth_eval.py   # E2
python evaluation/benchmarks/thinkgeo_eval.py    # E5

# --- WHEN DATASETS ARRIVE (~3 weeks) ---
# 10. Stage 2 LoRA fine-tuning
python training/stage2_lora_vra.py --data data/toolchain_train.jsonl
python training/stage2_lora_ga.py  --data data/toolchain_train.jsonl
python training/stage2_lora_pa.py  --data data/toolchain_train.jsonl

# 11. Run TDRD experiments
python evaluation/benchmarks/tdrd_eval.py   # E3
python evaluation/benchmarks/conflict_eval.py  # E4
python evaluation/benchmarks/rescueadi_eval.py  # E1
python evaluation/ablations/run_all_ablations.py  # A1-A8
```
