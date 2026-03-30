# MAGRF — Multi-Agent Geospatial Reasoning Framework
## Master Build + Eval + Fine-tuning Guide

---

## Models — Under 12GB VRAM, Best Available

**Constraint:** ≤12GB VRAM per model. Needs vision + reasoning. Free open-source. Best tool calling.

| Agent | Model | VRAM | Vision | Thinking | Download |
|---|---|---|---|---|---|
| **VRA** | `Qwen/Qwen3-VL-4B-Instruct` | **~8 GB** | ✅ Yes | ✅ Yes | 8.05 GB |
| **ORC** | `Qwen/Qwen3-4B-Instruct-2507` | **~8 GB** | ❌ No | ❌ No | 8.05 GB |
| **GA** | `Qwen/Qwen3-4B-Instruct-2507` | shared | ❌ No | ❌ No | same as ORC |
| **PA** | `Qwen/Qwen3-4B-Instruct-2507` | shared | ❌ No | ❌ No | same as ORC |

**GPU layout (2× 12GB):**
```
GPU 0 (12GB): VRA  — Qwen3-VL-4B-Instruct       ~8 GB  [vision + thinking]
GPU 1 (12GB): ORC + GA + PA — Qwen3-4B shared   ~8 GB  [text, sequential]
              Prithvi-300M                        ~1.3 GB
              Tool server                         ~0.5 GB
```
Both cards comfortable with headroom. No quantization needed.

**Why these two models:**
- Qwen3-VL-4B: Best vision-language model under 12GB. Native thinking mode. Strong tool calling. Benchmarks above Qwen2.5-VL-7B on several RS tasks despite being smaller.
- Qwen3-4B-Instruct-2507: July 2025 update. Better instruction following, tool use, and logical reasoning than original Qwen3-4B. ORC, GA, PA share one vLLM instance running sequentially — they never run at the same time.

```bash
# Download both models
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir models/weights/qwen3-vl-4b
# 8.05 GB

huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/weights/qwen3-4b-2507
# 8.05 GB

huggingface-cli download ibm-nasa-geospatial/Prithvi-300M \
  --local-dir models/weights/prithvi-300m
# 1.3 GB
```

---

## Datasets to Download NOW

Only the datasets your pipeline directly trains on or evaluates against.

### Training Datasets

**1. xBD (xView2) — Stage 1 damage MLP training**
```
What:    ~19,986 pre/post satellite image pairs, 4-class building damage labels
Size:    ~14.5 GB
Why:     Trains Prithvi damage MLP head — gives VRA 4-class building damage output
Register: xview2.org/challenge (free, 24-48h approval — DO THIS NOW)
Save to: data/xbd/train/images/, data/xbd/train/labels/
         data/xbd/test/images/,  data/xbd/test/labels/
```

**2. FloodNet — Stage 1 flood segmentation**
```
What:    2,343 aerial images, 10-class labels (flooded road, flooded building, water...)
Size:    ~3.6 GB
Why:     Trains flood segmentation head — improves VRA flood extent output (A+B)
```
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ker0sene/FloodNet-Dataset', repo_type='dataset', local_dir='data/floodnet')
"
```

**3. Sen1Floods11 — Stage 1 change detection + phase LSTM**
```
What:    4,831 Sentinel-1/2 pairs, 11 flood events, flood/no-flood labels
Size:    ~13 GB
Why:     Trains temporal change head and phase LSTM (onset/peak/recession/recovery)
```
```bash
aws s3 sync s3://sen1floods11 data/sen1floods11 --no-sign-request
# Alternative:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('isp-uv-es/Sen1Floods11', repo_type='dataset', local_dir='data/sen1floods11')
"
```

**4. LEVIR-CD — Stage 1 SAM2-CD change detection**
```
What:    637 image pairs with per-pixel building change labels (0.5m resolution)
Size:    ~1.5 GB
Why:     Trains SAM2-CD adapter — improves ChangeDetection tool by ~5 IoU pts
```
```bash
pip install gdown
python -c "
import gdown, zipfile
gdown.download('https://drive.google.com/uc?id=1NJ2foGjFrFbB2YNqMFDXuGKbgbFSwsMF',
               'data/levircd/LEVIR-CD.zip')
zipfile.ZipFile('data/levircd/LEVIR-CD.zip').extractall('data/levircd/')
"
```

### Evaluation Datasets

**5. OpenEarthAgent eval split — E2 benchmark**
```
What:    1,169 verified multi-step RS task instances
Size:    ~6 GB
Why:     E2 benchmark — compare against OEA 4B (ArgV baseline 62.10)
```
```bash
git clone https://github.com/mbzuai-oryx/OpenEarthAgent.git data/openearth_repo
cd data/openearth_repo && python scripts/download_data.py --split eval --output ../openearth_agent
```

**6. ThinkGeo / GeoBenchX — E5 benchmark**
```
What:    ~300 GIS routing tasks, solvable + unsolvable
Size:    ~500 MB
Why:     E5 — N3 vs LLM on constrained routing (baseline 42%, target 65%+)
```
```bash
git clone https://github.com/MBZUAI-Oryx/GeoAgent.git data/thinkgeo_repo
cp -r data/thinkgeo_repo/benchmark/ data/thinkgeo/
```

**Verify all downloads:**
```bash
python scripts/verify_downloads.py
# Checks file counts and sizes for all 6 datasets
```

---

## Full File Structure

Build this exactly. Every file described below.

```
magrf/
├── .env                         # CDSE_USERNAME, CDSE_PASSWORD, ports
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── agents.yaml              # Per-agent: model port, tools, max_turns, memory limit
│   ├── models.yaml              # Model IDs, VRAM, quantization, LoRA paths
│   └── tools.yaml               # Tool server host, port, per-tool timeout
│
├── tools/                       # All 28 tools as a FastAPI server
│   ├── server.py                # FastAPI app, all 28 endpoints
│   ├── schemas.py               # Pydantic Input/Output for every tool
│   ├── vision/
│   │   ├── _models.py           # Singleton: GroundingDINO + SAM2 loaded once
│   │   ├── object_detection.py
│   │   ├── segmentation.py
│   │   ├── description.py
│   │   ├── text_to_bbox.py
│   │   ├── region_attribute.py
│   │   ├── counting.py
│   │   ├── ocr.py
│   │   ├── change_detection.py
│   │   ├── draw_box.py
│   │   └── add_text.py
│   ├── gis/
│   │   ├── area_boundary.py
│   │   ├── pois_layer.py
│   │   ├── compute_distance.py
│   │   ├── display_map.py
│   │   ├── bbox_from_geotiff.py
│   │   └── display_geotiff.py
│   ├── spectral/
│   │   ├── add_index_layer.py
│   │   ├── compute_index_change.py
│   │   └── show_index_layer.py
│   ├── math_tools/
│   │   ├── calculator.py
│   │   ├── solver.py
│   │   └── plot.py
│   ├── utility/
│   │   ├── google_search.py
│   │   └── terminate.py
│   └── novel/
│       ├── n1_temporal_stack_loader.py
│       ├── n2_road_damage_scorer.py
│       ├── n3_evacuation_route_planner.py
│       └── n4_prithvi_embed.py
│
├── agents/
│   ├── base_agent.py            # ReAct loop, Hermes parsing, tool dispatch
│   ├── orc.py                   # Orchestrator: decompose, route, judge, compress
│   ├── vra.py                   # Visual Reasoning Agent
│   ├── ga.py                    # Geospatial Agent
│   └── pa.py                    # Planning Agent
│
├── framework/
│   ├── episode_runner.py        # Top-level: runs one full episode
│   ├── mpc/
│   │   ├── message.py           # Message dataclass, all 7 types
│   │   ├── broker.py            # Routes messages, priority queue, TTL
│   │   └── subscriptions.py     # Which agents receive which message types
│   ├── memory/
│   │   ├── episodic_memory.py   # Shared versioned KV store, CCQ tracking
│   │   ├── working_memory.py    # Per-agent 4K token scratchpad, auto-flush
│   │   └── trajectory_log.py   # Append-only, ORC-only write, saves to JSONL
│   └── protocols/
│       ├── deadlock.py          # DFS wait-for graph + Break-and-Replay
│       ├── conflict.py          # Tier-1 confidence gap + Tier-2 ORC judge
│       ├── compression.py       # Safety-aware EM compression, CCQ ≥ 0.98
│       └── safety.py            # RSS check on all routes before output
│
├── models/
│   ├── prithvi/
│   │   ├── encoder.py           # Load frozen Prithvi-300M
│   │   ├── damage_mlp.py        # 2-layer MLP: 768 → 256 → 4 (damage classes)
│   │   └── phase_lstm.py        # 2-layer LSTM: change scores → phase label
│   └── serving/
│       ├── start_vra.sh         # vllm serve Qwen3-VL-4B on GPU 0 port 8001
│       ├── start_orc_ga_pa.sh   # vllm serve Qwen3-4B on GPU 1 port 8002
│       └── start_tool_server.sh # uvicorn tools.server:app port 9000
│
├── training/
│   ├── stage1_damage_mlp.py     # Train Prithvi MLP on xBD — RUN NOW
│   ├── stage1_flood_head.py     # Train flood head on FloodNet — RUN NOW
│   ├── stage1_change_head.py    # Train change head on Sen1Floods11 — RUN NOW
│   ├── stage1_sam2cd.py         # Train SAM2-CD on LEVIR-CD — RUN NOW
│   ├── stage2_lora_vra.py       # LoRA VRA — NEEDS TOOLCHAIN DATASET
│   ├── stage2_lora_ga.py        # LoRA GA — NEEDS TOOLCHAIN DATASET
│   ├── stage2_lora_pa.py        # LoRA PA — NEEDS TOOLCHAIN DATASET
│   ├── stage3_dpo_orc.py        # DPO ORC — NEEDS ALIGNMENT DATASET
│   └── data_prep/
│       ├── extract_xbd_embeddings.py   # Pre-compute Prithvi embeddings for xBD
│       ├── generate_alignment_ab.py    # Type A+B negatives from OEA episodes
│       └── generate_alignment_c.py     # Type C negatives — NEEDS TDRD
│
├── evaluation/
│   ├── metrics.py               # All 19 metrics with formulas
│   ├── run_eval.py              # Main eval script: takes dataset + output path
│   ├── benchmarks/
│   │   ├── openearth_eval.py    # E2 — OEA 1,169 instances
│   │   ├── thinkgeo_eval.py     # E5 — ThinkGeo ~300 tasks
│   │   ├── rescueadi_eval.py    # E1 — needs RescueADI data
│   │   ├── tdrd_eval.py         # E3 — needs TDRD
│   │   └── conflict_eval.py     # E4 — needs TDRD conflict subset
│   └── ablations/
│       ├── ablation_configs.py  # A1-A8 config overrides
│       └── run_ablations.py     # Runs all ablations, saves results
│
├── logs/                        # All run logs go here
│   └── .gitkeep
│
├── results/                     # All eval outputs as JSON go here
│   └── .gitkeep
│
└── trajectories/                # Per-episode trajectory JSONL saved here
    └── .gitkeep
```

---

## The Four Agents — Exactly How They Work

### ORC — Orchestrator

ORC is the only agent that never calls a domain tool. Its job is:
1. Decompose the query into a plan (L1 → L2 → L3 DAG)
2. Send TASK messages to the right agents
3. Receive RESULT messages and check consistency
4. Resolve conflicts (Tier-1 confidence gap, Tier-2 LLM judge if needed)
5. Manage Episodic Memory budget (compress if > 80%)
6. Send SYNC to PA once VRA and GA both return
7. Verify RSS ≥ 1.0 on all routes
8. Write trajectory log and terminate

**ORC routing rules (hardcoded in system prompt — no ambiguity):**
```
IF query contains: image, satellite, building, flood, damage, count, detect, segment,
                   spectral, NDVI, NDWI, change, before/after
→ ASSIGN VRA  (tools: all vision + spectral + N1 + N4)

IF query contains: area, boundary, shelter, hospital, road network, POI, distance,
                   GIS, map, layer, proximity, accessibility
→ ASSIGN GA  (tools: all GIS + N2)

IF query contains: route, path, evacuate, safest, navigate, reach, travel,
                   supply, conflict, re-plan
→ ASSIGN PA  (tools: N3 + Calculator + Solver + Plot)

ORC can assign MULTIPLE agents. VRA and GA run in PARALLEL.
PA waits for SYNC from ORC (which comes after both VRA and GA return).
ORC itself handles: GoogleSearch, Calculator, Terminate — nothing else.
```

**ORC can call the same agent MULTIPLE times.** Example: VRA called at T2, then called
again at T4 with updated flood polygons from EM. PA called with evacuation mode,
then called again with conflict_check mode if routes change. This is explicit and logged.

### VRA — Visual Reasoning Agent

Uses **Qwen3-VL-4B-Instruct** (vision + thinking). Its ReAct loop:

```
[THINK — internal scratchpad]
"I have a Sentinel-2 image. I should first get bounds, then detect buildings,
then run PrithviEmbed for spectral analysis since we have 6 bands."

[ACT — tool call in Hermes format]
{"name": "GetBboxFromGeotiff", "arguments": {"geotiff_path": "data/stacks/aoi_001_T2.tif"}}

[OBSERVE — tool server returns result]
{"gpkg_path": "data/tmp/bbox_001.gpkg", "bbox": [-95.53, 29.68, -95.45, 29.72], "success": true}

[THINK]
"Bounds confirmed. Now detect damaged buildings."

[ACT]
{"name": "ObjectDetection", "arguments": {"image_path": "...", "confidence_threshold": 0.65}}
...
[continues until it has damage_polygons + flood_extent + spectral indices]
[writes all to Episodic Memory]
[sends RESULT to ORC]
```

Tools VRA owns: `GetBboxFromGeotiff, ObjectDetection, SegmentObjectPixels, ImageDescription,
TextToBbox, RegionAttributeDescription, CountGivenObject, OCR, ChangeDetection, DrawBox,
AddText, AddIndexLayer, ComputeIndexChange, ShowIndexLayer, TemporalStackLoader(N1), PrithviEmbed(N4)`

### GA — Geospatial Agent

Uses **Qwen3-4B-Instruct-2507** (text only — never sees raw images, works on GeoPackage paths).

```
[THINK]
"I need the road network for this AOI, then score each segment against VRA's flood extent."

[ACT]
{"name": "GetAreaBoundary", "arguments": {"place_name_or_bbox": [-95.53, 29.68, -95.45, 29.72]}}

[OBSERVE]
{"gpkg_path": "data/tmp/boundary_001.gpkg", "success": true}

[ACT]
{"name": "AddPoisLayer", "arguments": {"gpkg_path": "...", "query_string": "hospital shelter",
                                        "layer_name": "shelters"}}
[OBSERVE] ...

[ACT]
{"name": "RoadDamageScorer", "arguments": {"gpkg_path": "...",
                                             "damage_raster_layer": "flood_extent_T2"}}
[OBSERVE]
{"gpkg_path": "...", "n_roads_scored": 847, "n_impassable": 203, "success": true}

[writes road_damage_scores to EM]
[sends RESULT to ORC]
```

Tools GA owns: `GetAreaBoundary, AddPoisLayer, GetBboxFromGeotiff, ComputeDistance,
DisplayOnMap, DisplayOnGeotiff, RoadDamageScorer(N2)`

### PA — Planning Agent

Uses **Qwen3-4B-Instruct-2507** (shared vLLM server with ORC+GA). Runs after SYNC from ORC.

```
[THINK]
"I have road_damage_scores from GA. I'll find top-3 evacuation routes,
then run conflict_check if multiple epochs exist."

[ACT]
{"name": "EvacuationRoutePlanner",
 "arguments": {"gpkg_path": "...", "road_scores_layer": "road_damage_scores",
               "origins": [[29.70, -95.50]], "destinations": [[29.72, -95.47]],
               "mode": "evacuation", "top_k_routes": 3}}

[OBSERVE]
{"routes": [{"id": "r1", "length_km": 4.2, "eta_min": 51, "rss": 1.0, ...}], "success": true}

[ACT — if multiple epochs exist]
{"name": "EvacuationRoutePlanner", "arguments": {..., "mode": "conflict_check",
                                                  "epoch_T": 2, "epoch_T1": 4}}
[OBSERVE]
{"conflict_detected": true, "newly_blocked_segments": ["seg_id_047"], "success": true}

[if conflict: call EvacuationRoutePlanner again with mode="evacuation" using T+1 scores]

[ACT]
{"name": "Plot", "arguments": {"code": "solution()..."}}

[writes routes + conflict_events to EM]
[sends RESULT to ORC]
```

Tools PA owns: `EvacuationRoutePlanner(N3), ComputeDistance, Calculator, Solver, Plot`

---

## MPC Protocol — Complete

### Message Schema

```python
@dataclass
class Message:
    msg_id:        str              # UUID4
    sender:        str              # "VRA" | "GA" | "PA" | "ORC"
    receiver:      str              # always routed through ORC first
    msg_type:      str              # TASK | RESULT | QUERY | ALERT | SYNC | ERROR | ACK
    payload:       dict
    priority:      int              # 0=heartbeat, 1=normal, 2=high, 3=safety-critical
    confidence:    float            # [0.0, 1.0]
    epoch_ref:     int | None       # which temporal epoch this covers
    token_count:   int
    timestamp_ms:  int
    ttl_ms:        int = 30000      # drop message if not consumed in 30s
```

### Message flow for a full episode

```
USER QUERY
    │
    ▼
ORC: decompose_task()                        [THINK in ORC loop]
    │
    ├──► TASK msg (priority=1) ──► VRA        [parallel]
    └──► TASK msg (priority=1) ──► GA         [parallel]
              │                         │
              │    ReAct loops          │    ReAct loops
              │    Call tools           │    Call tools
              │    Write to EM          │    Write to EM
              │                         │
              ▼                         ▼
          RESULT msg ──► ORC       RESULT msg ──► ORC
              │                         │
              └──────────┬──────────────┘
                         │
                    ORC: consistency check
                    ORC: conflict resolution if needed
                    ORC: EM compression if > 80%
                         │
                    SYNC + TASK msg ──► PA
                         │
                    PA: ReAct loop → N3 evacuation
                    PA: N3 conflict_check if multi-epoch
                    PA: re-plan if conflict detected
                    PA: write routes to EM
                         │
                    RESULT msg ──► ORC
                         │
                    ORC: verify RSS ≥ 1.0
                    ORC: log_trajectory()
                    ORC: Terminate(final_answer)
```

### Subscription map (who receives what)

```python
SUBSCRIPTIONS = {
    "VRA":  ["TASK", "HEARTBEAT"],
    "GA":   ["TASK", "HEARTBEAT"],
    "PA":   ["TASK", "SYNC", "ALERT"],
    "ORC":  ["TASK", "RESULT", "QUERY", "ALERT", "SYNC", "ERROR", "ACK"],
}
# VRA and GA never talk to each other directly.
# All cross-agent data goes through Episodic Memory, not messages.
```

### Conflict resolution

```python
def resolve_conflict(vra_output, ga_output):
    conf_gap = abs(vra_output.confidence - ga_output.confidence)

    # Tier-1: confidence comparator (~0ms)
    if conf_gap > 0.20:
        winner = vra_output if vra_output.confidence > ga_output.confidence else ga_output
        return winner, "tier1_confidence_gap"

    if max(vra_output.confidence, ga_output.confidence) >= 0.85:
        winner = vra_output if vra_output.confidence >= ga_output.confidence else ga_output
        return winner, "tier1_high_confidence"

    # Tier-2: ORC LLM judge (called via ORC's own vLLM)
    judgment = orc_llm_judge(vra_output, ga_output, em_context)
    return judgment.winner, "tier2_llm_judge"

# Safety override — unconditional, cannot be bypassed
def safety_check(routes):
    for route in routes:
        if route.rss < 1.0:
            # Force PA to re-route before any output is returned
            return False, "route_safety_violation"
    return True, "safe"
```

### Deadlock detection

```python
# After every ACK or RESULT message:
def check_deadlock(wait_graph):
    # DFS on wait_graph: {agent: set_of_agents_it_waits_for}
    # O(|V|+|E|) = O(4+6) = negligible
    cycle = dfs_find_cycle(wait_graph)
    if cycle:
        # Break: lowest priority agent in cycle produces provisional output
        # flagged uncertain=True in EM
        # Clean re-run scheduled after episode
        breaker = min(cycle, key=lambda a: PRIORITY[a])
        produce_provisional(breaker)
    # Timeout: no ACK within 30s → restart agent from last EM snapshot
```

---

## Context Management — Episodic Memory

```python
class EpisodicMemory:
    SOFT_LIMIT = 32_000    # tokens — trigger compression at 80% = 25,600
    HARD_LIMIT = 64_000    # tokens — hard stop

    # These keys are NEVER compressed or evicted
    SAFETY_KEYS = {
        "damage_polygons",      # building damage — never lose these
        "flood_extent",         # flood boundary — never lose these
        "impassable_roads",     # blocked segments — never lose these
        "conflict_events",      # route invalidations — never lose these
    }

    # Keys evicted first (largest, least safety-critical)
    EVICT_FIRST = ["prithvi_embed_", "spectral_raw_"]

    # Keys summarised (replace with statistics)
    SUMMARISE = ["change_map_", "spectral_index_"]
```

**Compression trigger:**
```python
# Called by ORC after every EM write
if em.budget_used_fraction() > 0.80:
    compress_context(em)

def compress_context(em):
    # Step 1: evict raw embeddings
    for key in em.get_keys_matching(EVICT_FIRST):
        em.evict(key)

    # Step 2: summarise spectral indices
    for key in em.get_keys_matching(SUMMARISE):
        summary = compute_statistics(em.read(key))  # mean, std, max per epoch
        em.replace_with_summary(key, summary)

    # Step 3: verify CCQ
    ccq = em.compute_ccq()
    if ccq < 0.98:
        # Abort compression — restore from snapshot
        em.rollback()
        log_compression_failure(ccq)
```

**Working Memory:** per-agent, per-turn, 4K token limit, automatically flushed after each tool call is committed to EM. Agents never see each other's working memory.

---

## Novel Tools — Complete Implementation Spec

### N1: TemporalStackLoader

```python
# tools/novel/n1_temporal_stack_loader.py
# Input: aoi_bbox [W,S,E,N], date_range [start, end], sensor, max_cloud_pct
# Output: GeoPackage with N co-registered raster layers + metadata JSON
#
# How:
# 1. Query Copernicus STAC API for matching scenes
# 2. Download GeoTIFFs
# 3. ECC sub-pixel co-registration (OpenCV) → all epochs aligned to T1
# 4. Save as named layers in GeoPackage: epoch_1, epoch_2, ... epoch_N
# 5. Return: {gpkg_path, n_epochs, dates, resolution_m, crs, success}
#
# Fallback: if local_stack_path provided → skip download, load from disk
# Use case: smoke test with manually downloaded Harvey tiles
```

### N2: RoadDamageScorer

```python
# tools/novel/n2_road_damage_scorer.py
# Input: gpkg_path, road_layer_name, damage_raster_layer, buffer_meters=15
# Output: GeoPackage with road_damage_scores layer added
#         each edge: {score: float[0-1], status: passable|degraded|impassable}
#
# How:
# 1. Load road network from OSMnx (or from GeoPackage if already downloaded)
# 2. Load damage/flood polygons from damage_raster_layer
# 3. Buffer each road segment by 15m (OSMnx to UTM, then buffer, then back)
# 4. For each segment: score = max(building_damage_weight, flood_intersection_score)
#    building_damage_weights: {0:0.0, 1:0.3, 2:0.7, 3:1.0}
#    flood_score: 0.85 if segment intersects flood polygon
# 5. status: passable (<0.4), degraded (0.4-0.6), impassable (≥0.6)
# 6. Append as new layer "road_damage_scores" to GeoPackage
# 7. Return: {gpkg_path, n_roads_scored, n_impassable, n_degraded, n_passable, success}
```

### N3: EvacuationRoutePlanner

```python
# tools/novel/n3_evacuation_route_planner.py
# Input: gpkg_path, road_scores_layer, origins [[lat,lon],...], destinations [[lat,lon],...],
#         mode: "evacuation" | "supply" | "conflict_check",
#         top_k_routes=3, epoch_T=None, epoch_T1=None
# Output: {routes: [{id, origin, dest, length_km, eta_min, rss, geometry_geojson,
#                    nl_instructions, segments: [{osm_id, damage_score},...]}],
#          conflict_detected: bool, newly_blocked_segments: [...], success: bool}
#
# How:
# EVACUATION mode:
#   1. Build NetworkX DiGraph from road_scores_layer
#      edge weight = (1.0 + damage_score * 3.0) * length_m
#      remove edges where status == "impassable"
#   2. A* from each origin to nearest K destinations
#   3. RSS = safe_segments / total_segments for each route
#   4. Sort by (RSS descending, length ascending)
#   5. Return top_k_routes
#
# SUPPLY mode:
#   Same as evacuation but swap origins/destinations + use OR-Tools VRP for multi-depot
#
# CONFLICT_CHECK mode:
#   1. Load routes from epoch_T (stored in EM or re-compute)
#   2. Load road_scores for epoch_T1
#   3. For each route segment: if status was passable/degraded at T and impassable at T+1
#      → flag as conflict
#   4. Return conflict_detected=True if any segment invalidated
#
# CRITICAL: every route's segments must be populated with real damage_score values
# RSS must be computed from these — never hardcoded to 1.0
```

### N4: PrithviEmbed

```python
# tools/novel/n4_prithvi_embed.py
# Input: geotiff_path, date_list, output_type: "change_score"|"spectral_index"|"embedding"|"phase_label"
#        index_type (for spectral_index): "NDVI"|"NDWI"|"NBR"|"dNBR"|"NDBI"
# Output: {output_path, output_type, shape, phase_label (if phase_label mode), success}
#
# How (per output_type):
# "spectral_index":  compute index from 6-band array using numpy formulas
#                    NDWI=(Green-NIR)/(Green+NIR), NBR=(NIR-SWIR2)/(NIR+SWIR2), etc.
#                    Save as GeoTIFF
# "change_score":    run Prithvi forward pass on T1 and TN → diff embeddings → magnitude map
#                    Save as float32 GeoTIFF normalised to [0,1]
# "embedding":       run Prithvi forward pass → (H/16, W/16, 768) patch embeddings
#                    Save as .npy
# "phase_label":     load trained phase LSTM → run on change_score time series
#                    Return scalar: "onset"|"peak"|"recession"|"recovery"
#                    Fallback: zero-shot heuristic from change_score magnitude if LSTM not trained
```

---

## Metrics — All 19, Mapped to Datasets

### Group A: Inherited RS Metrics (compare with prior work)

| Metric | Formula | Measured on |
|---|---|---|
| `IoU` | TP/(TP+FP+FN) pixel-level per class | xBD test (damage classes), FloodNet test (flood classes) |
| `mIoU` | mean IoU over C classes | OEA eval (spatial outputs), xBD (4 damage classes) |
| `weighted_F1` | per-class F1 weighted by frequency | xBD primary metric — target ≥ 0.71 |
| `Inst` | correct_answers / total × 100 (±5% numeric) | OEA eval — baseline 99.51 |
| `Tool` | correct_tool_names / total_calls × 100 | OEA eval — baseline 97.18 |
| `ArgN` | correct_arg_names / total_arg_slots × 100 | OEA eval — baseline 96.08 |
| `ArgV` | correct_arg_values / total_arg_slots × 100 | OEA eval — baseline 62.10 **(key target)** |
| `Summ` | ROUGE-L F1 vs GT summary × 100 | OEA eval — baseline 83.64 |
| `TSR` | correct_episodes / total × 100 | ThinkGeo, RescueADI (when available) |
| `HRR` | correctly_rejected_unsolvable / total_unsolvable × 100 | ThinkGeo unsolvable tasks |

### Group B: Novel Coordination Metrics (new contributions)

| Metric | Formula | Measured on |
|---|---|---|
| `MGAR` | episodes where ALL subtasks complete AND answer correct / total × 100 | All multi-agent episodes |
| `RSS` | route_segments_with_score < 0.6 / total_segments × 100 | All routing episodes |
| `MTCS` | epoch_pairs_with_consistent_phase / total_epoch_pairs × 100 | Multi-epoch episodes |
| `TLS` | GT_tool_calls / actual_tool_calls (>1.0 = more efficient) | All episodes |
| `CRR` | ORC_resolutions_matching_GT / total_conflicts × 100 | Episodes with conflicts |
| `DDF1` | F1 on deadlock detection (injected test set) | Deadlock injection test |
| `CCQ` | safety_facts_after_compression / safety_facts_before ≥ 0.98 | Episodes with compression |
| `CDF1` | F1 on route invalidation detection (conflict subset) | TDRD conflict subset (later) |
| `ReSR` | re_plans_with_RSS=1.0 / conflicts_detected × 100 | TDRD conflict subset (later) |

### Metrics per dataset

```
OpenEarthAgent eval (E2):
  → Inst, Tool, ArgN, ArgV, Summ         [primary: ArgV vs 62.10 baseline]
  → MGAR, TLS                             [novel: multi-agent efficiency]

ThinkGeo (E5):
  → TSR (simple GIS), TSR (routing), HRR  [primary: routing TSR vs 42% baseline]
  → MGAR, TLS                             [novel]
  → Ablation A8: full vs no N1/N2/N3      [key delta: N3 contribution]

xBD test (Stage 1 eval):
  → weighted_F1, IoU per class            [target: weighted F1 ≥ 0.71]

FloodNet test (Stage 1 eval):
  → mIoU, IoU (flooded vs non-flooded)   [target: flood IoU ≥ 0.78]

Sen1Floods11 test (Stage 1 eval):
  → Change detection IoU                  [target: ≥ 0.72]

Harvey smoke test:
  → RSS, MTCS, TLS, MGAR, ACL (wall time) [qualitative case study for paper]

[AFTER TDRD ARRIVES]:
  → All 19 metrics on TDRD test split
  → CDF1, ReSR on 468 conflict subset
  → Full ablation A1-A8 on TDRD
```

---

## Trajectory Log — What Gets Saved Per Episode

Every episode saves a JSONL record to `trajectories/{episode_id}.jsonl`:

```json
{
  "episode_id": "uuid4",
  "query": "Count destroyed buildings in Meyerland and find safest shelter route",
  "source": "openearth_eval",
  "difficulty_tier": 2,
  "tool_calls": [
    {
      "step": 1,
      "agent": "ORC",
      "tool": "decompose_task",
      "input_args": {"query": "..."},
      "output": {"plan": {"VRA": ["ObjectDetection", "CountGivenObject"], "GA": ["AddPoisLayer"]}},
      "thinking_trace": "I need to split this into visual counting and GIS lookup...",
      "latency_ms": 340,
      "success": true
    },
    {
      "step": 2,
      "agent": "VRA",
      "tool": "ObjectDetection",
      "input_args": {"image_path": "data/openearth_agent/eval_images/tile_042.tif", "confidence_threshold": 0.65},
      "output": {"labels": ["building","building"], "scores": [0.89, 0.82], "success": true},
      "trigger_msg_id": "msg_001",
      "latency_ms": 2100,
      "success": true
    }
  ],
  "mpc_messages": [
    {"id":"msg_001","sender":"ORC","receiver":"VRA","type":"TASK","confidence":1.0,"epoch_ref":null}
  ],
  "conflict_events": [],
  "em_writes": [
    {"key": "detection_result", "agent": "VRA", "token_cost": 240}
  ],
  "final_answer": {"count": 47, "shelter": "Memorial Hospital"},
  "gt_answer": {"count": 47, "shelter": "Memorial Hospital"},
  "correct": true,
  "episode_metrics": {
    "mgar": 1, "rss": 1.0, "tls": 0.94,
    "n_tool_calls": 8, "n_mpc_messages": 6,
    "wall_time_ms": 12400
  }
}
```

This trajectory log is the seed for the Toolchain Dataset (Person 2) and Alignment Dataset.

---

## Step-by-Step Execution Plan

### PHASE 1 — Download and Serve (Day 1)

```bash
# Step 1: Download models
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir models/weights/qwen3-vl-4b
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/weights/qwen3-4b-2507
huggingface-cli download ibm-nasa-geospatial/Prithvi-300M --local-dir models/weights/prithvi-300m

# Step 2: Download evaluation datasets (run in parallel background)
# Terminal 1:
git clone https://github.com/mbzuai-oryx/OpenEarthAgent.git data/openearth_repo &
git clone https://github.com/MBZUAI-Oryx/GeoAgent.git data/thinkgeo_repo &

# Terminal 2 (training data):
aws s3 sync s3://sen1floods11 data/sen1floods11 --no-sign-request &
python -c "from huggingface_hub import snapshot_download; snapshot_download('ker0sene/FloodNet-Dataset', repo_type='dataset', local_dir='data/floodnet')" &
# xBD: register at xview2.org NOW, download when approved

# Step 3: Start model servers
# GPU 0 — VRA
CUDA_VISIBLE_DEVICES=0 vllm serve models/weights/qwen3-vl-4b \
  --port 8001 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768 &

# GPU 1 — ORC + GA + PA (shared)
CUDA_VISIBLE_DEVICES=1 vllm serve models/weights/qwen3-4b-2507 \
  --port 8002 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768 &

# Step 4: Start tool server
uvicorn tools.server:app --port 9000 --workers 4 &

# Step 5: Verify everything
python scripts/verify_system.py
# Checks: all 4 servers respond, tool server healthy, eval data present
```

### PHASE 2 — Build Framework (Days 1-4)

Build in this exact order. Each has a test gate.

```bash
# Gate 1: Tool server
pytest tests/test_tools/ -v           # all 28 tools must pass

# Gate 2: Memory + MPC
pytest tests/test_framework/ -v       # memory, MPC, deadlock, conflict must pass

# Gate 3: Agents (with mock vLLM responses)
pytest tests/test_agents/ -v

# Gate 4: Integration (mock full episode)
pytest tests/test_integration/ -v

# Gate 5: Real inference smoke test
python scripts/smoke_test_oea.py      # run 5 OEA samples with real models
# Expected: Inst ~95%, at least 4/5 correct, all tools real (not mock)
```

### PHASE 3 — Stage 1 Training (Days 3-5, start as soon as xBD available)

```bash
# Pre-compute Prithvi embeddings for xBD (overnight job, ~4-6h)
python training/data_prep/extract_xbd_embeddings.py \
  --xbd_dir data/xbd \
  --output_dir data/xbd_embeddings \
  2>&1 | tee logs/xbd_embeddings.log &

# Train damage MLP once embeddings ready (~2h on GPU)
python training/stage1_damage_mlp.py \
  --embeddings_dir data/xbd_embeddings \
  --output models/prithvi/damage_mlp.pt \
  --epochs 50 \
  2>&1 | tee logs/stage1_damage_mlp.log

# Train flood head on FloodNet (~3h)
python training/stage1_flood_head.py \
  --data_dir data/floodnet \
  --output models/prithvi/flood_head.pt \
  --epochs 40 \
  2>&1 | tee logs/stage1_flood.log

# Train change head + phase LSTM on Sen1Floods11 (~4h)
python training/stage1_change_head.py \
  --data_dir data/sen1floods11 \
  --output_change models/prithvi/change_head.pt \
  --output_lstm models/prithvi/phase_lstm.pt \
  --epochs 30 \
  2>&1 | tee logs/stage1_change.log

# Update configs/models.yaml to point to trained heads:
# prithvi.damage_mlp_path: models/prithvi/damage_mlp.pt
# prithvi.flood_head_path: models/prithvi/flood_head.pt
# prithvi.phase_lstm_path: models/prithvi/phase_lstm.pt
```

**Stage 1 targets:**
```
Damage MLP:     weighted F1 ≥ 0.71 on xBD test
Flood head:     IoU ≥ 0.78 on FloodNet test
Change head:    change IoU ≥ 0.72 on Sen1Floods11 test
Phase LSTM:     phase accuracy ≥ 0.80 on Sen1Floods11 val
```

### PHASE 4 — Run E2 and E5 Evals (Days 4-6)

```bash
# E2: OpenEarthAgent — PRIMARY eval, compare with OEA 4B
python evaluation/benchmarks/openearth_eval.py \
  --data data/openearth_agent/eval.jsonl \
  --output results/e2_oea_baseline.json \
  --log_trajectories trajectories/e2/ \
  2>&1 | tee logs/e2_oea.log

# Print results
python evaluation/run_eval.py --results results/e2_oea_baseline.json --dataset oea

# E5: ThinkGeo — routing is the headline number
python evaluation/benchmarks/thinkgeo_eval.py \
  --data data/thinkgeo/eval.jsonl \
  --output results/e5_thinkgeo_baseline.json \
  --log_trajectories trajectories/e5/ \
  2>&1 | tee logs/e5_thinkgeo.log

python evaluation/run_eval.py --results results/e5_thinkgeo_baseline.json --dataset thinkgeo

# A8 ablation on ThinkGeo (N3 vs no N3)
python evaluation/ablations/run_ablations.py \
  --ablation A8 \
  --benchmark thinkgeo \
  --output results/ablation_a8_thinkgeo.json \
  2>&1 | tee logs/ablation_a8.log
```

**What to look for immediately:**

E2 — ArgV number: if below 55, improve VRA system prompt with explicit arg format examples.
E5 — Routing TSR: if below 42% (same as baseline), ORC is not calling PA. Check routing rule.

```bash
# Quick diagnostic: are tools being called correctly?
python evaluation/run_eval.py \
  --results results/e2_oea_baseline.json \
  --show_failures \
  --top_n 10
# Shows the 10 most common failure modes
```

### PHASE 5 — Verbose Agent Behaviour Analysis

```bash
# After E2 and E5, analyse how ORC is delegating
python scripts/analyse_trajectories.py \
  --trajectories_dir trajectories/e2/ \
  --output reports/e2_agent_analysis.txt

# This prints:
# - How many times each agent was called
# - Which tools were most called
# - Where ORC routed to wrong agent (if any)
# - Mean tool latencies
# - Mean episode wall time
# - Cases where PA was called instead of GA or vice versa
# - Cases where same agent called multiple times in one episode
# - Confidence scores per agent per episode
```

### PHASE 6 — Stage 2 LoRA Fine-tuning (after Toolchain Dataset arrives, ~Day 16)

```bash
# These scripts are ready. Fire all 3 in parallel the moment data lands.

# VRA LoRA (~12h on 1x GPU)
CUDA_VISIBLE_DEVICES=0 python training/stage2_lora_vra.py \
  --data data/toolchain_train.jsonl \
  --base_model models/weights/qwen3-vl-4b \
  --output models/lora/vra_lora \
  --lora_r 16 --lora_alpha 32 \
  --epochs 3 --lr 2e-4 \
  2>&1 | tee logs/stage2_lora_vra.log &

# GA LoRA (~8h)
CUDA_VISIBLE_DEVICES=1 python training/stage2_lora_ga.py \
  --data data/toolchain_train.jsonl \
  --base_model models/weights/qwen3-4b-2507 \
  --output models/lora/ga_lora \
  --lora_r 16 --lora_alpha 32 \
  --epochs 3 --lr 2e-4 \
  2>&1 | tee logs/stage2_lora_ga.log &

# PA LoRA (~8h, can share GPU with GA if sequential)
python training/stage2_lora_pa.py \
  --data data/toolchain_train.jsonl \
  --base_model models/weights/qwen3-4b-2507 \
  --output models/lora/pa_lora \
  --lora_r 16 --lora_alpha 32 \
  --epochs 3 --lr 2e-4 \
  2>&1 | tee logs/stage2_lora_pa.log

# After LoRA training: update configs/models.yaml:
# vra.lora_path: models/lora/vra_lora
# ga.lora_path:  models/lora/ga_lora
# pa.lora_path:  models/lora/pa_lora

# Restart vLLM servers with LoRA adapters:
# vllm serve ... --lora-modules vra=models/lora/vra_lora

# Re-run E2 and E5 with LoRA
python evaluation/benchmarks/openearth_eval.py \
  --output results/e2_oea_lora.json ...

python evaluation/benchmarks/thinkgeo_eval.py \
  --output results/e5_thinkgeo_lora.json ...

# Compare baseline vs LoRA
python evaluation/run_eval.py \
  --compare results/e2_oea_baseline.json results/e2_oea_lora.json \
  --label "Baseline" "After Stage 2 LoRA"
```

**Stage 2 targets (on OEA eval):**
```
ArgV: baseline ~58-62% → target ~67%  (+5-9 pts from LoRA)
Tool: baseline ~93-95% → target ~97%
Routing TSR on ThinkGeo: no direct change (routing is N3, not LLM)
```

### PHASE 7 — TDRD Experiments (after TDRD arrives, ~Day 16)

```bash
# E3: TDRD full eval
python evaluation/benchmarks/tdrd_eval.py \
  --data data/tdrd_test.jsonl \
  --output results/e3_tdrd.json \
  --log_trajectories trajectories/e3/ \
  2>&1 | tee logs/e3_tdrd.log

# E4: Conflict re-planning
python evaluation/benchmarks/conflict_eval.py \
  --data data/tdrd_conflict_subset.jsonl \
  --output results/e4_conflict.json \
  2>&1 | tee logs/e4_conflict.log

# Full ablation suite A1-A8 on TDRD
python evaluation/ablations/run_ablations.py \
  --all \
  --benchmark tdrd \
  --output_dir results/ablations/ \
  2>&1 | tee logs/ablations_tdrd.log

# Generate Type C alignment negatives (unsafe routes)
python training/data_prep/generate_alignment_c.py \
  --tdrd_data data/tdrd_train.jsonl \
  --output data/alignment/type_c_unsafe_routes.jsonl

# Stage 3 DPO for ORC (optional — improves CRR and TLS)
python training/stage3_dpo_orc.py \
  --data data/alignment/ \
  --base_model models/weights/qwen3-4b-2507 \
  --output models/lora/orc_dpo \
  2>&1 | tee logs/stage3_dpo.log

# Final re-run of all experiments with DPO-tuned ORC
# Then compile all results for paper
```

---

## Summary of Experiments and What Each Produces for the Paper

| Experiment | When | Input | Output | Paper table |
|---|---|---|---|---|
| Stage 1 eval (damage MLP) | Phase 3 | xBD test | weighted F1 | Table in Training section |
| E2: OpenEarthAgent baseline | Phase 4 | OEA eval 1,169 instances | Inst/Tool/ArgN/ArgV/Summ | **Table 1** |
| E5: ThinkGeo baseline | Phase 4 | ThinkGeo ~300 tasks | TSR/routing TSR/HRR | **Table 1** |
| A8: No novel tools | Phase 4 | ThinkGeo routing tasks | TSR delta | Table 1 ablation row |
| Harvey smoke test | Phase 4 | Real Harvey Sentinel data | RSS/MTCS/qualitative | Case study Figure |
| E2: After LoRA | Phase 6 | OEA eval | ArgV improvement | Table 1 LoRA row |
| E3: TDRD full | Phase 7 | TDRD test 350 samples | All 9 novel metrics | **Table 2** |
| E4: Conflict re-planning | Phase 7 | TDRD conflict 468 samples | CDF1/ReSR | **Table 3** |
| A1-A7: TDRD ablations | Phase 7 | TDRD test | Per-metric delta | **Table 4** |

---

## Before Starting — One Verification Command

```bash
python - << 'PYEOF'
import subprocess, requests, sys
from pathlib import Path

print("=== MAGRF System Check ===\n")
checks = {
    "VRA server (Qwen3-VL-4B)":    "http://localhost:8001/v1/models",
    "ORC/GA/PA server (Qwen3-4B)": "http://localhost:8002/v1/models",
    "Tool server":                   "http://localhost:9000/health",
}
data_checks = {
    "OEA eval data":    "data/openearth_agent/eval.jsonl",
    "ThinkGeo data":    "data/thinkgeo",
    "FloodNet train":   "data/floodnet/train",
    "Sen1Floods11":     "data/sen1floods11",
}

all_ok = True
for name, url in checks.items():
    try:
        r = requests.get(url, timeout=5)
        print(f"  OK   {name}")
    except:
        print(f"  FAIL {name} — start server first")
        all_ok = False

for name, path in data_checks.items():
    exists = Path(path).exists()
    print(f"  {'OK' if exists else 'MISS'} {name}")
    if not exists: all_ok = False

print(f"\nReady: {all_ok}")
if not all_ok:
    print("Fix failing checks before running any evaluation.")
PYEOF
```
