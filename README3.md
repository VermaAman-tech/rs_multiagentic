# MAGRF — Next Steps: Real Tools, Real Models, Real Results
## Everything completable before TDRD and Toolchain arrive

**Current state (from status doc):**
- Framework architecture: ✅ complete
- MPC, memory, protocols, episode runner: ✅ complete
- All 28 tools: ⚠️ mock outputs only — need real implementations
- Models: ✅ downloaded but not serving
- Tests: ✅ passing (on mock data)
- E2/E5 benchmarks: ⚠️ running on proxy data, not real model inference

**The single biggest gap:** tools return mock responses. Until N1/N2/N3/N4 and the vision
tools return real outputs, every metric you compute is fiction. Fix this first.

---

## Priority 1 — Fix ORC model size (do this today)

Before anything else. The 235B ORC cannot run experiments efficiently.

```bash
# Download Qwen3-30B-A3B (replaces 235B ORC)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-30B-A3B',
                  local_dir='models/weights/qwen3-30b-a3b')
print('Done')
"

# Update configs/models.yaml — change orc section to:
# model_id: Qwen/Qwen3-30B-A3B
# tensor_parallel: 1
# fp8_quantization: true

# Restart ORC vLLM server
vllm serve models/weights/qwen3-30b-a3b \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768

# GA and PA: share one server (they run sequentially, not in parallel)
# Update configs/models.yaml: pa.vllm_port = 8002 (same as ga)
```

**Why this is fine:** Qwen3-30B-A3B scores 68+ on BFCL-V4 tool-calling — better than
GPT-4o mini. 3B active params means fast inference. For ORC's jobs (decomposition,
conflict resolution, routing decisions) this is more than sufficient.

**VRAM after swap on 2x A100 80GB:**
```
GPU 0: ORC (30B-A3B, ~20GB) + GA/PA shared (32B fp8, ~28GB) + Prithvi (~1.3GB) = ~50GB ✓
GPU 1: VRA (7B, ~14GB) = ~14GB ✓  (or upgrade to VL-32B later)
```

---

## Priority 2 — Implement real tools (this is the core work)

Replace every mock return with real logic. Do them in this order — later tools depend on earlier ones.

### 2A. Vision tools — wrap GroundingDINO + SAM2

All 10 vision tools in `tools/vision/` currently return mock dicts.
Replace with actual model calls. You only need two models: GroundingDINO (detection/grounding)
and SAM2 (segmentation). All 10 tools are just different ways of calling these two.

```bash
# Install
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/sam2.git

# Download weights
mkdir -p models/weights/groundingdino models/weights/sam2

python -c "
from huggingface_hub import hf_hub_download
# GroundingDINO
hf_hub_download('IDEA-Research/grounding-dino-tiny',
                filename='pytorch_model.bin',
                local_dir='models/weights/groundingdino')
# SAM2
hf_hub_download('facebook/sam2.1-hiera-large',
                filename='sam2.1_hiera_large.pt',
                local_dir='models/weights/sam2')
"
```

Create `tools/vision/_models.py` — a singleton loader so models load once and stay in memory:

```python
# tools/vision/_models.py
import torch
from groundingdino.util.inference import load_model as load_gdino, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

_gdino = None
_sam2  = None

def get_gdino():
    global _gdino
    if _gdino is None:
        _gdino = load_gdino(
            "models/weights/groundingdino/GroundingDINO_SwinT_OGC.py",
            "models/weights/groundingdino/pytorch_model.bin",
            device="cuda"
        )
    return _gdino

def get_sam2():
    global _sam2
    if _sam2 is None:
        model = build_sam2("sam2.1_hiera_large.yaml",
                           "models/weights/sam2/sam2.1_hiera_large.pt",
                           device="cuda")
        _sam2 = SAM2ImagePredictor(model)
    return _sam2
```

Now replace mock returns in each vision tool. Example for ObjectDetection:

```python
# tools/vision/object_detection.py
import numpy as np
from PIL import Image
from groundingdino.util.inference import predict
from tools.vision._models import get_gdino

def run(req):
    try:
        img = Image.open(req.image_path).convert("RGB")
        img_np = np.array(img)
        model = get_gdino()
        text_prompt = ", ".join(req.object_classes) if req.object_classes else "building . road . vehicle . person"
        boxes, logits, phrases = predict(
            model=model,
            image=img_np,
            caption=text_prompt,
            box_threshold=req.confidence_threshold,
            text_threshold=0.25,
        )
        return {
            "labels": phrases,
            "boxes": boxes.tolist(),
            "scores": logits.tolist(),
            "success": True,
        }
    except Exception as e:
        return {"labels": [], "boxes": [], "scores": [], "success": False, "error": str(e)}
```

Do the same pattern for all 10 vision tools:

| Tool | GroundingDINO call | SAM2 call | Notes |
|---|---|---|---|
| ObjectDetection | predict(prompt=object_classes) | — | Returns boxes + labels |
| SegmentObjectPixels | predict(prompt=object_name) → boxes | set_image + predict(box=boxes) | Returns mask + pixel count |
| TextToBbox | predict(prompt=text_description) | — | Returns single best box |
| CountGivenObject | predict(prompt=object_name) | — | Returns len(boxes) |
| ImageDescription | — | — | Use VRA's LLM directly via ORC query |
| RegionAttributeDescription | crop image to bbox | describe crop via LLM | Crop + VRA LLM call |
| ChangeDetection | run on both images | diff masks | Simple pixel diff on SAM2 masks |
| DrawBox | PIL ImageDraw | — | Pure Python, no model |
| AddText | PIL ImageDraw | — | Pure Python, no model |
| OCR | easyocr or pytesseract | — | `pip install easyocr` |

### 2B. GIS tools — wire up OSMnx and rasterio

All 6 GIS tools in `tools/gis/` return mocks. Real implementations:

```python
# tools/gis/area_boundary.py
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box as sbox

def run(req):
    try:
        if isinstance(req.place_name_or_bbox, str):
            gdf = ox.geocode_to_gdf(req.place_name_or_bbox)
        else:
            bbox = req.place_name_or_bbox  # [west, south, east, north]
            gdf = gpd.GeoDataFrame(geometry=[sbox(*bbox)], crs="EPSG:4326")
        if req.buffer:
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
            gdf_proj["geometry"] = gdf_proj.geometry.buffer(req.buffer)
            gdf = gdf_proj.to_crs("EPSG:4326")
        out_path = f"data/tmp/boundary_{hash(str(req))}.gpkg"
        gdf.to_file(out_path, driver="GPKG")
        return {"gpkg_path": out_path, "success": True}
    except Exception as e:
        return {"gpkg_path": None, "success": False, "error": str(e)}
```

```python
# tools/gis/pois_layer.py
import osmnx as ox
import geopandas as gpd

def run(req):
    try:
        gdf = gpd.read_file(req.gpkg_path)
        bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
        tags = _query_to_osm_tags(req.query_string)
        pois = ox.features_from_bbox(
            north=bbox[3], south=bbox[1],
            east=bbox[2], west=bbox[0],
            tags=tags
        )
        pois["geometry"] = pois.geometry.centroid
        pois.to_file(req.gpkg_path, layer=req.layer_name, driver="GPKG")
        return {"gpkg_path": req.gpkg_path, "poi_count": len(pois), "success": True}
    except Exception as e:
        return {"gpkg_path": req.gpkg_path, "poi_count": 0, "success": False, "error": str(e)}

def _query_to_osm_tags(query):
    q = query.lower()
    if "shelter" in q or "hospital" in q:
        return {"amenity": ["hospital", "clinic", "shelter"], "emergency": ["shelter"]}
    if "road" in q or "street" in q:
        return {"highway": True}
    if "building" in q:
        return {"building": True}
    return {"amenity": True}  # fallback
```

```python
# tools/gis/compute_distance.py
import geopandas as gpd
from shapely.ops import nearest_points

def run(req):
    try:
        source = gpd.read_file(req.gpkg_path, layer=req.source_layer)
        target = gpd.read_file(req.gpkg_path, layer=req.target_layer)
        # Project to UTM for metric distances
        utm = source.estimate_utm_crs()
        source_proj = source.to_crs(utm)
        target_proj = target.to_crs(utm)
        distances = source_proj.geometry.apply(
            lambda g: target_proj.distance(g).min()
        )
        summary = {
            "mean_distance_m": float(distances.mean()),
            "min_distance_m":  float(distances.min()),
            "max_distance_m":  float(distances.max()),
        }
        result_layer = f"distances_{req.source_layer}_{req.target_layer}"
        source["distance_m"] = distances.values
        source.to_file(req.gpkg_path, layer=result_layer, driver="GPKG")
        return {"summary": summary, "layer_name": result_layer, "success": True}
    except Exception as e:
        return {"summary": {}, "layer_name": None, "success": False, "error": str(e)}
```

```python
# tools/gis/display_map.py
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path

def run(req):
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        colours = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]
        for i, layer_name in enumerate(req.layer_names):
            gdf = gpd.read_file(req.gpkg_path, layer=layer_name)
            gdf.to_crs("EPSG:3857").plot(ax=ax, color=colours[i % len(colours)],
                                          alpha=0.6, label=layer_name)
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass  # No internet — skip basemap
        ax.legend()
        out_path = f"data/tmp/map_{hash(str(req))}.png"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"image_path": out_path, "success": True}
    except Exception as e:
        return {"image_path": None, "success": False, "error": str(e)}
```

### 2C. Spectral tools — these are mostly pure rasterio, implement directly

```python
# tools/spectral/add_index_layer.py
import numpy as np, rasterio
from pathlib import Path

INDEX_FORMULAS = {
    "NDVI":  lambda b: (b[3]-b[2])/(b[3]+b[2]+1e-8),
    "NDWI":  lambda b: (b[1]-b[3])/(b[1]+b[3]+1e-8),
    "NBR":   lambda b: (b[3]-b[5])/(b[3]+b[5]+1e-8) if b.shape[0]>5 else (b[3]-b[3])/(1e-8),
    "dNBR":  lambda b: (b[3]-b[5])/(b[3]+b[5]+1e-8) if b.shape[0]>5 else np.zeros(b.shape[1:]),
    "NDBI":  lambda b: (b[4]-b[3])/(b[4]+b[3]+1e-8) if b.shape[0]>4 else np.zeros(b.shape[1:]),
}

def run(req):
    try:
        with rasterio.open(req.gpkg_path) as src:
            bands = src.read().astype(np.float32)
            meta = src.meta.copy()
            transform = src.transform
        formula = INDEX_FORMULAS.get(req.index_type)
        if not formula:
            return {"success": False, "error": f"Unknown index: {req.index_type}"}
        index_arr = formula(bands)
        out_path = req.gpkg_path.replace(".tif", f"_{req.index_type}_{req.year}.tif")
        meta.update(count=1, dtype="float32")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(index_arr[np.newaxis])
        stats = {"mean": float(index_arr.mean()), "std": float(index_arr.std()),
                 "min": float(index_arr.min()), "max": float(index_arr.max())}
        return {"layer_path": out_path, "statistics": stats, "success": True}
    except Exception as e:
        return {"layer_path": None, "statistics": {}, "success": False, "error": str(e)}
```

### 2D. Novel tools — N1 and N4 require Copernicus credentials

N2 and N3 are pure Python (OSMnx + NetworkX) — implement these first, they have no external deps.

For N1 (TemporalStackLoader), the real implementation needs Copernicus credentials.
While waiting to set those up, implement a **local fallback** that loads a pre-downloaded
stack from disk — this lets E3/smoke test work without live API calls:

```python
# tools/novel/temporal_stack_loader.py  — add fallback mode
def temporal_stack_loader(aoi_bbox, date_range, sensor="sentinel-2",
                           max_cloud_pct=25, output_dir="data/stacks",
                           local_stack_path=None):  # <-- add this param
    # If a pre-downloaded stack is provided, skip the API call entirely
    if local_stack_path and Path(local_stack_path).exists():
        meta = json.load(open(local_stack_path.replace(".tif", "_meta.json")))
        return {
            "gpkg_path": local_stack_path,
            "n_epochs": meta["n_epochs"],
            "dates": meta["dates"],
            "resolution_m": meta.get("resolution_m", 10.0),
            "crs": meta.get("crs", "EPSG:32614"),
            "success": True,
        }
    # Otherwise: run full Copernicus download (existing implementation)
    ...
```

**Download one Harvey stack manually for smoke tests:**
```bash
# Go to: https://dataspace.copernicus.eu/browser/
# Search: Hurricane Harvey, Houston, August-September 2017, Sentinel-2
# Download 3-4 scenes manually as GeoTIFFs
# Save to: data/stacks/harvey_manual/
# Then smoke test uses local_stack_path="data/stacks/harvey_manual/"
```

### 2E. Test each real tool individually before wiring to agents

```bash
# Test each tool one at a time — catch failures early
python - << 'PYEOF'
import sys; sys.path.insert(0, '.')
import json

print("=== Testing ObjectDetection ===")
from tools.vision.object_detection import run
from tools.schemas import ObjectDetectionInput
result = run(ObjectDetectionInput(image_path="data/test/sample.tif", confidence_threshold=0.65))
print(f"  success={result['success']}, n_detections={len(result.get('labels',[]))}")
assert result['success'], f"FAIL: {result.get('error')}"
print("  PASS")

print("=== Testing GetAreaBoundary ===")
from tools.gis.area_boundary import run as run_boundary
from tools.schemas import GetAreaBoundaryInput
result = run_boundary(GetAreaBoundaryInput(place_name_or_bbox="Houston, Texas"))
print(f"  success={result['success']}, gpkg={result.get('gpkg_path')}")
assert result['success'], f"FAIL: {result.get('error')}"
print("  PASS")

print("=== Testing RoadDamageScorer ===")
from tools.novel.road_damage_scorer import road_damage_scorer
result = road_damage_scorer(
    gpkg_path="data/test/sample_roads.gpkg",
    damage_raster_layer="flood_extent"
)
print(f"  success={result['success']}, n_scored={result.get('n_roads_scored')}")
print("  PASS" if result['success'] else "  WARN: no road data")

print("=== Testing EvacuationRoutePlanner ===")
from tools.novel.evacuation_route_planner import evacuation_route_planner
result = evacuation_route_planner(
    gpkg_path="data/test/sample_roads.gpkg",
    origins=[[29.70, -95.50]],
    destinations=[[29.75, -95.45]],
    mode="evacuation"
)
print(f"  success={result['success']}, n_routes={len(result.get('routes',[]))}")
if result['success'] and result['routes']:
    print(f"  RSS={result['routes'][0]['rss']}, length={result['routes'][0]['length_km']}km")
print("  PASS" if result['success'] else "  FAIL: check N3 implementation")

print("\nAll tool tests complete.")
PYEOF
```

---

## Priority 3 — Real model inference (after tools are real)

Once tools return real outputs, connect agents to real vLLM servers.
The base_agent.py already calls vLLM — just need servers running properly.

```bash
# Verify all 4 servers respond before running any benchmark
python - << 'PYEOF'
import requests

servers = {
    "ORC  (Qwen3-30B-A3B)": "http://localhost:8000/v1/models",
    "VRA  (Qwen2.5-VL-7B)": "http://localhost:8001/v1/models",
    "GA   (Qwen3-32B fp8)":  "http://localhost:8002/v1/models",
    "Tool server":            "http://localhost:9000/health",
}
all_ok = True
for name, url in servers.items():
    try:
        r = requests.get(url, timeout=5)
        print(f"  {name}: OK ({r.status_code})")
    except Exception as e:
        print(f"  {name}: FAIL — {e}")
        all_ok = False

print("\nAll servers ready:", all_ok)
PYEOF

# If any server is down, restart it:
bash scripts/start_all_vllm.sh
```

**Validate one real multi-turn ReAct call:**
```bash
python - << 'PYEOF'
import sys; sys.path.insert(0, '.')
from agents.vra import VRA
import yaml

cfg = yaml.safe_load(open('configs/models.yaml'))
vra = VRA(model_port=cfg['vra']['vllm_port'])

# Single real call — this hits the real Qwen2.5-VL-7B model
result = vra.run(
    task_message={
        "task_desc": "Describe what you see in this satellite image and count any buildings visible.",
        "tool_schema_subset": ["ImageDescription", "CountGivenObject"]
    },
    em_context={}
)

print("Agent:", result['agent'])
print("Tool calls made:", [tc['tool'] for tc in result['tool_calls']])
print("Thinking traces present:", sum(1 for tc in result['tool_calls'] if tc.get('thinking_trace')))
print("All succeeded:", all(tc['output'].get('success', True) for tc in result['tool_calls']))
print("Final output:", str(result['output'])[:300])
PYEOF
```

---

## Priority 4 — Stage 1 training (run now, xBD is public)

This is real, consequential fine-tuning you can do today. It gives VRA its 4-class
building damage capability — currently it can only describe damage in natural language.

```bash
# Step 1: Register and download xBD at xview2.org (free, takes ~24h to approve)
# Save to: data/xbd/train/ and data/xbd/test/

# Step 2: Pre-compute Prithvi embeddings for all xBD polygons
# Run this overnight — ~4-6 hours for full xBD
python - << 'PYEOF'
import sys; sys.path.insert(0, '.')
from tools.novel.prithvi_embed import prithvi_embed
from pathlib import Path
import json

xbd_img_dir = Path("data/xbd/train/images")
out_dir = Path("data/xbd_embeddings/train")
out_dir.mkdir(parents=True, exist_ok=True)

images = list(xbd_img_dir.glob("*post*.tif"))
print(f"Processing {len(images)} post-event images...")

for i, img_path in enumerate(images):
    out_file = out_dir / f"{img_path.stem}_embed.npy"
    if out_file.exists():
        continue
    result = prithvi_embed(
        geotiff_path=str(img_path),
        date_list=["post"],
        output_type="embedding",
        output_dir=str(out_dir)
    )
    if not result['success']:
        print(f"  FAIL: {img_path.name} — {result.get('error')}")
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(images)}")

print("Embedding pre-computation complete.")
PYEOF

# Step 3: Train the damage MLP
python training/stage1_damage_mlp.py \
  --embeddings_dir data/xbd_embeddings \
  --output models/prithvi/damage_mlp.pt \
  --epochs 50 \
  --batch_size 512 \
  --lr 1e-3

# Step 4: Evaluate
python - << 'PYEOF'
import torch, sys; sys.path.insert(0, '.')
from models.prithvi.damage_mlp import DamageMLPHead
from evaluation.metrics import compute_weighted_f1
# Load trained model and run on test embeddings
# Target: weighted F1 >= 0.71
PYEOF
```

---

## Priority 5 — Real E2 and E5 benchmarks

Once tools and models are real, rerun with actual inference.

```bash
# E2: OpenEarthAgent
# Your proxy run used mock tool outputs — this uses real ones
python evaluation/benchmarks/openearth_eval.py \
  --data data/openearth_agent/eval.jsonl \
  --output results/e2_oea_real.json

# What to look for in results:
python - << 'PYEOF'
import json
r = json.load(open('results/e2_oea_real.json'))
m = r['metrics']
print("=== E2 Real Results ===")
print(f"Inst:  {m.get('Inst','?')}   (OEA 4B baseline: 99.51)")
print(f"Tool:  {m.get('Tool','?')}   (OEA 4B baseline: 97.18)")
print(f"ArgV:  {m.get('ArgV','?')}   (OEA 4B baseline: 62.10) <- KEY METRIC")
print(f"Summ:  {m.get('Summ','?')}   (OEA 4B baseline: 83.64)")
print()
argv = float(m.get('ArgV', 0))
if argv > 62.1:
    print("SUCCESS: Zero-shot 7B beats trained 4B on ArgV. Strong story for paper.")
elif argv > 58:
    print("EXPECTED: Slightly below baseline. LoRA (Stage 2) will close this gap.")
else:
    print("ACTION NEEDED: ArgV < 58. Check VRA system prompt for explicit arg format examples.")
PYEOF

# E5: ThinkGeo — your most important current result
python evaluation/benchmarks/thinkgeo_eval.py \
  --data data/thinkgeo/eval.jsonl \
  --output results/e5_thinkgeo_real.json

python - << 'PYEOF'
import json
r = json.load(open('results/e5_thinkgeo_real.json'))
m = r['metrics']
print("=== E5 Real Results ===")
print(f"Simple GIS TSR:        {m.get('simple_gis_tsr','?')}%  (ThinkGeo GPT-4o: ~78%)")
print(f"Constrained routing:   {m.get('routing_tsr','?')}%    (ThinkGeo GPT-4o: ~42%) <- KEY DELTA")
print(f"HRR:                   {m.get('hrr','?')}%   (ThinkGeo GPT-4o: ~71%)")
print(f"Overall TSR:           {m.get('overall_tsr','?')}%  (ThinkGeo GPT-4o: ~63%)")
print()
routing = float(m.get('routing_tsr', 0))
if routing > 60:
    print("N3 IS WORKING: +18pts or more over GPT-4o baseline. This is a paper result.")
elif routing > 42:
    print("N3 helps but weakly. Check if EvacuationRoutePlanner is actually being called.")
else:
    print("ACTION: N3 may not be getting called. Check ORC routing for 'route' queries.")
PYEOF
```

**If E5 routing TSR is not better than 42%:** ORC is not routing to PA. Fix the ORC prompt:

```python
# In agents/orc.py — add this to the system prompt explicitly:
"""
ROUTING RULE — MANDATORY:
Any query containing words: route, path, evacuation, safest way, navigate, travel, reach shelter
MUST be assigned to PA with tools: [EvacuationRoutePlanner, Calculator, Plot]
Never use ComputeDistance as a substitute for EvacuationRoutePlanner.
ComputeDistance returns straight-line distance only. EvacuationRoutePlanner returns graph-optimal routes.
"""
```

---

## Priority 6 — Ablation A8 on ThinkGeo (run now, no datasets needed)

This is one of your most important ablations and fully runnable today.
A8 removes N1/N2/N3 and shows how much they contribute.

```python
# evaluation/ablations/ablation_configs.py — ensure A8 is defined as:
ABLATION_A8 = {
    "name": "A8_no_novel_tools",
    "description": "Remove N1/N2/N3/N4. Use only original 24 tools.",
    "disabled_tools": ["TemporalStackLoader", "RoadDamageScorer",
                       "EvacuationRoutePlanner", "PrithviEmbed"],
    "fallback_tools": {
        "EvacuationRoutePlanner": "ComputeDistance",   # Degrades to straight-line
        "RoadDamageScorer": None,                      # No road scoring available
        "TemporalStackLoader": "ChangeDetection",      # Only 2-image change detection
    }
}
```

```bash
# Run A8 on ThinkGeo routing tasks
python evaluation/ablations/run_all_ablations.py \
  --ablation A8 \
  --benchmark thinkgeo \
  --output results/ablation_a8_thinkgeo.json

# Compare full system vs A8
python - << 'PYEOF'
import json
full = json.load(open('results/e5_thinkgeo_real.json'))['metrics']
a8   = json.load(open('results/ablation_a8_thinkgeo.json'))['metrics']
delta = float(full.get('routing_tsr',0)) - float(a8.get('routing_tsr',0))
print(f"Full system routing TSR: {full.get('routing_tsr')}%")
print(f"A8 (no N1/N2/N3) routing TSR: {a8.get('routing_tsr')}%")
print(f"Delta: +{delta:.1f} pts — this is what N3 contributes")
print()
print("This is Table 5, Row A8 in your paper.")
PYEOF
```

---

## Priority 7 — Smoke test on real Harvey data

This validates the full 4-agent pipeline on a real multi-epoch case.

```bash
# Requires: Copernicus credentials OR manually downloaded Harvey GeoTIFFs
# Manual Harvey data: https://dataspace.copernicus.eu/browser/
# Search: T15RUQ (Houston UTM tile), August-September 2017, Sentinel-2, <20% cloud

bash scripts/smoke_test.sh

# Review the output
python - << 'PYEOF'
import json
ep = json.load(open('results/smoke_test_harvey.json'))

print("=== Harvey Smoke Test ===")
print(f"Tool calls: {ep['metrics']['n_tool_calls']}")
print(f"MPC messages: {ep['metrics']['n_mpc_messages']}")
print(f"RSS: {ep['metrics']['rss']}")
print(f"Wall time: {ep['metrics']['wall_time_ms']/1000:.1f}s")
print()
print("Tool call sequence:")
for tc in ep['tool_calls']:
    ok = "✓" if tc['output'].get('success', True) else "✗"
    print(f"  {ok} [{tc['agent']}] {tc['tool']}")

# Check RSS is computed from real segment scores, not hardcoded
routes = ep.get('routes', [])
for r in routes[:2]:
    segs = r.get('segments', [])
    if segs:
        print(f"\nRoute '{r.get('id')}': {len(segs)} segments, RSS={r.get('rss')}")
        print(f"  Segment scores: {[s['damage_score'] for s in segs[:5]]}")
    else:
        print("WARNING: Route has no segment-level scores — RSS may be trivially 1.0")
PYEOF
```

---

## Priority 8 — Three validation checks from Phase 2 doc

These are the three things that can silently be wrong even when tests pass:

### Check 1: N3 is being called, not ComputeDistance
```bash
python - << 'PYEOF'
import json
results = json.load(open('results/e5_thinkgeo_real.json'))
routing_eps = [ep for ep in results['per_episode'] if 'route' in ep['query'].lower()]
for ep in routing_eps[:5]:
    tools_called = [tc['tool'] for tc in ep.get('tool_calls', [])]
    used_n3 = 'EvacuationRoutePlanner' in tools_called
    used_fallback = 'ComputeDistance' in tools_called and not used_n3
    print(f"Q: {ep['query'][:60]}")
    print(f"   Tools: {tools_called}")
    print(f"   N3 called: {used_n3} | Fallback to distance: {used_fallback}")
    print()
PYEOF
```

### Check 2: RSS is from real segment scores, not hardcoded 1.0
```bash
python - << 'PYEOF'
import sys; sys.path.insert(0, '.')
from tools.novel.evacuation_route_planner import evacuation_route_planner
import geopandas as gpd

# Create a test road network with some damaged segments
import osmnx as ox
G = ox.graph_from_bbox(29.72, 29.68, -95.45, -95.53, network_type='drive')
roads_gdf = ox.graph_to_gdfs(G, nodes=False)
roads_gdf['damage_score'] = 0.0
# Manually mark 20% of roads as impassable
import numpy as np
n_impassable = int(len(roads_gdf) * 0.20)
roads_gdf.loc[roads_gdf.index[:n_impassable], 'damage_score'] = 0.80
roads_gdf.loc[roads_gdf.index[:n_impassable], 'traversability'] = 'impassable'
roads_gdf.to_file('/tmp/test_roads.gpkg', layer='road_damage_scores', driver='GPKG')

result = evacuation_route_planner(
    gpkg_path='/tmp/test_roads.gpkg',
    origins=[[29.70, -95.50]],
    destinations=[[29.72, -95.47]],
    mode='evacuation'
)
print("Routes returned:", len(result.get('routes', [])))
if result['routes']:
    r = result['routes'][0]
    print(f"RSS: {r['rss']} — should be < 1.0 since some roads are impassable")
    if r['rss'] >= 1.0:
        print("WARNING: RSS=1.0 even with impassable roads — N3 may not read segment scores correctly")
    else:
        print("PASS: RSS correctly reflects road damage")
PYEOF
```

### Check 3: Hermes tool call parsing handles multi-turn
```bash
python - << 'PYEOF'
import sys; sys.path.insert(0, '.')
from agents.vra import VRA
import yaml

cfg = yaml.safe_load(open('configs/models.yaml'))
vra = VRA(model_port=cfg['vra']['vllm_port'])

# Task that requires multiple sequential tool calls
result = vra.run(
    task_message={
        "task_desc": "Count flooded buildings: first get the image bounds, then detect buildings, then count those labelled as flooded.",
        "tool_schema_subset": ["GetBboxFromGeotiff", "ObjectDetection", "CountGivenObject"]
    },
    em_context={}
)

tools_used = [tc['tool'] for tc in result['tool_calls']]
print("Tools called in sequence:", tools_used)
print("n_turns:", result['n_turns'])

if len(tools_used) >= 2:
    print("PASS: Multi-turn ReAct working correctly")
else:
    print("FAIL: Agent stopped after 1 tool call — check Hermes parsing in base_agent.py")
    print("Likely cause: model.tool_calls not being re-appended to messages correctly")
PYEOF
```

---

## Summary — exact sequence to follow

```
TODAY:
  [ ] Swap ORC to Qwen3-30B-A3B, verify GPU fits
  [ ] Verify all 4 vLLM servers respond
  [ ] Run Check 1, 2, 3 above to find silent bugs

THIS WEEK:
  [ ] Implement real vision tools (GroundingDINO + SAM2 — ~1 day)
  [ ] Implement real GIS tools (OSMnx + rasterio — ~1 day)
  [ ] Implement real spectral tools (~half day, mostly done in prithvi_embed)
  [ ] Implement N2 and N3 with real NetworkX + OSMnx (~1 day)
  [ ] Download Harvey GeoTIFFs manually for N1 local fallback
  [ ] Test each tool individually with the test script in Priority 2E
  [ ] Re-run E2 and E5 with real inference, compare to baseline tables
  [ ] Run A8 ablation on ThinkGeo
  [ ] Run smoke test on Harvey

NEXT WEEK:
  [ ] Start xBD download and Prithvi embedding computation (overnight job)
  [ ] Train Stage 1 damage MLP once embeddings ready
  [ ] Generate Type A + B alignment negatives from OEA episodes
  [ ] Write paper: Intro, Related Work, Task, Framework, Dataset pipeline sections
  [ ] Write paper: Table 1 (E2) and Table 5 (E5) with real numbers

WHEN DATASETS ARRIVE (Day ~16):
  [ ] Fire all 3 Stage 2 LoRA training runs simultaneously
  [ ] Run E3 (TDRD), E4 (conflict), remaining ablations A1-A7
  [ ] Generate Type C alignment negatives (unsafe routes from corrupted damage scores)
  [ ] Write paper: Results section with all numbers
  [ ] Final submission
```

---

## One-line test to confirm you're ready to run real experiments

```bash
python - << 'PYEOF'
import subprocess, sys
checks = {
    "ORC server":   "curl -sf http://localhost:8000/v1/models",
    "VRA server":   "curl -sf http://localhost:8001/v1/models",
    "Tool server":  "curl -sf http://localhost:9000/health",
    "OEA data":     "test -f data/openearth_agent/eval.jsonl",
    "ThinkGeo data":"test -f data/thinkgeo/eval.jsonl",
}
all_ok = True
for name, cmd in checks.items():
    r = subprocess.run(cmd, shell=True, capture_output=True)
    ok = r.returncode == 0
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok: all_ok = False
print()
print("Ready to run real experiments:", all_ok)
if not all_ok:
    print("Fix failing checks above before running any benchmark.")
PYEOF
```