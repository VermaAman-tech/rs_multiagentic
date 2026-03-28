# MAGRF — All Datasets to Download Now
## Every dataset used directly by the pipeline for training or evaluation

These are the datasets YOUR pipeline trains on and evaluates against.
Datasets used only by Person 1 (TDRD generation) or Person 2 (Toolchain generation)
are excluded — those are their responsibility.

---

## Quick summary

| # | Dataset | Size | Used for | Priority |
|---|---|---|---|---|
| 1 | xBD (xView2) | ~14.5 GB | Stage 1: Prithvi damage MLP training | CRITICAL |
| 2 | FloodNet | ~3.6 GB | Stage 1: flood segmentation head training | CRITICAL |
| 3 | Sen1Floods11 | ~13 GB | Stage 1: change detection + phase LSTM training | CRITICAL |
| 4 | LEVIR-CD | ~1.5 GB | Stage 1: SAM2-CD change detection training | HIGH |
| 5 | OpenEarthAgent eval | ~6 GB | E2 benchmark evaluation | CRITICAL |
| 6 | ThinkGeo / GeoBenchX | ~500 MB | E5 benchmark evaluation | CRITICAL |
| 7 | RescueADI | ~8 GB | E1 benchmark evaluation | HIGH |
| 8 | DOTA v2.0 | ~11 GB | Object detection evaluation | MEDIUM |

**Total disk needed: ~58 GB**

Start all downloads simultaneously — they are independent.

---

## Dataset 1 — xBD (xView2)

**What it is:** The largest publicly available building damage dataset.
~19,986 pre/post satellite image pairs across 18 disaster types. ~800,000 building
polygons each labelled: no-damage (0), minor (1), major (2), destroyed (3).

**Why your pipeline needs it:**
The Prithvi damage MLP (Stage 1 training) is a 2-layer classifier trained on
Prithvi-300M patch embeddings extracted from xBD post-disaster images. Without
this training, VRA can only describe damage in natural language — it cannot output
a 4-class damage score per building polygon. This is the single most important
training dataset for your pipeline.

**Size:** ~14.5 GB (train ~10 GB + test ~4.5 GB)

**Download:**
```
1. Go to: https://xview2.org/challenge
2. Click "Register" — free account, approval takes 24-48 hours
3. After approval: download "train" and "test" splits
4. Save structure:
   data/xbd/
   ├── train/
   │   ├── images/          # GeoTIFF files: {id}_pre_disaster.tif, {id}_post_disaster.tif
   │   └── labels/          # JSON files: {id}_pre_disaster.json, {id}_post_disaster.json
   └── test/
       ├── images/
       └── labels/
```

**Verify download:**
```bash
python - << 'PYEOF'
from pathlib import Path
xbd = Path("data/xbd")
train_imgs = list((xbd/"train"/"images").glob("*post*.tif"))
test_imgs  = list((xbd/"test" /"images").glob("*post*.tif"))
print(f"Train post-event images: {len(train_imgs)}")  # Expect ~14,000
print(f"Test post-event images:  {len(test_imgs)}")   # Expect ~5,000
print(f"Total: {len(train_imgs)+len(test_imgs)}")     # Expect ~19,986
PYEOF
```

**Label format (what you need to parse):**
```json
{
  "features": {
    "xy": [
      {
        "wkt": "POLYGON ((lon1 lat1, lon2 lat2, ...))",
        "properties": {
          "subtype": "major-damage"
        }
      }
    ]
  }
}
```
Damage classes: `"no-damage"→0`, `"minor-damage"→1`, `"major-damage"→2`, `"destroyed"→3`

**Start xBD registration now** — this takes 24-48 hours to approve. Do this first.

---

## Dataset 2 — FloodNet

**What it is:** 2,343 high-resolution UAV aerial images from Hurricane Harvey.
10 semantic classes including flooded building, non-flooded building, flooded road,
non-flooded road, water, tree, vehicle, pool, grass.

**Why your pipeline needs it:**
VRA's flood segmentation capability is trained on FloodNet. Without it, VRA uses
SAM2 zero-shot for flood detection (~0.55 IoU). With FloodNet-trained head via
Prithvi: ~0.78 IoU. This directly improves VRA's flood extent outputs (Output A
and B of the 6-output task).

**Size:** ~3.6 GB

**Download:**
```bash
# Option A: HuggingFace (easiest)
pip install huggingface_hub
python - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ker0sene/FloodNet-Dataset",
    repo_type="dataset",
    local_dir="data/floodnet"
)
print("FloodNet downloaded")
PYEOF

# Option B: GitHub release page
# https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021
# Download the Google Drive links listed in the README

# Option C: Direct from paper authors
# Email: binaworks@gmail.com — they respond quickly
```

**Expected structure:**
```
data/floodnet/
├── train/
│   ├── train-org-img/     # Original images (.jpg)
│   └── train-label-img/   # Semantic masks (.png) — 10 classes
├── val/
│   ├── val-org-img/
│   └── val-label-img/
└── test/
    └── test-org-img/
```

**Class mapping for flood segmentation head:**
```python
FLOODNET_CLASSES = {
    0: "Background",
    1: "Building-flooded",     # Key class for VRA
    2: "Building-non-flooded", # Key class for VRA
    3: "Road-flooded",         # Key class for GA/N2
    4: "Road-non-flooded",
    5: "Water",                # Key class for flood extent
    6: "Tree",
    7: "Vehicle",
    8: "Pool",
    9: "Grass",
}
```

**Verify:**
```bash
python - << 'PYEOF'
from pathlib import Path
imgs = list(Path("data/floodnet/train/train-org-img").glob("*.jpg"))
masks = list(Path("data/floodnet/train/train-label-img").glob("*.png"))
print(f"Train images: {len(imgs)}")   # Expect ~1,445
print(f"Train masks:  {len(masks)}")  # Expect ~1,445
PYEOF
```

---

## Dataset 3 — Sen1Floods11

**What it is:** 4,831 Sentinel-1 SAR and Sentinel-2 optical paired images across
11 global flood events. Pixel-wise flood/no-flood labels. The benchmark dataset
for satellite-based flood detection.

**Why your pipeline needs it:**
Two direct uses: (1) Training the temporal change detection head used by VRA's
PrithviEmbed tool (output_type='change_score'). (2) Training the phase LSTM
(onset/peak/recession/recovery classification). Both of these give VRA its
temporal reasoning capability — without them, VRA has no way to classify
damage escalation phase across epochs.

**Size:** ~13 GB

**Download:**
```bash
git clone https://github.com/cloudtostreet/Sen1Floods11.git data/sen1floods11_repo
cd data/sen1floods11_repo

# Download the actual data (requires their download script)
python download.py --all_data --output_dir ../sen1floods11

# OR download directly from their S3 bucket:
pip install awscli
aws s3 sync s3://sen1floods11 data/sen1floods11 --no-sign-request
```

**Alternative — HuggingFace mirror:**
```bash
python - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="isp-uv-es/Sen1Floods11",
    repo_type="dataset",
    local_dir="data/sen1floods11"
)
PYEOF
```

**Expected structure:**
```
data/sen1floods11/
├── v1.1/
│   ├── data/
│   │   ├── flood_events/
│   │   │   ├── HandLabeled/      # 446 hand-labeled images — use for training
│   │   │   └── WeaklyLabeled/    # 4,385 weakly labeled — use for pre-training
│   │   └── splits/
│   │       ├── flood_handlabeled_train.txt
│   │       ├── flood_handlabeled_val.txt
│   │       └── flood_handlabeled_test.txt
│   └── catalog/
│       └── sen1floods11.json
```

**Each sample contains:**
- Sentinel-1 VV+VH bands (2 bands, 512×512)
- Sentinel-2 13 bands (512×512)
- Binary flood label mask (1=flood, 0=no flood)
- Event metadata (location, date)

**Verify:**
```bash
python - << 'PYEOF'
from pathlib import Path
hand_labeled = list(Path("data/sen1floods11").rglob("*HandLabeled*/*.tif"))
print(f"Hand-labeled files: {len(hand_labeled)}")  # Expect ~1,338 (446 events × 3 files each)
PYEOF
```

---

## Dataset 4 — LEVIR-CD

**What it is:** 637 pairs of high-resolution (0.5m/px) Google Earth images taken
at different times, with per-pixel building change annotations. Standard benchmark
for remote sensing change detection.

**Why your pipeline needs it:**
Used to train the SAM2-CD change detection adapter — the lightweight modules
(ASG + GLCAM) added to SAM2 that improve binary change detection by ~4-6 IoU
points over vanilla SAM2. VRA's ChangeDetection tool uses this trained adapter.
Training takes ~8 hours on 1× A100.

**Size:** ~1.5 GB

**Download:**
```bash
# Option A: Official Google Drive (from paper authors)
# Paper: https://arxiv.org/abs/2002.10867
# Data: https://justchenhao.github.io/LEVIR/
# Direct link: https://drive.google.com/file/d/1NJ2foGjFrFbB2YNqMFDXuGKbgbFSwsMF

pip install gdown
python - << 'PYEOF'
import gdown, zipfile
from pathlib import Path

Path("data/levircd").mkdir(exist_ok=True)
url = "https://drive.google.com/uc?id=1NJ2foGjFrFbB2YNqMFDXuGKbgbFSwsMF"
gdown.download(url, "data/levircd/LEVIR-CD.zip", quiet=False)
with zipfile.ZipFile("data/levircd/LEVIR-CD.zip", "r") as z:
    z.extractall("data/levircd/")
print("LEVIR-CD extracted")
PYEOF

# Option B: HuggingFace
python - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="qingwangcs/LEVIR-CD",
    repo_type="dataset",
    local_dir="data/levircd"
)
PYEOF
```

**Expected structure:**
```
data/levircd/
├── train/
│   ├── A/          # Before images (image pairs)
│   ├── B/          # After images
│   └── label/      # Binary change masks (255=change, 0=no change)
├── val/
│   ├── A/, B/, label/
└── test/
    ├── A/, B/, label/
```

**256×256 patches, 637 image pairs total (split: 445/64/128)**

**Verify:**
```bash
python - << 'PYEOF'
from pathlib import Path
train_a = list(Path("data/levircd/train/A").glob("*.png"))
print(f"Train image pairs: {len(train_a)}")   # Expect 445
PYEOF
```

---

## Dataset 5 — OpenEarthAgent Eval Split

**What it is:** 1,169 verified multi-step geospatial intelligence evaluation
instances across object detection, segmentation, spectral index, change detection,
spatial statistics, and VQA+GIS composite tasks. The published eval split from the
OpenEarthAgent paper (MBZUAI-Oryx).

**Why your pipeline needs it:**
This is Experiment E2 — the direct comparison with the published OEA 4B baseline
(Inst=99.51, Tool=97.18, ArgN=96.08, ArgV=62.10, Summ=83.64). Your VRA evaluated
on this split gives you Table 1 Row 3 of the paper.

**Size:** ~6 GB (includes eval images + annotation JSONs)

**Download:**
```bash
# Clone the full repository (includes download scripts)
git clone https://github.com/mbzuai-oryx/OpenEarthAgent.git data/openearth_repo

cd data/openearth_repo
pip install -e .

# Download eval data specifically
python scripts/download_data.py --split eval --output ../openearth_agent

# Alternative: HuggingFace dataset
python - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mbzuai-oryx/OpenEarthAgent",
    repo_type="dataset",
    local_dir="data/openearth_agent",
    ignore_patterns=["train*"]  # Only download eval split
)
PYEOF
```

**Expected structure:**
```
data/openearth_agent/
├── eval.jsonl           # 1,169 evaluation instances
├── eval_images/         # Referenced satellite image tiles
│   └── *.tif
└── eval_metadata.json   # Task type distribution
```

**Each eval.jsonl line contains:**
```json
{
  "query": "How many buildings are present in this image?",
  "image_path": "eval_images/tile_0042.tif",
  "gt_tool_calls": ["ObjectDetection", "CountGivenObject"],
  "gt_answer": 47,
  "task_type": "counting",
  "gt_tool_args": [{"tool": "CountGivenObject", "args": {"object_name": "building"}}]
}
```

**Verify:**
```bash
python - << 'PYEOF'
import json
from pathlib import Path
lines = [json.loads(l) for l in open("data/openearth_agent/eval.jsonl")]
print(f"Eval instances: {len(lines)}")           # Expect 1,169
task_types = set(l.get('task_type') for l in lines)
print(f"Task types: {task_types}")
PYEOF
```

---

## Dataset 6 — ThinkGeo / GeoBenchX

**What it is:** Human-curated multi-step geospatial reasoning tasks for evaluating
tool-calling LLM agents on GIS workflows. Includes solvable tasks, intentionally
unsolvable tasks (for hallucination rejection testing), and reference implementations.
Released by MBZUAI alongside their GeoAgent framework.

**Why your pipeline needs it:**
This is Experiment E5 — your most impactful zero-shot result. ThinkGeo's best
reported GPT-4o TSR on constrained routing is ~42%. Your N3 EvacuationRoutePlanner
replaces LLM spatial guessing with graph-optimal A*, expected to reach ~65-70%.
This delta is the headline number you can publish today, no training needed.

**Size:** ~500 MB

**Download:**
```bash
git clone https://github.com/MBZUAI-Oryx/GeoAgent.git data/thinkgeo_repo

# Copy eval data to data/thinkgeo/
cp -r data/thinkgeo_repo/benchmark/ data/thinkgeo/

# Also get GeoBenchX if separate
python - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="MBZUAI-Oryx/GeoBenchX",
    repo_type="dataset",
    local_dir="data/geobenchx"
)
PYEOF
```

**Expected structure:**
```
data/thinkgeo/
├── eval.jsonl              # All evaluation tasks
├── solvable_tasks.jsonl    # Tasks with valid answers
├── unsolvable_tasks.jsonl  # Tasks for HRR evaluation
└── reference_solutions/    # Reference tool-call implementations
```

**Task categories in the eval split:**
```
- simple_gis: buffer, intersect, spatial join, measure area  (~40% of tasks)
- constrained_routing: damage-avoiding, obstacle-avoiding paths  (~30%)  ← your main win
- service_area: accessibility zones, nearest facility  (~15%)
- unsolvable: missing data, impossible constraints  (~15%, used for HRR)
```

**Verify:**
```bash
python - << 'PYEOF'
import json
from pathlib import Path
for fname in ["eval.jsonl", "solvable_tasks.jsonl", "unsolvable_tasks.jsonl"]:
    p = Path(f"data/thinkgeo/{fname}")
    if p.exists():
        lines = p.read_text().strip().split('\n')
        print(f"{fname}: {len(lines)} tasks")
PYEOF
```

---

## Dataset 7 — RescueADI

**What it is:** 4,044 high-resolution aerial images with 16,949 semantic masks,
14,483 bounding boxes, and 13,424 natural-language interpretation requests across
9 request types (counting, area estimation, damage recognition, path finding, VQA,
and composite multi-step). Created by Liu et al. (IEEE TGRS).

**Why your pipeline needs it:**
This is Experiment E1 — comparing against the RS-Agent baseline (~63% TSR).
Your VRA with Prithvi backbone targets ~72% TSR (+9% over RS-Agent). Also used
as query templates for understanding what single-agent RS systems expect.

**Size:** ~8 GB (estimated)

**Status:** Not publicly released yet. Paper is published, data is pending.

**How to get it:**
```
Option A — Email the authors (most reliable):
  Lead author: Bo Yuan (BUPT)
  Email search: "Bo Yuan" + "RescueADI" + "BUPT" on Google Scholar
  Subject: "Request for RescueADI Dataset — Research Use"
  They have shared it with several groups already.

Option B — Check the paper GitHub:
  Search GitHub: "RescueADI" or "RS-Agent"
  Paper: "RS-Agent: Automating Remote Sensing Tasks through Intelligent Agent"
  arXiv: https://arxiv.org/abs/2406.07089
  Some versions have the data linked

Option C — Use a proxy for E1 until RescueADI releases:
  DIOR-RSVG is similar (visual grounding in RS, 17,402 instances)
  Download: https://github.com/ZhanYang-nwpu/RSVG-pytorch
  Use this to test E1 evaluation pipeline structure even if numbers differ
```

**Expected structure when available:**
```
data/rescueadi/
├── images/          # High-res aerial images
├── masks/           # Semantic segmentation masks
├── bboxes/          # Bounding box annotations
└── requests.jsonl   # 13,424 NL interpretation requests
```

**For now:** script `evaluation/benchmarks/rescueadi_eval.py` is ready,
waiting for data. E1 can run the day data arrives.

---

## Dataset 8 — DOTA v2.0

**What it is:** The largest aerial object detection dataset. 11,268 images,
1.79 million instances across 18 object categories (plane, ship, storage-tank,
baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor,
bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field,
swimming-pool, container-crane, airport, helipad).

**Why your pipeline needs it:**
Used to evaluate VRA's ObjectDetection tool performance specifically on aerial
imagery — confirming GroundingDINO + SAM2 achieves acceptable detection accuracy
on satellite/aerial images. Also included in OpenEarthAgent eval (OEA uses DOTA
images). This is a supplementary eval, not a training dataset.

**Size:** ~11 GB

**Download:**
```bash
# Register and download from the official DOTA website
# https://captain-whu.github.io/DOTA/dataset.html
# (Free registration required)

# After registration, use their download tool:
python - << 'PYEOF'
# They provide a download script after registration
# Save to: data/dota_v2/
# Structure:
#   data/dota_v2/
#   ├── train/
#   │   ├── images/     # Large aerial images
#   │   └── labelTxt/   # DOTA format annotation txts
#   └── val/
#       ├── images/
#       └── labelTxt/
print("Register at https://captain-whu.github.io/DOTA/dataset.html")
print("Download link provided after registration")
PYEOF

# Alternative: subset via FAIR1M or DIOR (similar, fully open):
python - << 'PYEOF'
from huggingface_hub import snapshot_download
# DIOR is a good alternative (20 categories, fully open)
snapshot_download(
    repo_id="EarthNets/Dataset4EO",  # Check for DOTA or DIOR here
    repo_type="dataset",
    local_dir="data/dota_v2"
)
PYEOF
```

**Why priority is MEDIUM:**
DOTA is part of the OEA eval pipeline. If you already downloaded OEA eval images,
you likely have the relevant DOTA tiles included. Check OEA eval split first —
if all eval images load correctly, you may not need DOTA separately.

---

## Download all at once

Run these simultaneously in separate terminal tabs or tmux panes:

```bash
# Terminal 1: xBD (register first — takes 24-48h)
# Go to xview2.org/challenge and register NOW, then download later

# Terminal 2: FloodNet
pip install huggingface_hub
python - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("ker0sene/FloodNet-Dataset", repo_type="dataset", local_dir="data/floodnet")
PYEOF

# Terminal 3: Sen1Floods11
pip install awscli
aws s3 sync s3://sen1floods11 data/sen1floods11 --no-sign-request

# Terminal 4: LEVIR-CD
pip install gdown
python - << 'PYEOF'
import gdown, zipfile
gdown.download("https://drive.google.com/uc?id=1NJ2foGjFrFbB2YNqMFDXuGKbgbFSwsMF",
               "data/levircd/LEVIR-CD.zip")
zipfile.ZipFile("data/levircd/LEVIR-CD.zip").extractall("data/levircd/")
PYEOF

# Terminal 5: OpenEarthAgent eval
git clone https://github.com/mbzuai-oryx/OpenEarthAgent.git data/openearth_repo
cd data/openearth_repo && python scripts/download_data.py --split eval --output ../openearth_agent

# Terminal 6: ThinkGeo
git clone https://github.com/MBZUAI-Oryx/GeoAgent.git data/thinkgeo_repo
cp -r data/thinkgeo_repo/benchmark/ data/thinkgeo/

# Terminal 7: DOTA v2.0
# Register at captain-whu.github.io/DOTA/dataset.html then download
```

---

## Verify everything after downloading

```bash
python - << 'PYEOF'
from pathlib import Path
import json

checks = {
    "xBD train post images":     (Path("data/xbd/train/images").glob("*post*.tif"), 10000, "Register at xview2.org if missing"),
    "xBD test post images":      (Path("data/xbd/test/images").glob("*post*.tif"),   3000, "Register at xview2.org if missing"),
    "FloodNet train images":     (Path("data/floodnet/train/train-org-img").glob("*.jpg"), 1000, "pip install huggingface_hub"),
    "Sen1Floods11 hand-labeled": (Path("data/sen1floods11").rglob("*HandLabeled*/*.tif"), 400, "aws s3 sync s3://sen1floods11"),
    "LEVIR-CD train A":          (Path("data/levircd/train/A").glob("*.png"), 400, "gdown from Google Drive link"),
    "OEA eval.jsonl":            (None, None, None),
    "ThinkGeo eval tasks":       (None, None, None),
}

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

for name, (glob_iter, min_count, fix) in checks.items():
    if glob_iter is None:
        # Special file check
        if "OEA" in name:
            p = Path("data/openearth_agent/eval.jsonl")
            if p.exists():
                n = sum(1 for _ in p.open())
                ok = n >= 1000
                print(f"  {'OK' if ok else 'FAIL'}  {name}: {n} instances {'(target 1169)' if ok else fix}")
            else:
                print(f"  MISS  {name}: file not found — git clone mbzuai-oryx/OpenEarthAgent")
        elif "ThinkGeo" in name:
            p = Path("data/thinkgeo")
            if p.exists():
                files = list(p.rglob("*.jsonl"))
                print(f"  OK    {name}: {len(files)} jsonl files found")
            else:
                print(f"  MISS  {name}: git clone MBZUAI-Oryx/GeoAgent")
        continue

    files = list(glob_iter)
    n = len(files)
    ok = n >= min_count
    status = "OK  " if ok else ("MISS" if n == 0 else "LOW ")
    print(f"  {status}  {name}: {n} files {'' if ok else f'(expected >={min_count}) — {fix}'}")

print("=" * 60)
print("\nDisk usage:")
import subprocess
for d in ["data/xbd","data/floodnet","data/sen1floods11","data/levircd",
          "data/openearth_agent","data/thinkgeo","data/dota_v2"]:
    if Path(d).exists():
        result = subprocess.run(["du","-sh",d], capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
PYEOF
```

---

## How each dataset maps to training and evaluation

```
STAGE 1 TRAINING (run before any experiments):

  xBD  ──────────────────►  Prithvi damage MLP (4-class building damage)
                              training/stage1_damage_mlp.py
                              Expected: weighted F1 >= 0.71 on xBD test

  FloodNet  ────────────────►  Prithvi flood segmentation head
                              training/stage1_flood_head.py
                              Expected: flood IoU >= 0.78

  Sen1Floods11  ────────────►  Change detection head + phase LSTM
                              training/stage1_change_head.py
                              Expected: change IoU >= 0.72

  LEVIR-CD  ────────────────►  SAM2-CD lightweight change detection adapter
                              training/stage1_sam2cd.py
                              Expected: +4-6 IoU over vanilla SAM2


BENCHMARK EVALUATION (run as experiments):

  OpenEarthAgent eval  ────►  E2 benchmark
                              evaluation/benchmarks/openearth_eval.py
                              Baseline: OEA 4B ArgV=62.10, target >=67%

  ThinkGeo  ───────────────►  E5 benchmark
                              evaluation/benchmarks/thinkgeo_eval.py
                              Baseline: GPT-4o routing TSR=42%, target >=65%

  RescueADI (pending)  ────►  E1 benchmark
                              evaluation/benchmarks/rescueadi_eval.py
                              Baseline: RS-Agent TSR=63%, target ~72%

  DOTA v2.0  ──────────────►  Object detection supplementary eval
                              Used within OEA eval pipeline
                              Confirms GroundingDINO works on aerial imagery
```

---

## What is NOT in this list and why

| Dataset | Reason excluded |
|---|---|
| Raw Sentinel-1/2 scenes | Person 1 downloads these for TDRD generation only |
| UNOSAT / Copernicus EMS records | Person 1 uses these for AOI selection in TDRD |
| WorldPop population grids | Person 1 uses for evacuation GT in TDRD |
| OSM road data | Person 1 uses for road scoring in TDRD (your N2/N3 download OSM live via OSMnx) |
| TDRD itself | Person 1 generates this — arrives ~Day 16 |
| Toolchain Dataset | Person 2 generates this — arrives ~Day 16 |
| Alignment Dataset | You generate this from OEA episodes + TDRD later |