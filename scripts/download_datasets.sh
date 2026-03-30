#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

MODE="${1:-all}"
export MODE

python - << 'PY'
from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

mode = os.environ.get("MODE", "all")
data = Path("data")
data.mkdir(parents=True, exist_ok=True)

summary: dict[str, dict] = {}


def run(cmd: list[str], cwd: str | None = None, allow_fail: bool = False) -> int:
    print("$", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=cwd)
    if rc != 0 and not allow_fail:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")
    return rc


def in_mode(name: str) -> bool:
    if mode == "all":
        return True
    if mode == "public":
        return name in {"floodnet", "sen1floods11", "openearth", "thinkgeo"}
    if mode == "train":
        return name in {"xbd", "floodnet", "sen1floods11", "levircd"}
    return name == mode


def download_snapshot_candidates(
    candidates: list[str],
    out: Path,
    repo_type: str = "dataset",
    allow_patterns: list[str] | None = None,
) -> tuple[bool, str | None]:
    last_err: str | None = None
    for repo_id in candidates:
        try:
            print(f"Downloading {repo_id} -> {out}")
            snapshot_download(
                repo_id,
                repo_type=repo_type,
                local_dir=str(out),
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
            )
            return True, repo_id
        except Exception as exc:
            last_err = str(exc)
            print(f"Failed {repo_id}: {exc}")
    return False, last_err


def extract_zip_files(root: Path) -> int:
    extracted = 0
    for zip_path in root.rglob("*.zip"):
        try:
            print(f"Extracting {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(root)
            extracted += 1
        except Exception as exc:
            print(f"Failed to extract {zip_path}: {exc}")
    return extracted


if in_mode("xbd"):
    # xBD requires manual registration and acceptance.
    xbd_path = data / "xbd"
    xbd_path.mkdir(parents=True, exist_ok=True)
    summary["xbd"] = {
        "status": "manual_required",
        "path": str(xbd_path),
        "note": "Register at https://xview2.org/challenge and place train/test splits under data/xbd.",
    }

if in_mode("floodnet"):
    out = data / "floodnet"
    ok, src = download_snapshot_candidates(
        [
            "ker0sene/FloodNet-Dataset",
            "torchgeo/floodnet",
            "takara-ai/FloodNet_2021-Track_2_Dataset_HF",
        ],
        out,
        repo_type="dataset",
    )
    if ok:
        summary["floodnet"] = {"status": "downloaded", "path": str(out)}
    else:
        summary["floodnet"] = {
            "status": "failed",
            "path": str(out),
            "note": src,
        }

if in_mode("sen1floods11"):
    out = data / "sen1floods11"
    ok, src = download_snapshot_candidates(
        [
            "isp-uv-es/Sen1Floods11",
            "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11",
            "KozaMateusz/sen1floods11",
        ],
        out,
        repo_type="dataset",
    )
    if ok:
        summary["sen1floods11"] = {"status": "downloaded", "path": str(out), "source": src}
    else:
        summary["sen1floods11"] = {
            "status": "failed",
            "path": str(out),
            "note": src,
        }

if in_mode("levircd"):
    out = data / "levircd"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "LEVIR-CD.zip"
    # Plan source: Google Drive file id 1NJ2foGjFrFbB2YNqMFDXuGKbgbFSwsMF
    rc = run([
        "python",
        "-c",
        (
            "import gdown; "
            "gdown.download('https://drive.google.com/uc?id=1NJ2foGjFrFbB2YNqMFDXuGKbgbFSwsMF',"
            f"'{zip_path}', quiet=False)"
        ),
    ], allow_fail=True)
    if rc == 0 and zip_path.exists():
        run([
            "python",
            "-c",
            (
                "import zipfile; "
                f"zipfile.ZipFile('{zip_path}').extractall('{out}')"
            ),
        ])
        summary["levircd"] = {"status": "downloaded", "path": str(out), "source": "google_drive"}
    else:
        ok, src = download_snapshot_candidates(
            [
                "satellite-image-deep-learning/LEVIR-CD",
                "snchen1230/LEVIRCD",
                "QY1116/LEVIRCD",
            ],
            out,
            repo_type="dataset",
        )
        if ok:
            extracted = extract_zip_files(out)
            summary["levircd"] = {
                "status": "downloaded",
                "path": str(out),
                "source": src,
                "extracted_zips": extracted,
            }
        else:
            summary["levircd"] = {
                "status": "failed",
                "path": str(out),
                "note": src,
            }

if in_mode("openearth"):
    out = data / "openearth_agent"
    ok, src = download_snapshot_candidates(
        ["MBZUAI/OpenEarthAgent"],
        out,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            "test.json",
            "test/**",
            "gpkgs.zip",
            "gpkgs/**",
        ],
    )
    if ok:
        extracted = extract_zip_files(out)
        summary["openearth"] = {
            "status": "downloaded",
            "path": str(out),
            "source": src,
            "extracted_zips": extracted,
        }
    else:
        summary["openearth"] = {"status": "failed", "path": str(out), "note": src}

if in_mode("thinkgeo"):
    out = data / "thinkgeo"
    ok, src = download_snapshot_candidates(
        ["MBZUAI/ThinkGeo", "thinkgeo/ThinkGeo"],
        out,
        repo_type="dataset",
    )
    if ok:
        summary["thinkgeo"] = {"status": "downloaded", "path": str(out), "source": src}
    else:
        summary["thinkgeo"] = {"status": "failed", "path": str(out), "note": src}

summary_path = data / "dataset_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
