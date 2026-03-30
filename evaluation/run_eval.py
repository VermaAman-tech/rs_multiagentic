#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.metrics import ArgN, ArgV, HRR, Inst, Summ, TSR, Tool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"raw": data}


def _safe_pct(metric_fn, num: float, den: float) -> float:
    try:
        return round(float(metric_fn(num, den)), 4)
    except Exception:
        return 0.0


def eval_oea(payload: dict[str, Any]) -> dict[str, float]:
    # Prefer already-computed metrics if present.
    if all(k in payload for k in ("Inst", "Tool", "ArgN", "ArgV")):
        return {
            "Inst": float(payload.get("Inst", 0.0)),
            "Tool": float(payload.get("Tool", 0.0)),
            "ArgN": float(payload.get("ArgN", 0.0)),
            "ArgV": float(payload.get("ArgV", 0.0)),
            "Summ": float(payload.get("Summ", 0.0)),
        }

    # Fallback from counters.
    inst = _safe_pct(Inst, payload.get("inst_correct", 0), payload.get("total", 0))
    tool = _safe_pct(Tool, payload.get("tool_correct", 0), payload.get("total", 0))
    argn = _safe_pct(ArgN, payload.get("argn_correct", 0), payload.get("argn_total", 0))
    argv = _safe_pct(ArgV, payload.get("argv_correct", 0), payload.get("argv_total", 0))
    summ = round(float(payload.get("rouge_l_f1", 0.0) * 100.0), 4)
    return {"Inst": inst, "Tool": tool, "ArgN": argn, "ArgV": argv, "Summ": summ}


def eval_thinkgeo(payload: dict[str, Any]) -> dict[str, float]:
    if all(k in payload for k in ("TSR", "HRR")):
        return {
            "TSR": float(payload.get("TSR", 0.0)),
            "HRR": float(payload.get("HRR", 0.0)),
        }

    tsr = _safe_pct(TSR, payload.get("tsr_correct", 0), payload.get("total_solvable", 0))
    hrr = _safe_pct(HRR, payload.get("hrr_correct", 0), payload.get("total_unsolvable", 0))
    return {"TSR": tsr, "HRR": hrr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate benchmark result JSONs into a normalized metric report.")
    parser.add_argument("--results", required=True, help="Path to input result JSON")
    parser.add_argument("--dataset", required=True, choices=["oea", "thinkgeo"], help="Dataset type for metric normalization")
    parser.add_argument("--output", default=None, help="Output file path (defaults to results/eval_<dataset>.json)")
    args = parser.parse_args()

    in_path = Path(args.results)
    if not in_path.exists():
        raise SystemExit(f"Input results file not found: {in_path}")

    payload = _load_json(in_path)

    if args.dataset == "oea":
        metrics = eval_oea(payload)
    else:
        metrics = eval_thinkgeo(payload)

    out = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "input": str(in_path),
        "metrics": metrics,
    }

    out_path = Path(args.output) if args.output else Path(f"results/eval_{args.dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
