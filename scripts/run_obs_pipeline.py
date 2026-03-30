#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from framework.episode_runner import EpisodeRunner

def _load_dataset(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
        
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line.strip()))
                if limit > 0 and len(rows) >= limit:
                    break
        return rows
        
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    parsed_rows = []
    if isinstance(data, list):
        # OpenEarthAgent style test.json
        for item in data:
            q = ""
            if "conversation" in item:
                for msg in item["conversation"]:
                    if msg.get("from") == "human":
                        val = msg.get("value", "").replace("<AGENT_PROMPT>", "").strip()
                        q = val.split("Question:")[-1].strip() if "Question:" in val else val
                        if q:
                            break
            
            parsed_rows.append({
                "id": str(item.get("idx", len(parsed_rows))),
                "question": q,
                "source": "openearth"
            })
    elif isinstance(data, dict):
        # ThinkGeoBench style ThinkGeoBench.json
        for key, item in data.items():
            q = ""
            if "dialogs" in item:
                for msg in item["dialogs"]:
                    if msg.get("role") == "user":
                        q = msg.get("content", "").strip()
                        break
            parsed_rows.append({
                "id": str(key),
                "question": q,
                "answer": item.get("gt_answer"),
                "source": "thinkgeo"
            })
            
    if limit > 0:
        parsed_rows = parsed_rows[:limit]
    return parsed_rows

def _build_task(row: dict[str, Any], benchmark: str, idx: int) -> dict[str, Any]:
    q = row.get("question") or row.get("prompt") or row.get("query") or ""
    return {
        "objective": str(q),
        "scene_id": f"{benchmark}-scene-{row.get('id', idx)}",
        "region": benchmark,
        "benchmark": benchmark,
        "dataset_row_id": str(row.get("id", idx)),
        "ground_truth": row.get("answer") or row.get("gt_answer"),
        "source": row.get("source"),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Run observability pipeline on OEA + ThinkGeo datasets.")
    parser.add_argument("--oea-path", default="data/openearth_agent/test.json")
    parser.add_argument("--thinkgeo-path", default="data/thinkgeo/ThinkGeoBench.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--orc-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vra-model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--ga-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--pa-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--orc-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--vra-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--ga-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--pa-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--tool-server", default="http://127.0.0.1:9000")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--out-json", default="results/agent_benchmark_runs/full_pipeline_trace.json")
    parser.add_argument("--allow-mock-fallback", action="store_true")
    args = parser.parse_args()

    out_file = Path(args.out_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging full pipeline trace to: {out_file}")

    agent_model_map = {
        "orc": args.orc_model_id,
        "vra": args.vra_model_id,
        "ga": args.ga_model_id,
        "pa": args.pa_model_id,
    }
    agent_base_url_map = {
        "orc": args.orc_base_url,
        "vra": args.vra_base_url,
        "ga": args.ga_base_url,
        "pa": args.pa_base_url,
    }

    runner = EpisodeRunner(
        model_id=args.model_id,
        base_url=args.base_url,
        tool_server=args.tool_server,
        agent_model_map=agent_model_map,
        agent_base_url_map=agent_base_url_map,
        allow_mock_fallback=args.allow_mock_fallback,
        max_turns=args.max_turns,
    )

    oea_rows = _load_dataset(Path(args.oea_path), args.limit)
    tg_rows = _load_dataset(Path(args.thinkgeo_path), args.limit)

    manifest: dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "benchmarks": {"oea": {"n": len(oea_rows)}, "thinkgeo": {"n": len(tg_rows)}},
        "runs": {"oea": [], "thinkgeo": []},
    }

    def _run_rows(rows: list[dict[str, Any]], name: str) -> None:
        for i, row in enumerate(rows):
            task = _build_task(row, name, i)
            rec: dict[str, Any] = {
                "index": i,
                "task_id": task["scene_id"],
                "question": task["objective"],
            }
            try:
                ep = runner.run_single_episode(task)
                rec["ok"] = True
                rec["run_dir"] = ep.get("run_dir")
                rec["metrics"] = ep.get("metrics")
                if "bundle" in ep:
                    rec["agent_outputs"] = ep["bundle"].get("agent_outputs")
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec["ok"] = False
                rec["error"] = str(e)
            
            manifest["runs"][name].append(rec)
            # Checkpoint manifest incrementally
            out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            
            ccq = ep.get("metrics", {}).get("ccq", 0.0) if rec.get("ok") else 0.0
            tsr = "PASS" if rec.get("ok") and ep.get("result", {}).get("safety_valid", True) else "FAIL"
            print(f"[{name}] {i+1}/{len(rows)} ok={rec['ok']} run_dir={rec.get('run_dir')} tsr={tsr} ccq={ccq:.2f}")

    _run_rows(oea_rows, "oea")
    _run_rows(tg_rows, "thinkgeo")

    print("\nPipeline run complete.")
    for name in ("oea", "thinkgeo"):
        rs = manifest["runs"][name]
        ok = sum(1 for r in rs if r.get("ok"))
        print(f"  {name}: {ok}/{len(rs)} succeeded")

if __name__ == "__main__":
    main()
