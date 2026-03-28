#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from framework.episode_runner import EpisodeRunner


def _next_eval_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums: list[int] = []
    for p in root.glob("eval_*"):
        if p.is_dir():
            try:
                nums.append(int(p.name.split("_")[-1]))
            except ValueError:
                continue
    nxt = (max(nums) + 1) if nums else 1
    out = root / f"eval_{nxt:03d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _build_task(row: dict[str, Any], benchmark: str, idx: int) -> dict[str, Any]:
    q = row.get("question") or row.get("prompt") or row.get("query") or ""
    return {
        "objective": str(q),
        "scene_id": f"{benchmark}-scene-{idx}",
        "region": benchmark,
        "benchmark": benchmark,
        "dataset_row_id": row.get("id", str(idx)),
        "ground_truth": row.get("answer"),
        "source": row.get("source"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict 4-agent pipeline on OEA + ThinkGeo public datasets.")
    parser.add_argument("--oea-path", default="data/openearthagent_eval_public.jsonl")
    parser.add_argument("--thinkgeo-path", default="data/thinkgeo_eval_public.jsonl")
    parser.add_argument("--oea-limit", type=int, default=200)
    parser.add_argument("--thinkgeo-limit", type=int, default=200)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--tool-server", default="http://127.0.0.1:9000")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--strict-no-mock-fallback", action="store_true")
    parser.add_argument("--out-root", default="results/agent_benchmark_runs")
    args = parser.parse_args()

    out_dir = _next_eval_dir(Path(args.out_root))
    print(f"Evaluation output dir: {out_dir}")

    runner = EpisodeRunner(
        model_id=args.model_id,
        base_url=args.base_url,
        tool_server=args.tool_server,
        allow_mock_fallback=not args.strict_no_mock_fallback,
        max_turns=args.max_turns,
    )

    oea_rows = _load_jsonl(Path(args.oea_path), args.oea_limit)
    tg_rows = _load_jsonl(Path(args.thinkgeo_path), args.thinkgeo_limit)

    manifest: dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "out_dir": str(out_dir),
        "benchmarks": {"oea": {"n": len(oea_rows)}, "thinkgeo": {"n": len(tg_rows)}},
        "runs": {"oea": [], "thinkgeo": []},
    }

    def _run_rows(rows: list[dict[str, Any]], name: str) -> None:
        for i, row in enumerate(rows):
            task = _build_task(row, name, i)
            rec: dict[str, Any] = {
                "index": i,
                "question": task["objective"],
                "ground_truth": task.get("ground_truth"),
            }
            try:
                ep = runner.run_single_episode(task)
                rec["ok"] = True
                rec["run_dir"] = ep.get("run_dir")
                rec["result"] = ep.get("result")
                rec["ccq"] = ep.get("ccq")
            except Exception as e:
                rec["ok"] = False
                rec["error"] = str(e)
            manifest["runs"][name].append(rec)
            # incremental checkpoint
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[{name}] {i+1}/{len(rows)} ok={rec['ok']} run_dir={rec.get('run_dir')}")

    _run_rows(oea_rows, "oea")
    _run_rows(tg_rows, "thinkgeo")

    # simple summary
    summary = {}
    for name in ("oea", "thinkgeo"):
        rs = manifest["runs"][name]
        total = len(rs)
        ok = sum(1 for r in rs if r.get("ok"))
        failed = total - ok
        
        # compute TSR and HRR
        tsr_count = sum(1 for r in rs if r.get("ok") and r.get("result", {}).get("safety_valid", False))
        hrr_count = sum(1 for r in rs if r.get("ok") and (r.get("metrics", {}).get("hrr_proxy") == 1))
        
        # compute average metrics if available
        ccq_list = [r.get("metrics", {}).get("ccq", 0.0) for r in rs if r.get("ok")]
        avg_ccq = sum(ccq_list) / len(ccq_list) if ccq_list else 0.0
        
        tools_list = [r.get("metrics", {}).get("total_tool_calls", 0) for r in rs if r.get("ok")]
        avg_tools = sum(tools_list) / len(tools_list) if tools_list else 0.0

        struct_ans_count = sum(1 for r in rs if r.get("ok") and len(r.get("result", {})) > 5)

        summary[name] = {
            "total": total,
            "ok": ok,
            "failed": failed,
            "metrics": {
                "TSR": (tsr_count / total * 100) if total > 0 else 0.0,
                "Inst": (struct_ans_count / total * 100) if total > 0 else 0.0,
                "HRR": (hrr_count / total * 100) if total > 0 else 0.0,
                "avg_CCQ": avg_ccq,
                "avg_tool_calls": avg_tools
            }
        }
        
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

