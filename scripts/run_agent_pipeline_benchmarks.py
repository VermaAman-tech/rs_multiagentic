#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from framework.episode_runner import EpisodeRunner
from evaluation.metrics import ArgN, ArgV, Inst, Summ, Tool


_TOOL_NAMES = {
    "TemporalStackLoader",
    "RoadDamageScorer",
    "EvacuationRoutePlanner",
    "PrithviEmbed",
    "ObjectDetection",
    "SegmentObjectPixels",
    "ImageDescription",
    "TextToBbox",
    "RegionAttributeDescription",
    "CountGivenObject",
    "OCR",
    "ChangeDetection",
    "DrawBox",
    "AddText",
    "GetAreaBoundary",
    "AddPoisLayer",
    "ComputeDistance",
    "DisplayOnMap",
    "GetBboxFromGeotiff",
    "DisplayGeotiff",
    "DisplayOnGeotiff",
    "AddIndexLayer",
    "ComputeIndexChange",
    "ShowIndexLayer",
    "Calculator",
    "Solver",
    "Plot",
    "GoogleSearch",
    "Terminate",
}


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


def _load_dataset(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if limit <= 0:
        return rows
    if not path.exists():
        return rows

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    parsed_rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        # OpenEarthAgent style test.json
        for item in data:
            human_turns = _extract_oea_human_turns(item)
            gt_actions, gt_answer = _extract_oea_actions_and_answer(item)
            q = ""
            for turn in human_turns:
                if not turn.upper().startswith("OBSERVATION"):
                    q = turn
                    break
            if not q and human_turns:
                q = human_turns[0]
            parsed_rows.append(
                {
                    "id": str(item.get("idx", len(parsed_rows))),
                    "question": q,
                    "answer": gt_answer,
                    "action_sequence": gt_actions,
                    "images": item.get("images", []) if isinstance(item.get("images"), list) else [],
                    "human_turns": human_turns,
                    "source": "openearth",
                }
            )
    elif isinstance(data, dict):
        # ThinkGeoBench style ThinkGeoBench.json
        for key, item in data.items():
            q = ""
            if "dialogs" in item:
                for msg in item["dialogs"]:
                    if msg.get("role") == "user":
                        q = msg.get("content", "").strip()
                        break
            images: list[str] = []
            for f in item.get("files", []) if isinstance(item.get("files"), list) else []:
                if isinstance(f, dict) and f.get("type") == "image":
                    p = f.get("path")
                    if isinstance(p, str) and p.strip():
                        images.append(p.strip())
            parsed_rows.append(
                {
                    "id": str(key),
                    "question": q,
                    "answer": item.get("gt_answer"),
                    "images": images,
                    "source": "thinkgeo",
                }
            )

    if limit > 0:
        parsed_rows = parsed_rows[:limit]
    return parsed_rows


def _build_task(row: dict[str, Any], benchmark: str, idx: int) -> dict[str, Any]:
    q = row.get("question") or row.get("prompt") or row.get("query") or ""
    task = {
        "objective": str(q),
        "scene_id": f"{benchmark}-scene-{idx}",
        "region": benchmark,
        "benchmark": benchmark,
        "dataset_row_id": row.get("id", str(idx)),
        "ground_truth": row.get("answer"),
        "source": row.get("source"),
    }
    if benchmark == "oea" and str(q).strip():
        task["human_inputs"] = [str(q).strip()]
        images = row.get("images", []) if isinstance(row.get("images"), list) else []
        image_paths: list[str] = []
        for img in images:
            p = Path("data/openearth_agent/test") / str(img)
            image_paths.append(str(p))
        if image_paths:
            task["image_paths"] = image_paths
    if benchmark == "thinkgeo" and str(q).strip():
        task["human_inputs"] = [str(q).strip()]
        images = row.get("images", []) if isinstance(row.get("images"), list) else []
        image_paths = []
        for img in images:
            p = Path("data/thinkgeo") / str(img)
            image_paths.append(str(p))
        if image_paths:
            task["image_paths"] = image_paths
    return task


def _clean_oea_human_turn(value: Any) -> str:
    text = str(value or "").replace("<AGENT_PROMPT>", "").strip()
    if "Question:" in text:
        text = text.split("Question:", 1)[1].strip()
    return " ".join(text.split())


def _extract_oea_human_turns(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    conv = item.get("conversation", []) if isinstance(item.get("conversation"), list) else []
    for msg in conv:
        if not isinstance(msg, dict) or msg.get("from") != "human":
            continue
        clean = _clean_oea_human_turn(msg.get("value", ""))
        if clean:
            out.append(clean)
    return out


def _parse_json_obj(text: Any) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_oea_actions_and_answer(item: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    actions: list[dict[str, Any]] = []
    answer = ""
    conv = item.get("conversation", []) if isinstance(item.get("conversation"), list) else []
    for msg in conv:
        if not isinstance(msg, dict) or msg.get("from") != "gpt":
            continue
        parsed = _parse_json_obj(msg.get("value"))
        if not isinstance(parsed, dict):
            continue
        acts = parsed.get("actions") if isinstance(parsed.get("actions"), list) else []
        for act in acts:
            if not isinstance(act, dict):
                continue
            name = str(act.get("name", "")).strip()
            if not name:
                continue
            arguments = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
            actions.append({"name": name, "arguments": arguments})
            if name.lower() == "terminate":
                ans = arguments.get("ans")
                if ans is not None and str(ans).strip():
                    answer = str(ans).strip()
    return actions, answer


def _extract_pred_actions(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows = bundle.get("agents_output", []) if isinstance(bundle, dict) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("from") != "gpt":
            continue
        parsed = _parse_json_obj(row.get("value"))
        if not isinstance(parsed, dict):
            continue
        acts = parsed.get("actions") if isinstance(parsed.get("actions"), list) else []
        for act in acts:
            if not isinstance(act, dict):
                continue
            name = str(act.get("name", "")).strip()
            if not name or name not in _TOOL_NAMES:
                continue
            args = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
            out.append({"name": name, "arguments": args})
    return out


def _extract_final_answer(result: dict[str, Any], bundle: dict[str, Any]) -> str:
    for key in ("ans", "answer", "final_answer", "response"):
        val = result.get(key) if isinstance(result, dict) else None
        if val is not None and str(val).strip():
            return str(val).strip()

    merged = {}
    if isinstance(bundle, dict):
        outputs = bundle.get("agent_outputs")
        if isinstance(outputs, dict):
            merged = outputs.get("orc_merged", {}) if isinstance(outputs.get("orc_merged"), dict) else {}
    for key in ("ans", "answer", "final_answer", "response"):
        val = merged.get(key) if isinstance(merged, dict) else None
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _normalize_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return f"{float(v):.8f}".rstrip("0").rstrip(".")
    return " ".join(str(v).strip().lower().split())


def _scalar_match(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa = float(a)
        fb = float(b)
        tol = max(1e-6, abs(fb) * 0.01)
        return abs(fa - fb) <= tol
    return _normalize_scalar(a) == _normalize_scalar(b)


def _rouge_l_f1(pred: str, ref: str) -> float:
    p = [t for t in re.split(r"\s+", (pred or "").strip()) if t]
    r = [t for t in re.split(r"\s+", (ref or "").strip()) if t]
    if not p or not r:
        return 0.0

    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            if p[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec = lcs / len(p)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _extract_numbers(text: str) -> list[float]:
    vals: list[float] = []
    for m in re.findall(r"[-+]?\d*\.?\d+", text or ""):
        try:
            vals.append(float(m))
        except Exception:
            continue
    return vals


def _extract_key_numbers(text: str) -> list[float]:
    nums = _extract_numbers(text)
    out: list[float] = []
    for v in nums:
        # Ignore year-like integers to avoid false positives from date mentions.
        if 1900.0 <= abs(v) <= 2100.0 and abs(v - round(v)) <= 1e-9:
            continue
        out.append(float(v))
    return out


def _all_key_numbers_match(ref_nums: list[float], pred_nums: list[float]) -> bool:
    if not ref_nums:
        return True
    remaining = [float(v) for v in pred_nums]
    for rv in ref_nums:
        tol = max(0.5, abs(float(rv)) * 0.05)
        hit_idx = -1
        for i, pv in enumerate(remaining):
            if abs(float(pv) - float(rv)) <= tol:
                hit_idx = i
                break
        if hit_idx < 0:
            return False
        remaining.pop(hit_idx)
    return True


def _lcs_pairs(gt_names: list[str], pred_names: list[str]) -> list[tuple[int, int]]:
    m = len(gt_names)
    n = len(pred_names)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_names[i - 1] == pred_names[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    pairs: list[tuple[int, int]] = []
    i = m
    j = n
    while i > 0 and j > 0:
        if gt_names[i - 1] == pred_names[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _answer_match_with_tolerance(pred: str, ref: str) -> bool:
    pred_norm = _normalize_scalar(pred)
    ref_norm = _normalize_scalar(ref)
    if not pred_norm or not ref_norm:
        return False

    ref_key_nums = _extract_key_numbers(ref)
    if ref_key_nums:
        pred_key_nums = _extract_key_numbers(pred)
        if not _all_key_numbers_match(ref_key_nums, pred_key_nums):
            return False

    if pred_norm == ref_norm:
        return True
    return _rouge_l_f1(pred_norm, ref_norm) >= 0.75


def _evaluate_oea_case(
    gt_actions: list[dict[str, Any]],
    pred_actions: list[dict[str, Any]],
    gt_answer: str,
    pred_answer: str,
) -> dict[str, Any]:
    gt = [a for a in gt_actions if isinstance(a, dict) and a.get("name")]
    pred = [a for a in pred_actions if isinstance(a, dict) and a.get("name")]
    answer_match = _answer_match_with_tolerance(pred_answer, gt_answer)

    tool_total = len(gt)
    gt_names = [str(a.get("name", "")).strip().lower() for a in gt]
    pred_names = [str(a.get("name", "")).strip().lower() for a in pred]
    pairs = _lcs_pairs(gt_names, pred_names)
    tool_correct = len(pairs)
    argn_correct = 0
    argn_total = 0
    argv_correct = 0
    argv_total = 0

    for gt_i, pred_i in pairs:
        gt_args = gt[gt_i].get("arguments", {}) if isinstance(gt[gt_i].get("arguments"), dict) else {}
        pd_args = pred[pred_i].get("arguments", {}) if isinstance(pred[pred_i].get("arguments"), dict) else {}
        gt_keys = set(gt_args.keys())
        argn_total += len(gt_keys)
        argn_correct += len(gt_keys & set(pd_args.keys()))

        for k in gt_keys:
            argv_total += 1
            if k in pd_args and _scalar_match(pd_args.get(k), gt_args.get(k)):
                argv_correct += 1

    rouge_l = _rouge_l_f1(pred_answer, gt_answer)

    return {
        "inst_correct": 1 if answer_match else 0,
        "inst_total": 1,
        "tool_correct": tool_correct,
        "tool_total": tool_total,
        "argn_correct": argn_correct,
        "argn_total": argn_total,
        "argv_correct": argv_correct,
        "argv_total": argv_total,
        "rouge_l_f1": rouge_l,
        "answer_match": answer_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict 4-agent pipeline on OEA + ThinkGeo public datasets.")
    parser.add_argument("--oea-path", default="data/openearth_agent/test.json")
    parser.add_argument("--thinkgeo-path", default="data/thinkgeo/ThinkGeoBench.json")
    parser.add_argument("--oea-limit", type=int, default=200)
    parser.add_argument("--thinkgeo-limit", type=int, default=200)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--orc-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vra-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--ga-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--pa-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--orc-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--vra-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--ga-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--pa-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--tool-server", default="http://127.0.0.1:9000")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--strict-no-mock-fallback", action="store_true")
    parser.add_argument("--out-root", default="results/agent_benchmark_runs")
    args = parser.parse_args()

    out_dir = _next_eval_dir(Path(args.out_root))
    print(f"Evaluation output dir: {out_dir}")

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

    uses_hf = any("huggingface.co" in str(u).lower() or "hf.space" in str(u).lower() for u in agent_base_url_map.values())
    if uses_hf:
        hf_token = (
            str(os.getenv("HF_TOKEN", "")).strip()
            or str(os.getenv("HUGGINGFACEHUB_API_TOKEN", "")).strip()
            or str(os.getenv("HUGGING_FACE_HUB_TOKEN", "")).strip()
        )
        if not hf_token:
            raise SystemExit(
                "HF token missing. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) before running remote HF inference."
            )

    runner = EpisodeRunner(
        model_id=args.model_id,
        base_url=args.base_url,
        tool_server=args.tool_server,
        agent_model_map=agent_model_map,
        agent_base_url_map=agent_base_url_map,
        allow_mock_fallback=not args.strict_no_mock_fallback,
        max_turns=args.max_turns,
    )

    oea_rows = _load_dataset(Path(args.oea_path), args.oea_limit)
    tg_rows = _load_dataset(Path(args.thinkgeo_path), args.thinkgeo_limit)

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
                rec["metrics"] = ep.get("metrics", {})
                rec["ccq"] = rec["metrics"].get("ccq")
                bundle = ep.get("bundle", {}) if isinstance(ep.get("bundle"), dict) else {}
                pred_actions = _extract_pred_actions(bundle)
                final_answer = _extract_final_answer(rec.get("result", {}), bundle)
                rec["final_answer"] = final_answer
                rec["pred_actions"] = pred_actions
                if name == "oea":
                    rec["oea_eval"] = _evaluate_oea_case(
                        gt_actions=row.get("action_sequence", []) if isinstance(row.get("action_sequence"), list) else [],
                        pred_actions=pred_actions,
                        gt_answer=str(row.get("answer", "") or ""),
                        pred_answer=final_answer,
                    )
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

        metrics_payload: dict[str, Any] = {
            "TSR": (tsr_count / total * 100) if total > 0 else 0.0,
            "Inst": (struct_ans_count / total * 100) if total > 0 else 0.0,
            "HRR": (hrr_count / total * 100) if total > 0 else 0.0,
            "avg_CCQ": avg_ccq,
            "avg_tool_calls": avg_tools,
        }

        if name == "oea":
            eval_rows = [r.get("oea_eval", {}) for r in rs if r.get("ok") and isinstance(r.get("oea_eval"), dict)]
            inst_correct = sum(int(r.get("inst_correct", 0)) for r in eval_rows)
            inst_total = sum(int(r.get("inst_total", 0)) for r in eval_rows)
            tool_correct = sum(int(r.get("tool_correct", 0)) for r in eval_rows)
            tool_total = sum(int(r.get("tool_total", 0)) for r in eval_rows)
            argn_correct = sum(int(r.get("argn_correct", 0)) for r in eval_rows)
            argn_total = sum(int(r.get("argn_total", 0)) for r in eval_rows)
            argv_correct = sum(int(r.get("argv_correct", 0)) for r in eval_rows)
            argv_total = sum(int(r.get("argv_total", 0)) for r in eval_rows)
            rouge_vals = [float(r.get("rouge_l_f1", 0.0)) for r in eval_rows]
            answer_matches = sum(1 for r in eval_rows if bool(r.get("answer_match")))

            avg_rouge = (sum(rouge_vals) / len(rouge_vals)) if rouge_vals else 0.0
            metrics_payload.update(
                {
                    "Inst": Inst(inst_correct, inst_total),
                    "Tool": Tool(tool_correct, tool_total),
                    "ArgN": ArgN(argn_correct, argn_total),
                    "ArgV": ArgV(argv_correct, argv_total),
                    "Summ": Summ(avg_rouge),
                    "answer_match_pct": (answer_matches / len(eval_rows) * 100.0) if eval_rows else 0.0,
                }
            )

        summary[name] = {
            "total": total,
            "ok": ok,
            "failed": failed,
            "metrics": metrics_payload,
        }
        
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

