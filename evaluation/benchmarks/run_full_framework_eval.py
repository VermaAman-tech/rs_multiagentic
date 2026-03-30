#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.metrics import ArgN, ArgV, HRR, Inst, MGAR, RSS, Summ, TLS, TSR, Tool
from framework.episode_runner import EpisodeRunner


REFUSAL_TERMS = {
    "cannot solve",
    "cannot be determined",
    "i cannot",
    "unable to",
    "not possible",
    "insufficient",
    "no information",
    "i don't know",
    "cannot answer",
    "not enough",
    "sorry",
    "can't",
    "unsure",
    "uncertain",
}

TOOL_ALIASES = {
    "displaygeotiff": "displayongeotiff",
    "displayongeotiff": "displayongeotiff",
    "segmentation": "segmentobjectpixels",
    "segmentobjectpixels": "segmentobjectpixels",
}

ROUTING_HINTS = (
    "route",
    "path",
    "evacuation",
    "safest",
    "navigate",
    "travel",
    "reach shelter",
    "corridor",
)


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


def _safe_load_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_text(text: str) -> str:
    txt = _normalize_space(text).lower()
    txt = re.sub(r"[^a-z0-9\.\-\s]", " ", txt)
    return _normalize_space(txt)


def _normalize_tool_name(name: str) -> str:
    key = _normalize_text(name).replace(" ", "")
    return TOOL_ALIASES.get(key, key)


def _extract_numbers(text: str) -> list[float]:
    nums: list[float] = []
    for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or ""):
        try:
            nums.append(float(tok))
        except ValueError:
            continue
    return nums


def _to_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value):
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _value_match(gt_val: Any, pred_val: Any, rel_tol: float = 0.05) -> bool:
    g_num = _to_number(gt_val)
    p_num = _to_number(pred_val)
    if g_num is not None and p_num is not None:
        tol = max(abs(g_num) * rel_tol, 1e-6)
        return abs(g_num - p_num) <= tol

    if isinstance(gt_val, bool) and isinstance(pred_val, bool):
        return gt_val == pred_val

    if isinstance(gt_val, (dict, list)) or isinstance(pred_val, (dict, list)):
        return _normalize_text(json.dumps(gt_val, sort_keys=True)) == _normalize_text(
            json.dumps(pred_val, sort_keys=True)
        )

    return _normalize_text(str(gt_val)) == _normalize_text(str(pred_val))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _rouge_l_f1(pred: str, gt: str) -> float:
    p_tokens = _normalize_text(pred).split()
    g_tokens = _normalize_text(gt).split()
    if not p_tokens or not g_tokens:
        return 0.0
    lcs = _lcs_len(p_tokens, g_tokens)
    prec = lcs / len(p_tokens)
    rec = lcs / len(g_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _extract_answer_from_obj(obj: Any) -> str:
    if obj is None:
        return ""

    if isinstance(obj, (int, float, bool)):
        return str(obj)

    if isinstance(obj, str):
        txt = obj.strip()
        if not txt:
            return ""
        parsed = _safe_load_json(txt)
        if parsed:
            return _extract_answer_from_obj(parsed)
        return txt

    if isinstance(obj, dict):
        for key in ("final_answer", "ans", "answer", "response"):
            if key in obj and obj[key] not in (None, ""):
                return _extract_answer_from_obj(obj[key])

        actions = obj.get("actions")
        if isinstance(actions, list):
            for act in actions:
                if not isinstance(act, dict):
                    continue
                if _normalize_tool_name(str(act.get("name", ""))) == "terminate":
                    args = act.get("arguments", {})
                    if isinstance(args, dict):
                        return _extract_answer_from_obj(args.get("ans") or args.get("final_answer"))

        if obj.get("raw_output"):
            return _extract_answer_from_obj(obj["raw_output"])

    return ""


def _extract_episode_answer(ep: dict[str, Any]) -> str:
    result = ep.get("result", {}) if isinstance(ep.get("result"), dict) else {}
    bundle = ep.get("bundle", {}) if isinstance(ep.get("bundle"), dict) else {}
    agent_outputs = bundle.get("agent_outputs", {}) if isinstance(bundle.get("agent_outputs"), dict) else {}

    candidates = [
        result,
        result.get("selected_by_conflict"),
        agent_outputs.get("pa"),
        agent_outputs.get("orc_merged"),
        agent_outputs.get("vra"),
    ]
    for cand in candidates:
        ans = _extract_answer_from_obj(cand)
        if ans:
            return _normalize_space(ans)
    return ""


def _flatten_pred_calls(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    calls_by_agent = bundle.get("agent_tool_calls", {}) if isinstance(bundle, dict) else {}
    for agent_name in ("vra", "ga", "pa", "orc"):
        calls = calls_by_agent.get(agent_name, []) if isinstance(calls_by_agent, dict) else []
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            out.append(
                {
                    "agent": agent_name,
                    "name": call.get("tool", ""),
                    "arguments": call.get("args", {}) if isinstance(call.get("args"), dict) else {},
                    "output": call.get("output", {}),
                    "latency_ms": call.get("latency_ms"),
                }
            )
    return out


def _extract_oea_question(item: dict[str, Any]) -> str:
    conv = item.get("conversation", []) if isinstance(item.get("conversation"), list) else []
    for msg in conv:
        if not isinstance(msg, dict) or msg.get("from") != "human":
            continue
        val = _normalize_space(str(msg.get("value", ""))).replace("<AGENT_PROMPT>", "").strip()
        if "Question:" in val:
            q = _normalize_space(val.split("Question:", 1)[1])
            if q:
                return q
        if val and not val.startswith("OBSERVATION"):
            return val
    return ""


def _extract_gt_actions_from_oea(item: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    conv = item.get("conversation", []) if isinstance(item.get("conversation"), list) else []
    gt_calls: list[dict[str, Any]] = []
    gt_answer = ""

    for msg in conv:
        if not isinstance(msg, dict) or msg.get("from") != "gpt":
            continue
        parsed = _safe_load_json(str(msg.get("value", "")))
        actions = parsed.get("actions", []) if isinstance(parsed.get("actions"), list) else []
        for act in actions:
            if not isinstance(act, dict):
                continue
            name = str(act.get("name", "")).strip()
            args = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
            if not name:
                continue
            if _normalize_tool_name(name) == "terminate":
                if not gt_answer:
                    gt_answer = _normalize_space(str(args.get("ans") or args.get("final_answer") or ""))
            else:
                gt_calls.append({"name": name, "arguments": args})

    return gt_calls, gt_answer


def load_oea_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        gt_calls, gt_answer = _extract_gt_actions_from_oea(item)
        row = {
            "id": str(item.get("idx", i)),
            "question": _extract_oea_question(item),
            "gt_answer": gt_answer,
            "gt_tool_calls": gt_calls,
            "images": item.get("images", []),
            "dataset": "oea",
            "solvable": True,
            "is_routing": False,
        }
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _thinkgeo_gt_match(generated_text: str, gt_answer: Any) -> bool:
    if not gt_answer:
        return False

    gen = _normalize_text(generated_text)

    if isinstance(gt_answer, list):
        terms = [_normalize_text(str(x)) for x in gt_answer if _normalize_text(str(x))]
        return any(term in gen for term in terms)

    if isinstance(gt_answer, dict):
        blacklist = gt_answer.get("blacklist")
        if isinstance(blacklist, list):
            for term in blacklist:
                t = _normalize_text(str(term))
                if t and t in gen:
                    return False

        whitelist = gt_answer.get("whitelist")
        if isinstance(whitelist, list):
            for group in whitelist:
                if not isinstance(group, list):
                    group = [group]
                opts = [_normalize_text(str(item)) for item in group if _normalize_text(str(item))]
                if opts and not any(opt in gen for opt in opts):
                    return False
            return True

    return False


def _is_routing_task(question: str, tool_names: list[str], gt_calls: list[dict[str, Any]]) -> bool:
    q = _normalize_text(question)
    if any(hint in q for hint in ROUTING_HINTS):
        return True

    merged = [*tool_names, *(call.get("name", "") for call in gt_calls)]
    for name in merged:
        n = _normalize_tool_name(str(name))
        if "route" in n or "evacuation" in n or "path" in n:
            return True
    return False


def load_thinkgeo_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []

    rows: list[dict[str, Any]] = []
    for key, item in data.items():
        if not isinstance(item, dict):
            continue

        dialogs = item.get("dialogs", []) if isinstance(item.get("dialogs"), list) else []
        question = ""
        gt_calls: list[dict[str, Any]] = []

        for turn in dialogs:
            if not isinstance(turn, dict):
                continue
            if turn.get("role") == "user" and not question:
                question = _normalize_space(str(turn.get("content", "")))

            if turn.get("role") == "assistant":
                tcs = turn.get("tool_calls", []) if isinstance(turn.get("tool_calls"), list) else []
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                    name = str(fn.get("name", "")).strip()
                    args = fn.get("arguments", {}) if isinstance(fn.get("arguments"), dict) else {}
                    if name:
                        gt_calls.append({"name": name, "arguments": args})

        tools = item.get("tools", []) if isinstance(item.get("tools"), list) else []
        tool_names = [str(t.get("name", "")) for t in tools if isinstance(t, dict)]
        gt_answer = item.get("gt_answer")
        solvable = gt_answer is not None

        rows.append(
            {
                "id": str(key),
                "question": question,
                "gt_answer": gt_answer,
                "gt_tool_calls": gt_calls,
                "tool_names": tool_names,
                "dataset": "thinkgeo",
                "solvable": solvable,
                "is_routing": _is_routing_task(question, tool_names, gt_calls),
            }
        )
        if limit > 0 and len(rows) >= limit:
            break

    return rows


def _score_tool_and_args(
    gt_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    tool_correct = 0
    tool_total = len(gt_calls)
    argn_correct = 0
    argn_total = 0
    argv_correct = 0
    argv_total = 0

    for idx, gt_call in enumerate(gt_calls):
        gt_name = _normalize_tool_name(str(gt_call.get("name", "")))
        gt_args = gt_call.get("arguments", {}) if isinstance(gt_call.get("arguments"), dict) else {}

        pred_call = pred_calls[idx] if idx < len(pred_calls) else None
        pred_name = _normalize_tool_name(str(pred_call.get("name", ""))) if pred_call else ""
        pred_args = pred_call.get("arguments", {}) if pred_call and isinstance(pred_call.get("arguments"), dict) else {}

        if gt_name == pred_name:
            tool_correct += 1

        for arg_name, gt_val in gt_args.items():
            argn_total += 1
            argv_total += 1
            if arg_name in pred_args:
                argn_correct += 1
                if _value_match(gt_val, pred_args[arg_name]):
                    argv_correct += 1

    return {
        "tool_correct": tool_correct,
        "tool_total": tool_total,
        "argn_correct": argn_correct,
        "argn_total": argn_total,
        "argv_correct": argv_correct,
        "argv_total": argv_total,
    }


def _oea_answer_correct(pred_answer: str, gt_answer: str) -> bool:
    p = _normalize_space(pred_answer)
    g = _normalize_space(gt_answer)
    if not p or not g:
        return False

    g_nums = _extract_numbers(g)
    p_nums = _extract_numbers(p)
    if g_nums:
        if not p_nums:
            return False
        for g_num in g_nums:
            tol = max(abs(g_num) * 0.05, 1e-6)
            if not any(abs(p_num - g_num) <= tol for p_num in p_nums):
                return False

    rouge = _rouge_l_f1(p, g)
    norm_g = _normalize_text(g)
    norm_p = _normalize_text(p)
    return norm_g in norm_p or rouge >= 0.45


def _sample_route_stats(pred_calls: list[dict[str, Any]]) -> dict[str, Any]:
    safe_segments = 0
    total_segments = 0
    rss_values: list[float] = []
    conflict_checks = 0
    replans_safe = 0
    pending_conflict = False

    for call in pred_calls:
        if _normalize_tool_name(str(call.get("name", ""))) != "evacuationrouteplanner":
            continue

        args = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
        mode = str(args.get("mode", "evacuation")).lower()
        output = call.get("output", {}) if isinstance(call.get("output"), dict) else {}
        routes = output.get("routes", []) if isinstance(output.get("routes"), list) else []

        if mode == "conflict_check":
            conflict_checks += 1
            pending_conflict = True

        local_best_rss: float | None = None
        for route in routes:
            if not isinstance(route, dict):
                continue
            rss_val = route.get("rss")
            if isinstance(rss_val, (int, float)):
                rss = float(rss_val)
                rss_values.append(rss)
                local_best_rss = rss if local_best_rss is None else max(local_best_rss, rss)

            segments = route.get("segments")
            if isinstance(segments, list):
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    damage_score = seg.get("damage_score")
                    if isinstance(damage_score, (int, float)):
                        total_segments += 1
                        if float(damage_score) < 0.6:
                            safe_segments += 1

        if pending_conflict and mode == "evacuation":
            if local_best_rss is not None and local_best_rss >= 1.0:
                replans_safe += 1
            pending_conflict = False

    return {
        "safe_segments": safe_segments,
        "total_segments": total_segments,
        "rss_values": rss_values,
        "conflict_checks": conflict_checks,
        "replans_safe": replans_safe,
    }


def _subtasks_complete(bundle: dict[str, Any]) -> bool:
    calls_by_agent = bundle.get("agent_tool_calls", {}) if isinstance(bundle.get("agent_tool_calls"), dict) else {}
    return all(
        isinstance(calls_by_agent.get(agent), list) and len(calls_by_agent.get(agent)) > 0
        for agent in ("vra", "ga", "pa")
    )


def _build_task(row: dict[str, Any], dataset: str, idx: int) -> dict[str, Any]:
    return {
        "objective": row.get("question") or "",
        "scene_id": f"{dataset}-scene-{row.get('id', idx)}",
        "region": dataset,
        "benchmark": dataset,
        "dataset_row_id": row.get("id", str(idx)),
        "ground_truth": row.get("gt_answer"),
        "source": dataset,
    }


def _score_oea_sample(row: dict[str, Any], ep: dict[str, Any]) -> dict[str, Any]:
    bundle = ep.get("bundle", {}) if isinstance(ep.get("bundle"), dict) else {}
    pred_calls = _flatten_pred_calls(bundle)
    pred_answer = _extract_episode_answer(ep)
    gt_answer = str(row.get("gt_answer", ""))
    gt_calls = row.get("gt_tool_calls", []) if isinstance(row.get("gt_tool_calls"), list) else []

    call_scores = _score_tool_and_args(gt_calls, pred_calls)
    rouge = _rouge_l_f1(pred_answer, gt_answer)
    answer_ok = _oea_answer_correct(pred_answer, gt_answer)

    route_stats = _sample_route_stats(pred_calls)
    sample_tls = TLS(len(gt_calls), len(pred_calls)) if len(pred_calls) > 0 else 0.0
    ccq = float(ep.get("metrics", {}).get("ccq", 0.0))

    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "ok": True,
        "answer_correct": answer_ok,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "rouge_l_f1": rouge,
        "tool_correct": call_scores["tool_correct"],
        "tool_total": call_scores["tool_total"],
        "argn_correct": call_scores["argn_correct"],
        "argn_total": call_scores["argn_total"],
        "argv_correct": call_scores["argv_correct"],
        "argv_total": call_scores["argv_total"],
        "gt_tool_calls": len(gt_calls),
        "pred_tool_calls": len(pred_calls),
        "sample_tls": sample_tls,
        "subtasks_complete": _subtasks_complete(bundle),
        "mgar_hit": _subtasks_complete(bundle) and answer_ok,
        "route_stats": route_stats,
        "ccq": ccq,
        "run_dir": ep.get("run_dir"),
        "episode_metrics": ep.get("metrics", {}),
    }


def _score_thinkgeo_sample(row: dict[str, Any], ep: dict[str, Any]) -> dict[str, Any]:
    bundle = ep.get("bundle", {}) if isinstance(ep.get("bundle"), dict) else {}
    pred_calls = _flatten_pred_calls(bundle)
    pred_answer = _extract_episode_answer(ep)
    gt_answer = row.get("gt_answer")
    gt_calls = row.get("gt_tool_calls", []) if isinstance(row.get("gt_tool_calls"), list) else []

    solvable = bool(row.get("solvable", False))
    gt_match = _thinkgeo_gt_match(pred_answer, gt_answer) if solvable else False
    refused = any(term in _normalize_text(pred_answer) for term in REFUSAL_TERMS)

    tsr_hit = solvable and gt_match
    hrr_hit = (not solvable) and refused
    answer_ok = tsr_hit or hrr_hit

    route_stats = _sample_route_stats(pred_calls)
    sample_tls = TLS(len(gt_calls), len(pred_calls)) if len(pred_calls) > 0 else 0.0
    ccq = float(ep.get("metrics", {}).get("ccq", 0.0))

    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "ok": True,
        "solvable": solvable,
        "is_routing": bool(row.get("is_routing", False)),
        "tsr_hit": tsr_hit,
        "hrr_hit": hrr_hit,
        "answer_correct": answer_ok,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "gt_tool_calls": len(gt_calls),
        "pred_tool_calls": len(pred_calls),
        "sample_tls": sample_tls,
        "subtasks_complete": _subtasks_complete(bundle),
        "mgar_hit": _subtasks_complete(bundle) and answer_ok,
        "route_stats": route_stats,
        "ccq": ccq,
        "run_dir": ep.get("run_dir"),
        "episode_metrics": ep.get("metrics", {}),
    }


def _aggregate_oea(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    inst_correct = sum(1 for s in samples if s.get("answer_correct"))
    tool_correct = sum(int(s.get("tool_correct", 0)) for s in samples)
    tool_total = sum(int(s.get("tool_total", 0)) for s in samples)
    argn_correct = sum(int(s.get("argn_correct", 0)) for s in samples)
    argn_total = sum(int(s.get("argn_total", 0)) for s in samples)
    argv_correct = sum(int(s.get("argv_correct", 0)) for s in samples)
    argv_total = sum(int(s.get("argv_total", 0)) for s in samples)

    rouge_vals = [float(s.get("rouge_l_f1", 0.0)) for s in samples]
    avg_rouge = sum(rouge_vals) / len(rouge_vals) if rouge_vals else 0.0

    gt_tool_calls = sum(int(s.get("gt_tool_calls", 0)) for s in samples)
    pred_tool_calls = sum(int(s.get("pred_tool_calls", 0)) for s in samples)
    mgar_hits = sum(1 for s in samples if s.get("mgar_hit"))
    ccq_vals = [float(s.get("ccq", 0.0)) for s in samples]
    avg_ccq = sum(ccq_vals) / len(ccq_vals) if ccq_vals else 0.0

    safe_segments = sum(int(s.get("route_stats", {}).get("safe_segments", 0)) for s in samples)
    total_segments = sum(int(s.get("route_stats", {}).get("total_segments", 0)) for s in samples)
    rss_vals: list[float] = []
    for s in samples:
        rss_vals.extend([float(v) for v in s.get("route_stats", {}).get("rss_values", []) if isinstance(v, (int, float))])

    if total_segments > 0:
        rss_metric = RSS(safe_segments, total_segments)
        rss_mode = "segment_ratio"
    elif rss_vals:
        rss_metric = sum(max(0.0, min(1.0, v)) for v in rss_vals) / len(rss_vals) * 100.0
        rss_mode = "route_rss_proxy"
    else:
        rss_metric = None
        rss_mode = "unavailable"

    return {
        "benchmark_metrics": {
            "Inst": Inst(inst_correct, total),
            "Tool": Tool(tool_correct, tool_total),
            "ArgN": ArgN(argn_correct, argn_total),
            "ArgV": ArgV(argv_correct, argv_total),
            "Summ": Summ(avg_rouge),
        },
        "novel_metrics": {
            "MGAR": MGAR(mgar_hits, total),
            "TLS": TLS(gt_tool_calls, pred_tool_calls) if pred_tool_calls > 0 else 0.0,
            "RSS": rss_metric,
            "RSS_mode": rss_mode,
            "CCQ": avg_ccq,
            "CRR": None,
            "MTCS": None,
            "DDF1": None,
            "CDF1": None,
            "ReSR": None,
        },
        "counters": {
            "total": total,
            "inst_correct": inst_correct,
            "tool_correct": tool_correct,
            "tool_total": tool_total,
            "argn_correct": argn_correct,
            "argn_total": argn_total,
            "argv_correct": argv_correct,
            "argv_total": argv_total,
            "gt_tool_calls": gt_tool_calls,
            "pred_tool_calls": pred_tool_calls,
        },
    }


def _aggregate_thinkgeo(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    solvable_total = sum(1 for s in samples if s.get("solvable"))
    unsolvable_total = total - solvable_total

    tsr_correct = sum(1 for s in samples if s.get("tsr_hit"))
    hrr_correct = sum(1 for s in samples if s.get("hrr_hit"))

    routing_samples = [s for s in samples if s.get("solvable") and s.get("is_routing")]
    simple_samples = [s for s in samples if s.get("solvable") and not s.get("is_routing")]
    routing_correct = sum(1 for s in routing_samples if s.get("tsr_hit"))
    simple_correct = sum(1 for s in simple_samples if s.get("tsr_hit"))

    gt_tool_calls = sum(int(s.get("gt_tool_calls", 0)) for s in samples)
    pred_tool_calls = sum(int(s.get("pred_tool_calls", 0)) for s in samples)
    mgar_hits = sum(1 for s in samples if s.get("mgar_hit"))
    ccq_vals = [float(s.get("ccq", 0.0)) for s in samples]
    avg_ccq = sum(ccq_vals) / len(ccq_vals) if ccq_vals else 0.0

    safe_segments = sum(int(s.get("route_stats", {}).get("safe_segments", 0)) for s in samples)
    total_segments = sum(int(s.get("route_stats", {}).get("total_segments", 0)) for s in samples)
    rss_vals: list[float] = []
    conflict_checks = 0
    replans_safe = 0
    for s in samples:
        stats = s.get("route_stats", {})
        rss_vals.extend([float(v) for v in stats.get("rss_values", []) if isinstance(v, (int, float))])
        conflict_checks += int(stats.get("conflict_checks", 0))
        replans_safe += int(stats.get("replans_safe", 0))

    if total_segments > 0:
        rss_metric = RSS(safe_segments, total_segments)
        rss_mode = "segment_ratio"
    elif rss_vals:
        rss_metric = sum(max(0.0, min(1.0, v)) for v in rss_vals) / len(rss_vals) * 100.0
        rss_mode = "route_rss_proxy"
    else:
        rss_metric = None
        rss_mode = "unavailable"

    return {
        "benchmark_metrics": {
            "TSR": TSR(tsr_correct, solvable_total),
            "HRR": HRR(hrr_correct, unsolvable_total),
            "TSR_routing": TSR(routing_correct, len(routing_samples)),
            "TSR_simple_gis": TSR(simple_correct, len(simple_samples)),
        },
        "novel_metrics": {
            "MGAR": MGAR(mgar_hits, total),
            "TLS": TLS(gt_tool_calls, pred_tool_calls) if pred_tool_calls > 0 else 0.0,
            "RSS": rss_metric,
            "RSS_mode": rss_mode,
            "CCQ": avg_ccq,
            "ReSR": (replans_safe / conflict_checks * 100.0) if conflict_checks > 0 else None,
            "CRR": None,
            "MTCS": None,
            "DDF1": None,
            "CDF1": None,
        },
        "counters": {
            "total": total,
            "solvable_total": solvable_total,
            "unsolvable_total": unsolvable_total,
            "tsr_correct": tsr_correct,
            "hrr_correct": hrr_correct,
            "routing_total": len(routing_samples),
            "routing_correct": routing_correct,
            "simple_total": len(simple_samples),
            "simple_correct": simple_correct,
            "gt_tool_calls": gt_tool_calls,
            "pred_tool_calls": pred_tool_calls,
        },
    }


def _run_dataset(
    *,
    name: str,
    rows: list[dict[str, Any]],
    runner: EpisodeRunner,
    out_dir: Path,
) -> dict[str, Any]:
    sample_results: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        task = _build_task(row, name, i)
        rec: dict[str, Any] = {
            "id": row.get("id"),
            "index": i,
            "question": row.get("question"),
            "dataset": name,
        }
        try:
            ep = runner.run_single_episode(task)
            if name == "oea":
                scored = _score_oea_sample(row, ep)
            else:
                scored = _score_thinkgeo_sample(row, ep)
            rec.update(scored)
        except Exception as exc:
            rec.update({
                "ok": False,
                "error": str(exc),
                "gt_tool_calls": len(row.get("gt_tool_calls", [])),
                "pred_tool_calls": 0,
            })

        sample_results.append(rec)
        progress = f"[{name}] {i + 1}/{len(rows)}"
        status = "ok" if rec.get("ok") else "error"
        print(f"{progress} {status} id={rec.get('id')} run_dir={rec.get('run_dir')}")

        checkpoint = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dataset": name,
            "n_rows": len(rows),
            "sample_results": sample_results,
        }
        (out_dir / f"{name}_samples.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    ok_samples = [s for s in sample_results if s.get("ok")]
    failed = len(sample_results) - len(ok_samples)

    if name == "oea":
        agg = _aggregate_oea(ok_samples)
    else:
        agg = _aggregate_thinkgeo(ok_samples)

    return {
        "name": name,
        "n_total": len(sample_results),
        "n_ok": len(ok_samples),
        "n_failed": failed,
        "aggregate": agg,
        "sample_results": sample_results,
    }


def build_runner(args: argparse.Namespace) -> EpisodeRunner:
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

    return EpisodeRunner(
        model_id=args.model_id,
        base_url=args.base_url,
        tool_server=args.tool_server,
        agent_model_map=agent_model_map,
        agent_base_url_map=agent_base_url_map,
        allow_mock_fallback=args.allow_mock_fallback,
        max_turns=args.max_turns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-by-one 4-agent MAGRF evaluation on OpenEarthAgent and ThinkGeo, "
            "producing benchmark metrics plus novel coordination metrics."
        )
    )
    parser.add_argument("--oea-path", default="data/openearth_agent/test.json")
    parser.add_argument("--thinkgeo-path", default="data/thinkgeo/ThinkGeoBench.json")
    parser.add_argument("--oea-limit", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--thinkgeo-limit", type=int, default=0, help="0 means full dataset")

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

    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--allow-mock-fallback", action="store_true", help="Only for debugging without running model servers")
    parser.add_argument("--out-root", default="results/full_framework_eval")
    args = parser.parse_args()

    out_dir = _next_eval_dir(Path(args.out_root))
    print(f"Output directory: {out_dir}")

    oea_rows = load_oea_rows(Path(args.oea_path), limit=args.oea_limit)
    tg_rows = load_thinkgeo_rows(Path(args.thinkgeo_path), limit=args.thinkgeo_limit)

    print(f"Loaded OEA rows: {len(oea_rows)}")
    print(f"Loaded ThinkGeo rows: {len(tg_rows)}")

    runner = build_runner(args)

    oea_report = _run_dataset(name="oea", rows=oea_rows, runner=runner, out_dir=out_dir)
    tg_report = _run_dataset(name="thinkgeo", rows=tg_rows, runner=runner, out_dir=out_dir)

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "out_dir": str(out_dir),
        "datasets": {
            "oea": {
                "n_total": oea_report["n_total"],
                "n_ok": oea_report["n_ok"],
                "n_failed": oea_report["n_failed"],
                "aggregate": oea_report["aggregate"],
            },
            "thinkgeo": {
                "n_total": tg_report["n_total"],
                "n_ok": tg_report["n_ok"],
                "n_failed": tg_report["n_failed"],
                "aggregate": tg_report["aggregate"],
            },
        },
        "notes": {
            "metrics_unavailable_without_labels": [
                "CRR (ground-truth conflict labels required)",
                "MTCS (explicit epoch phase labels required)",
                "DDF1 (deadlock injection labels required)",
                "CDF1/ReSR (TDRD conflict subset labels required)",
            ]
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "oea_samples.json").write_text(
        json.dumps({"sample_results": oea_report["sample_results"]}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "thinkgeo_samples.json").write_text(
        json.dumps({"sample_results": tg_report["sample_results"]}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Aggregate Metrics ===")
    print(json.dumps(report["datasets"], indent=2))
    print(f"\nSaved summary to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
