#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from framework.episode_runner import EpisodeRunner


DEFAULT_QUERY = (
    "Which fire station and police station are closest to each other in Banff National Park, "
    "Alberta, Canada within 3000m buffered area?"
)


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    return str(value)


def preview(value: Any, max_chars: int = 400) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def endpoint_check(url: str, timeout_s: float = 8.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = httpx.get(url, timeout=timeout_s)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        payload: Any
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text[:600]
        return {
            "url": url,
            "ok": resp.status_code < 400,
            "status_code": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "payload_preview": preview(payload),
        }
    except Exception as exc:
        return {
            "url": url,
            "ok": False,
            "status_code": None,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "error": str(exc),
        }


def call_tool(tool_server: str, name: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    url = f"{tool_server.rstrip('/')}/tools/{name}"
    t0 = time.perf_counter()
    try:
        resp = httpx.post(url, json=payload, timeout=timeout_s)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        body: Any
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:600]}

        success_field = None
        if isinstance(body, dict) and isinstance(body.get("success"), bool):
            success_field = body.get("success")

        return {
            "tool": name,
            "ok": resp.status_code < 400,
            "status_code": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "request": payload,
            "response": body,
            "success_field": success_field,
            "has_error_field": bool(isinstance(body, dict) and body.get("error")),
        }
    except Exception as exc:
        return {
            "tool": name,
            "ok": False,
            "status_code": None,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "request": payload,
            "error": str(exc),
        }


def _extract_point(poi: Any) -> tuple[float, float] | None:
    if not isinstance(poi, dict):
        return None
    lat = poi.get("lat")
    lon = poi.get("lon")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    lat_f = float(lat)
    lon_f = float(lon)
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return None
    return (lat_f, lon_f)


def closest_pair_from_tool_calls(
    tool_server: str,
    fire_pois: list[dict[str, Any]],
    police_pois: list[dict[str, Any]],
    timeout_s: float,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for f in fire_pois:
        fp = _extract_point(f)
        if fp is None:
            continue
        for p in police_pois:
            pp = _extract_point(p)
            if pp is None:
                continue
            rec = call_tool(
                tool_server,
                "ComputeDistance",
                {"point_a": [fp[0], fp[1]], "point_b": [pp[0], pp[1]]},
                timeout_s=timeout_s,
            )
            dist = None
            if isinstance(rec.get("response"), dict):
                v = rec["response"].get("distance_meters")
                if isinstance(v, (int, float)):
                    dist = float(v)
            pairs.append(
                {
                    "fire_name": str(f.get("name", "")),
                    "police_name": str(p.get("name", "")),
                    "distance_meters": dist,
                    "compute_distance": rec,
                }
            )

    valid = [p for p in pairs if isinstance(p.get("distance_meters"), (int, float))]
    valid.sort(key=lambda x: float(x["distance_meters"]))
    return {
        "n_pairs_tested": len(pairs),
        "n_pairs_with_distance": len(valid),
        "closest_pair": valid[0] if valid else None,
        "pairs": valid[:20],
    }


def run_tool_stress(args: argparse.Namespace) -> dict[str, Any]:
    core_boundary = call_tool(
        args.tool_server,
        "GetAreaBoundary",
        {"area_name": "Banff National Park, Alberta, Canada", "buffer_m": 3000},
        timeout_s=args.tool_timeout_s,
    )

    boundary_bbox = None
    if isinstance(core_boundary.get("response"), dict):
        bbox = core_boundary["response"].get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            boundary_bbox = [float(x) for x in bbox]

    if boundary_bbox is None:
        boundary_bbox = [-117.317115, 50.7050607, -115.1649126, 52.2714549]

    core_fire = call_tool(
        args.tool_server,
        "AddPoisLayer",
        {"poi_category": "fire_station", "bbox": boundary_bbox},
        timeout_s=args.tool_timeout_s,
    )
    core_police = call_tool(
        args.tool_server,
        "AddPoisLayer",
        {"poi_category": "police_station", "bbox": boundary_bbox},
        timeout_s=args.tool_timeout_s,
    )

    fire_pois = []
    police_pois = []
    if isinstance(core_fire.get("response"), dict) and isinstance(core_fire["response"].get("pois"), list):
        fire_pois = [x for x in core_fire["response"].get("pois", []) if isinstance(x, dict)]
    if isinstance(core_police.get("response"), dict) and isinstance(core_police["response"].get("pois"), list):
        police_pois = [x for x in core_police["response"].get("pois", []) if isinstance(x, dict)]

    zero_coord_count = 0
    for row in fire_pois + police_pois:
        pt = _extract_point(row)
        if pt is not None and abs(pt[0]) < 1e-9 and abs(pt[1]) < 1e-9:
            zero_coord_count += 1

    closest = closest_pair_from_tool_calls(
        args.tool_server,
        fire_pois,
        police_pois,
        timeout_s=args.tool_timeout_s,
    )

    all_tool_payloads: list[tuple[str, dict[str, Any]]] = [
        ("TemporalStackLoader", {"aoi_bbox": [-117.0, 51.0, -116.5, 51.5], "date_range": ["2023-06-01", "2023-08-31"]}),
        ("RoadDamageScorer", {"gpkg_path": "data/tmp/missing.gpkg", "damage_raster_layer": "damage"}),
        ("EvacuationRoutePlanner", {"graph_path": "data/tmp/missing.gpkg", "origins": ["A"], "destinations": ["B"]}),
        ("PrithviEmbed", {"raster_path": "data/tmp/missing.tif"}),
        ("ObjectDetection", {"image_path": "data/tmp/missing.png", "text_prompt": "building"}),
        ("SegmentObjectPixels", {"image_path": "data/tmp/missing.png", "bboxes": [[0, 0, 10, 10]]}),
        ("ImageDescription", {"image_path": "data/tmp/missing.png"}),
        ("TextToBbox", {"image_path": "data/tmp/missing.png", "text_prompt": "helicopter"}),
        ("RegionAttributeDescription", {"image_path": "data/tmp/missing.png", "bbox": [0, 0, 10, 10]}),
        ("CountGivenObject", {"image_path": "data/tmp/missing.png", "text_prompt": "car"}),
        ("OCR", {"image_path": "data/tmp/missing.png"}),
        ("ChangeDetection", {"image_path_1": "data/tmp/pre.png", "image_path_2": "data/tmp/post.png"}),
        ("DrawBox", {"image_path": "data/tmp/missing.png", "bboxes": [[0, 0, 10, 10]], "output_path": "data/tmp/draw_box.png"}),
        ("AddText", {"image_path": "data/tmp/missing.png", "text": "test", "position": [5, 5], "output_path": "data/tmp/add_text.png"}),
        ("GetAreaBoundary", {"area_name": "Banff National Park, Alberta, Canada", "buffer_m": 3000}),
        ("AddPoisLayer", {"poi_category": "fire_station", "bbox": boundary_bbox}),
        ("ComputeDistance", {"point_a": [51.178, -115.571], "point_b": [51.176, -115.560]}),
        ("DisplayOnMap", {"features": [], "output_html": "data/tmp/map.html"}),
        ("GetBboxFromGeotiff", {"geotiff_path": "data/tmp/missing.tif"}),
        ("DisplayGeotiff", {"geotiff_path": "data/tmp/missing.tif", "features": [], "output_path": "data/tmp/display_geotiff.png"}),
        ("DisplayOnGeotiff", {"geotiff_path": "data/tmp/missing.tif", "features": [], "output_path": "data/tmp/display_on_geotiff.png"}),
        ("AddIndexLayer", {"geotiff_path": "data/tmp/missing.tif", "index_name": "NDVI"}),
        ("ComputeIndexChange", {"index_path_pre": "data/tmp/index_pre.tif", "index_path_post": "data/tmp/index_post.tif"}),
        ("ShowIndexLayer", {"index_array_path": "data/tmp/index.tif"}),
        ("Calculator", {"expression": "2+2"}),
        ("Solver", {"equation": "x**2-4"}),
        ("Plot", {"x_values": [0, 1, 2], "y_values": [0, 1, 4], "output_path": "data/tmp/plot.png"}),
        ("GoogleSearch", {"query": "Banff fire station police station nearest", "k": 3}),
        ("Terminate", {"ans": "ok"}),
    ]

    all_results = [
        call_tool(args.tool_server, name, payload, timeout_s=args.tool_timeout_s)
        for name, payload in all_tool_payloads
    ]

    ok_count = sum(1 for r in all_results if r.get("ok"))
    success_true_count = sum(1 for r in all_results if r.get("success_field") is True)
    with_error_count = sum(1 for r in all_results if r.get("has_error_field"))

    return {
        "core": {
            "get_area_boundary": core_boundary,
            "add_pois_fire": core_fire,
            "add_pois_police": core_police,
            "fire_pois_count": len(fire_pois),
            "police_pois_count": len(police_pois),
            "zero_coord_count": zero_coord_count,
            "closest_pair_computation": closest,
        },
        "all_tools": {
            "total": len(all_results),
            "http_ok_count": ok_count,
            "success_true_count": success_true_count,
            "responses_with_error_field": with_error_count,
            "results": all_results,
        },
    }


def check_react_compliance(agent_trace: list[dict[str, Any]]) -> dict[str, Any]:
    total_tool_turns = 0
    missing_thought = 0
    for tr in agent_trace:
        if not isinstance(tr, dict):
            continue
        calls = tr.get("tool_calls") if isinstance(tr.get("tool_calls"), list) else []
        if not calls:
            continue
        total_tool_turns += 1
        thought = tr.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            missing_thought += 1
    return {
        "tool_turns": total_tool_turns,
        "missing_thought_turns": missing_thought,
        "compliant": (missing_thought == 0),
    }


def detect_halving_pattern(distances: list[float]) -> bool:
    if len(distances) < 5:
        return False
    checks = []
    for i in range(1, len(distances)):
        prev = distances[i - 1]
        cur = distances[i]
        if prev <= 0:
            continue
        target = prev / 2.0
        checks.append(abs(cur - target) <= max(1.0, 0.15 * prev))
    return bool(checks) and all(checks)


def summarize_episode(ep: dict[str, Any], run_index: int) -> dict[str, Any]:
    bundle = ep.get("bundle", {}) if isinstance(ep.get("bundle"), dict) else {}
    metrics = ep.get("metrics", {}) if isinstance(ep.get("metrics"), dict) else {}

    traces = bundle.get("agent_traces", {}) if isinstance(bundle.get("agent_traces"), dict) else {}
    tool_calls = bundle.get("agent_tool_calls", {}) if isinstance(bundle.get("agent_tool_calls"), dict) else {}
    mpc_msgs = bundle.get("mpc_messages", []) if isinstance(bundle.get("mpc_messages"), list) else []

    ga_calls = tool_calls.get("ga", []) if isinstance(tool_calls.get("ga"), list) else []
    pa_calls = tool_calls.get("pa", []) if isinstance(tool_calls.get("pa"), list) else []

    ga_add_poi_outputs = [
        c.get("output", {})
        for c in ga_calls
        if isinstance(c, dict) and str(c.get("tool")) == "AddPoisLayer"
    ]
    ga_poi_points: list[tuple[float, float]] = []
    for out in ga_add_poi_outputs:
        if not isinstance(out, dict) or not isinstance(out.get("pois"), list):
            continue
        for poi in out.get("pois", []):
            pt = _extract_point(poi)
            if pt is not None:
                ga_poi_points.append(pt)

    ga_zero_points = [pt for pt in ga_poi_points if abs(pt[0]) < 1e-9 and abs(pt[1]) < 1e-9]

    pa_distances = []
    for c in pa_calls:
        if not isinstance(c, dict) or str(c.get("tool")) != "ComputeDistance":
            continue
        out = c.get("output", {}) if isinstance(c.get("output"), dict) else {}
        d = out.get("distance_meters")
        if isinstance(d, (int, float)):
            pa_distances.append(float(d))

    context_keys_by_agent: dict[str, list[str]] = {}
    for row in mpc_msgs:
        if not isinstance(row, dict):
            continue
        msg = row.get("message", {}) if isinstance(row.get("message"), dict) else {}
        if str(msg.get("msg_type")) != "TASK":
            continue
        receiver = str(msg.get("receiver", ""))
        payload = msg.get("payload", {}) if isinstance(msg.get("payload"), dict) else {}
        keys = payload.get("em_context_keys") if isinstance(payload.get("em_context_keys"), list) else []
        context_keys_by_agent[receiver] = [str(k) for k in keys]

    react = {
        "ga": check_react_compliance(traces.get("ga", []) if isinstance(traces.get("ga"), list) else []),
        "vra": check_react_compliance(traces.get("vra", []) if isinstance(traces.get("vra"), list) else []),
        "pa": check_react_compliance(traces.get("pa", []) if isinstance(traces.get("pa"), list) else []),
    }

    findings: list[str] = []
    if metrics.get("safety_valid") is False:
        findings.append("episode_safety_invalid")
    if metrics.get("agent_error_count", 0) not in (0, None):
        findings.append("agent_errors_present")
    if not ga_poi_points:
        findings.append("ga_no_poi_points")
    if ga_zero_points:
        findings.append("ga_zero_coordinate_pois")
    if detect_halving_pattern(pa_distances):
        findings.append("pa_distance_halving_pattern")
    if not react["ga"]["compliant"] or not react["vra"]["compliant"] or not react["pa"]["compliant"]:
        findings.append("react_thought_missing")

    return {
        "run_index": run_index,
        "run_dir": ep.get("run_dir"),
        "metrics": metrics,
        "episode_metrics": bundle.get("episode_metrics", {}),
        "plan": bundle.get("plan", {}),
        "executed_agent_order": bundle.get("executed_agent_order", []),
        "context_keys_by_agent": context_keys_by_agent,
        "ga_poi_points_count": len(ga_poi_points),
        "ga_zero_points_count": len(ga_zero_points),
        "pa_distance_call_count": len(pa_distances),
        "pa_distance_stats": {
            "min": min(pa_distances) if pa_distances else None,
            "max": max(pa_distances) if pa_distances else None,
            "mean": statistics.mean(pa_distances) if pa_distances else None,
        },
        "react": react,
        "findings": findings,
    }


def run_pipeline_stress(args: argparse.Namespace) -> dict[str, Any]:
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

    raw_runs: list[dict[str, Any]] = []
    summarized_runs: list[dict[str, Any]] = []

    for i in range(1, args.repeats + 1):
        t0 = time.perf_counter()
        ep_root = runner.run_episode(args.query)
        elapsed_s = round(time.perf_counter() - t0, 2)
        ep = ep_root.get("episode", {}) if isinstance(ep_root, dict) else {}
        raw_runs.append(
            {
                "index": i,
                "elapsed_s": elapsed_s,
                "run_dir": ep.get("run_dir"),
                "metrics": ep.get("metrics", {}),
                "result_preview": preview(ep.get("result", {})),
            }
        )
        summarized = summarize_episode(ep, run_index=i)
        summarized["elapsed_s"] = elapsed_s
        summarized_runs.append(summarized)

    safety_pass_runs = sum(1 for r in summarized_runs if r.get("metrics", {}).get("safety_valid") is True)
    task_success_runs = sum(1 for r in summarized_runs if r.get("metrics", {}).get("task_success") is True)

    return {
        "raw_runs": raw_runs,
        "summaries": summarized_runs,
        "aggregate": {
            "repeats": args.repeats,
            "safety_pass_runs": safety_pass_runs,
            "task_success_runs": task_success_runs,
            "all_task_success": task_success_runs == args.repeats,
        },
    }


def build_markdown_report(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Banff Full Pipeline Audit")
    lines.append("")
    lines.append(f"- created_at: {data.get('created_at')}")
    lines.append(f"- query: {data.get('query')}")
    lines.append(f"- repeats: {data.get('config', {}).get('repeats')}")
    lines.append("")

    lines.append("## Health")
    for rec in data.get("health", []):
        if not isinstance(rec, dict):
            continue
        status = "PASS" if rec.get("ok") else "FAIL"
        lines.append(f"- {status} {rec.get('url')} status={rec.get('status_code')} elapsed_ms={rec.get('elapsed_ms')}")
    lines.append("")

    lines.append("## Core Tool Chain")
    core = data.get("tool_stress", {}).get("core", {}) if isinstance(data.get("tool_stress"), dict) else {}
    lines.append(f"- fire_pois_count: {core.get('fire_pois_count')}")
    lines.append(f"- police_pois_count: {core.get('police_pois_count')}")
    lines.append(f"- zero_coord_count: {core.get('zero_coord_count')}")
    closest = core.get("closest_pair_computation", {}) if isinstance(core.get("closest_pair_computation"), dict) else {}
    lines.append(f"- pairwise_distance_candidates: {closest.get('n_pairs_with_distance')}")
    lines.append(f"- closest_pair: {preview(closest.get('closest_pair'))}")
    lines.append("")

    all_tools = data.get("tool_stress", {}).get("all_tools", {}) if isinstance(data.get("tool_stress"), dict) else {}
    lines.append("## All Tools Smoke")
    lines.append(f"- total_tools_checked: {all_tools.get('total')}")
    lines.append(f"- http_ok_count: {all_tools.get('http_ok_count')}")
    lines.append(f"- success_true_count: {all_tools.get('success_true_count')}")
    lines.append(f"- responses_with_error_field: {all_tools.get('responses_with_error_field')}")
    lines.append("")

    pipeline = data.get("pipeline_stress", {}) if isinstance(data.get("pipeline_stress"), dict) else {}
    agg = pipeline.get("aggregate", {}) if isinstance(pipeline.get("aggregate"), dict) else {}
    lines.append("## Pipeline Stress")
    lines.append(f"- repeats: {agg.get('repeats')}")
    lines.append(f"- safety_pass_runs: {agg.get('safety_pass_runs')}")
    lines.append(f"- task_success_runs: {agg.get('task_success_runs')}")
    lines.append("")

    lines.append("## Per-Run Findings")
    for row in pipeline.get("summaries", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- run {row.get('run_index')} dir={row.get('run_dir')} safety={row.get('metrics', {}).get('safety_valid')} "
            f"task_success={row.get('metrics', {}).get('task_success')} findings={row.get('findings')}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Banff testcase pipeline, tools, memory/context, and ReAct traces.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tool-timeout-s", type=float, default=60.0)

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

    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--allow-mock-fallback", action="store_true")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json) if args.out_json else Path(f"logs/banff_full_pipeline_audit_{stamp}.json")
    out_md = Path(args.out_md) if args.out_md else Path(f"logs/banff_full_pipeline_audit_{stamp}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    model_urls = sorted(
        {
            str(args.base_url).rstrip("/"),
            str(args.orc_base_url).rstrip("/"),
            str(args.vra_base_url).rstrip("/"),
            str(args.ga_base_url).rstrip("/"),
            str(args.pa_base_url).rstrip("/"),
        }
    )
    health = [endpoint_check(f"{args.tool_server.rstrip('/')}/health")]
    health.extend(endpoint_check(f"{u}/models") for u in model_urls)
    health.append(endpoint_check("http://127.0.0.1:8501"))

    tool_stress = run_tool_stress(args)
    pipeline_stress = run_pipeline_stress(args)

    report = {
        "created_at": utc_now(),
        "query": args.query,
        "config": {
            "repeats": args.repeats,
            "tool_timeout_s": args.tool_timeout_s,
            "max_turns": args.max_turns,
            "allow_mock_fallback": args.allow_mock_fallback,
            "model_id": args.model_id,
            "base_url": args.base_url,
            "tool_server": args.tool_server,
            "agent_models": {
                "orc": args.orc_model_id,
                "vra": args.vra_model_id,
                "ga": args.ga_model_id,
                "pa": args.pa_model_id,
            },
            "agent_base_urls": {
                "orc": args.orc_base_url,
                "vra": args.vra_base_url,
                "ga": args.ga_base_url,
                "pa": args.pa_base_url,
            },
        },
        "health": health,
        "tool_stress": tool_stress,
        "pipeline_stress": pipeline_stress,
    }

    out_json.write_text(json.dumps(safe_json(report), indent=2), encoding="utf-8")
    out_md.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"JSON report: {out_json}")
    print(f"Markdown report: {out_md}")


if __name__ == "__main__":
    main()
