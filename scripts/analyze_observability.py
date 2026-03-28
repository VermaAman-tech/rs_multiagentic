#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_json_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _parse_log_time(line: str, day: str) -> datetime | None:
    # Ex: INFO 03-24 16:15:28 [config.py:717] ...
    m = re.search(r"\b(\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\b", line)
    if not m:
        return None
    mm_dd, hh_mm_ss = m.group(1), m.group(2)
    month, day_num = mm_dd.split("-")
    year = day[:4]
    return datetime.strptime(f"{year}-{month}-{day_num} {hh_mm_ss}", "%Y-%m-%d %H:%M:%S")


def _analyze_log_timings(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"log_found": False}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    run_day = "2026-01-01"

    for line in lines:
        if "Job:" in line and "|" in line:
            # Ex: Job: 80189  |  Tue Mar 24 16:12:58 IST 2026
            m = re.search(r"\b(\d{4})\b", line)
            if m:
                run_day = f"{m.group(1)}-01-01"

        if line.startswith(">>> "):
            if current:
                sections.append(current)
            current = {"title": line.replace(">>> ", "").strip(), "first_ts": None, "last_ts": None}
            continue

        if current:
            ts = _parse_log_time(line, run_day)
            if ts:
                current["first_ts"] = current["first_ts"] or ts
                current["last_ts"] = ts

    if current:
        sections.append(current)

    out_sections: list[dict[str, Any]] = []
    for s in sections:
        duration_s = None
        if s["first_ts"] and s["last_ts"]:
            duration_s = int((s["last_ts"] - s["first_ts"]).total_seconds())
        out_sections.append(
            {
                "title": s["title"],
                "start": s["first_ts"].isoformat() if s["first_ts"] else None,
                "end": s["last_ts"].isoformat() if s["last_ts"] else None,
                "duration_s_from_info_lines": duration_s,
            }
        )

    return {"log_found": True, "sections": out_sections}


def analyze_oea_verbose(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    rows: list[dict[str, Any]] = data.get("per_sample", [])
    by_chain: dict[int, list[dict[str, Any]]] = defaultdict(list)
    tools = Counter()
    parse_ok = 0
    thought_count = 0
    actions_count = 0
    terminate_count = 0

    for r in rows:
        sid = int(r.get("sample_idx", -1))
        by_chain[sid].append(r)
        if r.get("parsed"):
            parse_ok += 1
        pred_name = r.get("pred_name")
        if pred_name:
            tools[pred_name] += 1
            if pred_name.lower() == "terminate":
                terminate_count += 1
        parsed = _try_json_text(r.get("full_raw_output", ""))
        if isinstance(parsed, dict):
            if "thought" in parsed:
                thought_count += 1
            if isinstance(parsed.get("actions"), list):
                actions_count += len(parsed["actions"])

    chain_lengths = []
    chain_tool_sequences: list[dict[str, Any]] = []
    for sid, chain_rows in by_chain.items():
        chain_rows.sort(key=lambda x: int(x.get("turn_idx", 0)))
        seq = [x.get("pred_name") for x in chain_rows if x.get("pred_name")]
        chain_lengths.append(len(chain_rows))
        chain_tool_sequences.append({"sample_idx": sid, "chain_length": len(chain_rows), "tool_sequence": seq})

    chain_tool_sequences.sort(key=lambda x: x["chain_length"], reverse=True)
    top_long = chain_tool_sequences[:10]

    return {
        "file": str(path),
        "n_instances": data.get("n_instances", len(rows)),
        "n_rows": len(rows),
        "n_chains": len(by_chain),
        "parse_success_rate_pct": round((parse_ok / len(rows) * 100) if rows else 0.0, 2),
        "avg_chain_length": round((sum(chain_lengths) / len(chain_lengths)) if chain_lengths else 0.0, 2),
        "max_chain_length": max(chain_lengths) if chain_lengths else 0,
        "terminate_calls": terminate_count,
        "thought_messages_detected": thought_count,
        "total_actions_in_outputs": actions_count,
        "top_tools": tools.most_common(20),
        "longest_chains": top_long,
        "headline_metrics": {k: data.get(k) for k in ["Inst", "Tool", "ArgN", "ArgV", "parse_failures", "n_parsed"]},
        "example_rows": rows[:5],
    }


def analyze_thinkgeo_verbose(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    rows: list[dict[str, Any]] = data.get("per_sample", [])
    tools = Counter()
    refusal_terms = [
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
    ]
    refusals = 0
    tool_call_outputs = 0
    thought_mentions = 0

    for r in rows:
        tc = r.get("extracted_tool_call")
        if isinstance(tc, dict) and tc.get("name"):
            tools[tc["name"]] += 1
            tool_call_outputs += 1
        text = (r.get("full_raw_output") or "").lower()
        if any(t in text for t in refusal_terms):
            refusals += 1
        if "thought" in text:
            thought_mentions += 1

    return {
        "file": str(path),
        "total_samples": data.get("total_samples", len(rows)),
        "solvable_count": data.get("solvable_count"),
        "unsolvable_count": data.get("unsolvable_count"),
        "TSR": data.get("TSR"),
        "HRR": data.get("HRR"),
        "tool_parse_rate": data.get("tool_parse_rate"),
        "detected_tool_calls_in_output": tool_call_outputs,
        "detected_refusals_in_output": refusals,
        "thought_mentions_in_output": thought_mentions,
        "top_extracted_tools": tools.most_common(20),
        "example_rows": rows[:5],
    }


def _render_markdown(oea: dict[str, Any] | None, tg: dict[str, Any] | None, timing: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Agent Observability Report")
    lines.append("")
    lines.append("This report is generated from verbose evaluation artifacts and scheduler logs.")
    lines.append("")

    if timing.get("log_found"):
        lines.append("## Runtime Timing (from infer log)")
        for s in timing.get("sections", []):
            lines.append(
                f"- `{s['title']}`: duration={s['duration_s_from_info_lines']}s, start={s['start']}, end={s['end']}"
            )
        lines.append("")

    if oea:
        lines.append("## OEA Agentic Chain Analysis")
        lines.append(f"- file: `{oea['file']}`")
        lines.append(f"- instances: {oea['n_instances']} (rows={oea['n_rows']})")
        lines.append(f"- inferred chains (by `sample_idx`): {oea['n_chains']}")
        lines.append(f"- parse success: {oea['parse_success_rate_pct']}%")
        lines.append(f"- avg chain length: {oea['avg_chain_length']}, max chain length: {oea['max_chain_length']}")
        lines.append(f"- `Terminate` calls: {oea['terminate_calls']}")
        lines.append(f"- outputs with explicit `thought`: {oea['thought_messages_detected']}")
        lines.append(f"- total `actions` objects in model outputs: {oea['total_actions_in_outputs']}")
        lines.append(f"- headline metrics: {json.dumps(oea['headline_metrics'])}")
        lines.append("- top tools called:")
        for name, count in oea["top_tools"][:10]:
            lines.append(f"  - {name}: {count}")
        lines.append("- longest chains:")
        for ch in oea["longest_chains"][:5]:
            seq = " -> ".join(ch["tool_sequence"][:12])
            lines.append(f"  - sample {ch['sample_idx']}: len={ch['chain_length']} | {seq}")
        lines.append("")

    if tg:
        lines.append("## ThinkGeo Output Analysis")
        lines.append(f"- file: `{tg['file']}`")
        lines.append(f"- total samples: {tg['total_samples']}")
        lines.append(f"- solvable/unsolvable: {tg['solvable_count']}/{tg['unsolvable_count']}")
        lines.append(f"- TSR={tg['TSR']} HRR={tg['HRR']} tool_parse_rate={tg['tool_parse_rate']}")
        lines.append(f"- detected tool-call outputs: {tg['detected_tool_calls_in_output']}")
        lines.append(f"- detected refusal outputs: {tg['detected_refusals_in_output']}")
        lines.append(f"- outputs mentioning thought: {tg['thought_mentions_in_output']}")
        lines.append("- top extracted tools:")
        for name, count in tg["top_extracted_tools"][:10]:
            lines.append(f"  - {name}: {count}")
        lines.append("")

    lines.append("## Notes")
    lines.append("- Current benchmark outputs are model-eval traces, not live VRA/GA/PA runtime episodes.")
    lines.append("- For true per-agent (VRA/GA/PA/ORC) timeline, use `framework/memory/trajectory_log.py` during `EpisodeRunner` runs.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze agent observability from verbose outputs.")
    parser.add_argument("--oea", default="results/e2_oea_verbose.json")
    parser.add_argument("--thinkgeo", default="results/e5_thinkgeo_verbose_fixcheck.json")
    parser.add_argument("--infer-log", default="logs/infer_80189.out")
    parser.add_argument("--out-json", default="results/agent_observability_report.json")
    parser.add_argument("--out-md", default="results/agent_observability_report.md")
    args = parser.parse_args()

    oea_path = Path(args.oea)
    tg_path = Path(args.thinkgeo)
    log_path = Path(args.infer_log)

    oea = analyze_oea_verbose(oea_path) if oea_path.exists() else None
    tg = analyze_thinkgeo_verbose(tg_path) if tg_path.exists() else None
    timing = _analyze_log_timings(log_path)

    payload = {"oea": oea, "thinkgeo": tg, "timing": timing}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(_render_markdown(oea, tg, timing), encoding="utf-8")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()

