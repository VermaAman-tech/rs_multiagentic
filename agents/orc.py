from __future__ import annotations
import json
import re
from typing import Any
from agents.base_agent import AgentResult, BaseAgent
from framework.protocols.conflict import resolve_conflict
from framework.protocols.safety import validate_safety_payload

class Orchestrator(BaseAgent):
    name = "orc"

    _AGENT_CAPABILITIES = {
        "vra": "ObjectDetection, TextToBbox, ImageDescription, ChangeDetection, CountGivenObject, SegmentObjectPixels, RegionAttributeDescription, OCR, DrawBox, AddText, GetBboxFromGeotiff, AddIndexLayer, ComputeIndexChange, ShowIndexLayer, TemporalStackLoader, PrithviEmbed",
        "ga": "GetAreaBoundary, AddPoisLayer, ComputeDistance, DisplayOnMap, GetBboxFromGeotiff, DisplayOnGeotiff, AddIndexLayer, ComputeIndexChange, RoadDamageScorer",
        "pa": "EvacuationRoutePlanner, ComputeDistance, Calculator, Solver, Plot",
    }

    def _get_system_prompt(self) -> str:
        return (
            "You are ORC, the Orchestrator of MAGRF (VRA, GA, PA).\n\n"
            "MISSION: Turn the user objective into a minimal, reliable plan. Route data between agents. "
            "Synthesize a concrete final answer grounded in their evidence.\n\n"
            "AGENT SELECTION (pick the smallest set that answers the question):\n"
            "- VRA: images, aerial/satellite photos, bbox, pixels, detect/count/OCR, visual change.\n"
            "- GA: place names, boundaries, POIs, real-world distance/proximity, spectral NDVI/NDBI/NDWI.\n"
            "- PA: routes, evacuation, math/Solver, pairwise ComputeDistance, final decision framing.\n\n"
            "PLAN RULES:\n"
            "- Spectral change (NDVI/NDBI/NDWI/index): GA only (optionally + PA for narration).\n"
            "- Proximity with images (e.g. 'distance between X and Y in this image'): VRA (+ PA for math).\n"
            "- Proximity with real-world POIs (fire/police/shelter/hospital): GA then PA.\n"
            "- Visual QA/detection/OCR: VRA only (add PA only if numeric arithmetic required).\n"
            "- Disaster/flood/damage: VRA + GA + PA.\n"
            "- Never add an agent that has no concrete contribution.\n\n"
            "SUB-TASK CONTRACT (what you send each agent):\n"
            "- State the exact data product needed (e.g. 'closest_pair + distance_meters between hospitals and shelters within 5000m of Tokyo Tower').\n"
            "- Pass forward upstream evidence in concrete terms (names, coordinates, bboxes, counts, layer paths).\n"
            "- Never say 'do your best' - always state the deliverable.\n\n"
            "CONFLICT / SAFETY:\n"
            "- Tier0 evidence presence wins over confidence. A high-confidence 'insufficient' answer loses to any answer with real distance_meters/closest_pair/bbox/etc.\n"
            "- Never finalize with fabricated coordinates, routes, or numbers.\n"
            "- When agents disagree, trust the one with more grounded tool calls.\n\n"
            "FINAL ANSWER SYNTHESIS:\n"
            "- Output a single concrete answer (name + number + unit when applicable).\n"
            "- Prefer structured facts from agent outputs (distance_meters, closest_pair, counts, index stats) verbatim.\n"
            "- If evidence is partial, give a partial concrete answer - do NOT blanket-fail.\n\n"
            "TOOL BOUNDARY: ORC may only call GoogleSearch, Calculator, Terminate. Domain work goes to VRA/GA/PA."
        )

    def _classify_query_type(self, objective: str) -> str:
        o = (objective or "").lower()
        if any(k in o for k in ("ndbi", "ndvi", "index", "urban growth", "urban decrease")):
            return "spectral_change"
        # Visual/imagery signals - covers both explicit image mentions and pixel-based
        # measurement hints (GSD/m/pixel) that are intrinsic to image reasoning.
        imagery_signals = (
            "image", "aerial", "satellite", "detect", "bbox", "pixel",
            "ocr", "gsd", "m/pixel", "m/px", "per pixel", "meter per pixel",
        )
        if any(k in o for k in imagery_signals):
            if any(k in o for k in ("change", "before", "after", "temporal")):
                return "change_detection"
            return "visual_qa"
        if any(k in o for k in ("flood", "damage", "disaster", "evacuation", "route", "hazard")):
            return "disaster_assessment"
        if any(k in o for k in ("closest", "nearest", "distance", "proximity", "within")):
            return "geospatial_proximity"
        return "mixed_analysis"

    def _default_sub_task(self, agent_name: str, objective: str) -> str:
        objective_l = (objective or "").lower()
        is_spectral = any(k in objective_l for k in ("ndbi", "ndvi", "index", "urban growth", "urban decrease"))
        if is_spectral and agent_name == "ga":
            return (
                "Run boundary-first spectral analysis: call GetAreaBoundary for the AOI, compute index layers "
                "for each required epoch with AddIndexLayer, then call ComputeIndexChange and summarize the "
                "observed trend with explicit numeric evidence."
            )
        if is_spectral and agent_name == "pa":
            return (
                "Validate GA spectral outputs, ensure the summary is numerically consistent, and produce a concise "
                "decision-ready statement without inventing unavailable statistics."
            )
        if is_spectral and agent_name == "vra":
            return "Provide backup spectral/imagery interpretation only if GA spectral outputs are missing or inconsistent."
        if agent_name == "vra":
            return f"Extract visual evidence needed for objective: {objective}"
        if agent_name == "ga":
            return f"Run GIS analysis and produce structured geospatial facts for objective: {objective}"
        if agent_name == "pa":
            return f"Compute distances/routes and return decision-ready planning outputs for objective: {objective}"
        return f"Support objective: {objective}"

    def _normalize_query_type(self, query_type: str, objective: str) -> str:
        q = str(query_type or "").strip().lower()
        if not q:
            return self._classify_query_type(objective)
        objective_l = (objective or "").lower()
        imagery_hint = any(k in objective_l for k in ("image", "aerial", "satellite", "detect", "bbox", "pixel", "ocr", "gsd", "m/pixel", "m/px", "per pixel"))
        if "spectral" in q or "index" in q:
            return "spectral_change"
        if "change" in q and any(k in q for k in ("visual", "image", "temporal")):
            return "change_detection"
        if "visual" in q:
            return "visual_qa"
        if "change" in q:
            return "change_detection"
        if "disaster" in q or "hazard" in q:
            return "disaster_assessment"
        # Preserve imagery distance/count questions as visual tasks.
        # Any imagery hint (including GSD/m/pixel) beats proximity/distance routing.
        if imagery_hint and ("distance" in q or "proximity" in q):
            return "visual_qa"
        if "proximity" in q or "distance" in q:
            return "geospatial_proximity"
        if q in {"mixed", "mixed_analysis"}:
            return "mixed_analysis"
        return self._classify_query_type(objective)

    def _default_plan(self, objective: str, available_agents: list[str]) -> dict[str, Any]:
        qtype = self._classify_query_type(objective)
        objective_l = (objective or "").lower()
        if qtype == "geospatial_proximity":
            seq = [a for a in ["ga", "pa"] if a in available_agents]
        elif qtype == "spectral_change":
            if "ga" in available_agents:
                seq = ["ga"]
                if "pa" in available_agents:
                    seq.append("pa")
            else:
                seq = [a for a in ["vra", "pa"] if a in available_agents]
        elif qtype in {"visual_qa", "change_detection"}:
            if qtype == "visual_qa":
                needs_math = any(k in objective_l for k in ("distance", "area", "meter", "meters", "pixel", "count"))
                preferred = ["vra", "pa"] if needs_math else ["vra"]
                seq = [a for a in preferred if a in available_agents]
            else:
                seq = [a for a in ["vra", "ga", "pa"] if a in available_agents]
        elif qtype == "disaster_assessment":
            seq = [a for a in ["vra", "ga", "pa"] if a in available_agents]
        else:
            seq = [a for a in ["ga", "vra", "pa"] if a in available_agents]

        sub_tasks: dict[str, str] = {}
        for a in seq:
            sub_tasks[a] = self._default_sub_task(a, objective)

        return {
            "query_type": qtype,
            "agent_sequence": seq,
            "sub_tasks": sub_tasks,
            "rationale": "Fallback deterministic ORC plan based on objective classification.",
        }

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def plan(
        self,
        *,
        objective: str,
        available_agents: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fallback = self._default_plan(objective, available_agents)

        messages = [
            {
                "role": "system",
                "content": (
                    self._get_system_prompt()
                    + "\n\nReturn only strict JSON with keys: query_type, agent_sequence, sub_tasks, rationale."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": objective,
                        "available_agents": available_agents,
                        "agent_capabilities": {k: self._AGENT_CAPABILITIES.get(k, "") for k in available_agents},
                        "context": context or {},
                        "constraints": {
                            "agent_sequence_must_use_subset_of": available_agents,
                            "avoid_unnecessary_vra_for_non_imagery_queries": True,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        resp = self._call_llm(messages, tools=[])
        parsed = self._extract_json(str(resp.get("content", "")))
        if not isinstance(parsed, dict):
            return fallback

        seq = parsed.get("agent_sequence")
        if not isinstance(seq, list):
            seq = fallback["agent_sequence"]
        seq = [str(a).lower().strip() for a in seq if str(a).lower().strip() in available_agents]
        if not seq:
            seq = fallback["agent_sequence"]

        sub_tasks = parsed.get("sub_tasks")
        if not isinstance(sub_tasks, dict):
            sub_tasks = {}
        merged_sub_tasks = dict(fallback["sub_tasks"])
        for k, v in sub_tasks.items():
            kk = str(k).lower().strip()
            if kk in seq and isinstance(v, str) and v.strip():
                merged_sub_tasks[kk] = v.strip()

        resolved_query_type = self._normalize_query_type(
            str(parsed.get("query_type") or fallback["query_type"]),
            objective,
        )
        if resolved_query_type == "geospatial_proximity":
            objective_l = (objective or "").lower()
            imagery_hint = any(k in objective_l for k in ("image", "aerial", "satellite", "detect", "bbox", "pixel", "ocr", "gsd", "m/pixel", "m/px", "per pixel"))
            if imagery_hint and "vra" in available_agents:
                resolved_query_type = "visual_qa"
                seq = [a for a in ["vra", "pa"] if a in available_agents]
            else:
                seq = [a for a in seq if a in {"ga", "pa"}]
                if not seq:
                    seq = [a for a in ["ga", "pa"] if a in available_agents]
        elif resolved_query_type == "visual_qa":
            objective_l = (objective or "").lower()
            needs_math = any(k in objective_l for k in ("distance", "area", "meter", "meters", "pixel", "count", "calculate"))
            preferred = ["vra", "pa"] if needs_math else ["vra"]
            seq = [a for a in preferred if a in available_agents]
        elif resolved_query_type == "spectral_change":
            if "ga" in available_agents:
                keep_pa = "pa" in available_agents and "pa" in seq
                seq = ["ga"] + (["pa"] if keep_pa else [])
            else:
                seq = [a for a in ["vra", "pa"] if a in available_agents and a in seq]
                if not seq:
                    seq = [a for a in ["vra", "pa"] if a in available_agents]

        resolved_sub_tasks: dict[str, str] = {}
        for a in seq:
            sub = merged_sub_tasks.get(a)
            if not isinstance(sub, str) or not sub.strip():
                sub = self._default_sub_task(a, objective)
            resolved_sub_tasks[a] = sub

        return {
            "query_type": resolved_query_type,
            "agent_sequence": seq,
            "sub_tasks": resolved_sub_tasks,
            "rationale": str(parsed.get("rationale") or fallback["rationale"]),
        }

    def evaluate_next_step(
        self,
        *,
        objective: str,
        completed_agents: list[str],
        pending_agents: list[str],
        latest_agent: str,
        latest_error: str | None,
        latest_output_preview: str,
        retry_count: int = 0,
    ) -> dict[str, Any]:
        default_action = {
            "action": ("skip" if pending_agents else "terminate") if latest_error else ("continue" if pending_agents else "terminate"),
            "next_agent": (pending_agents[0] if pending_agents else None),
            "reason": "Fallback ORC next-step decision.",
        }

        messages = [
            {
                "role": "system",
                "content": (
                    self._get_system_prompt()
                    + "\n\nReturn strict JSON: {action, next_agent, reason}."
                    + " Action must be one of: retry, continue, skip, abort, terminate."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": objective,
                        "completed_agents": completed_agents,
                        "pending_agents": pending_agents,
                        "latest_agent": latest_agent,
                        "latest_error": latest_error,
                        "latest_output_preview": latest_output_preview,
                        "retry_count": int(retry_count),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        resp = self._call_llm(messages, tools=[])
        parsed = self._extract_json(str(resp.get("content", "")))
        if not isinstance(parsed, dict):
            return default_action

        action = str(parsed.get("action", "")).lower().strip()
        if action not in {"retry", "continue", "skip", "abort", "terminate"}:
            action = default_action["action"]

        if action == "retry":
            next_agent_norm = latest_agent
        elif action in {"continue", "skip"}:
            next_agent = parsed.get("next_agent")
            next_agent_norm = str(next_agent).lower().strip() if isinstance(next_agent, str) else None
            if next_agent_norm not in pending_agents:
                next_agent_norm = pending_agents[0] if pending_agents else None
        else:
            next_agent_norm = None

        return {
            "action": action,
            "next_agent": next_agent_norm,
            "reason": str(parsed.get("reason", "")) or default_action["reason"],
        }

    def synthesize_answer(
        self,
        *,
        objective: str,
        agent_results: dict[str, AgentResult],
        merged_state: dict[str, Any],
    ) -> str:
        def _compact(value: Any, max_chars: int = 1800) -> str:
            try:
                txt = json.dumps(value, ensure_ascii=False)
            except Exception:
                txt = str(value)
            if len(txt) <= max_chars:
                return txt
            return txt[: max_chars - 3] + "..."

        packed_results: dict[str, Any] = {}
        for name, res in agent_results.items():
            if not hasattr(res, "output") or not isinstance(res.output, dict):
                continue
            packed_results[name] = {
                "output": res.output,
                "tool_calls": len(getattr(res, "tool_calls", []) or []),
                "turns": int(getattr(res, "n_turns", 0) or 0),
            }

        # Evidence-richness check: before declaring failure, look for any concrete
        # evidence (distance_meters, closest_pair, bbox, pois, etc.) that can
        # anchor a real answer. Only bail out if ZERO evidence exists.
        evidence_keys = (
            "distance_meters", "closest_pair", "segment_scores", "damage_polygons",
            "flood_extent", "impassable_roads", "change_map_path", "routes",
            "pois", "boundary", "bbox", "index_statistics", "distance_evidence",
            "distance_observations", "detections", "count", "description",
            "index_change", "ndvi", "ndbi", "ndwi",
        )
        any_evidence_found = False
        if agent_results:
            for name, res in agent_results.items():
                output = res.output if isinstance(res.output, dict) else {}
                for k in evidence_keys:
                    v = output.get(k)
                    if v not in (None, "", [], {}):
                        any_evidence_found = True
                        break
                if any_evidence_found:
                    break
            # Also check merged_state (post-aggregate) for conflict-resolved evidence.
            if not any_evidence_found and isinstance(merged_state, dict):
                for k in evidence_keys:
                    v = merged_state.get(k)
                    if v not in (None, "", [], {}):
                        any_evidence_found = True
                        break

        # Only return consolidated failure if NO evidence AND every agent explicitly failed.
        if agent_results and not any_evidence_found:
            failing_agents: dict[str, str] = {}
            for name, res in agent_results.items():
                output = res.output if isinstance(res.output, dict) else {}
                ans = str(output.get("ans", "") or "").strip()
                ans_l = ans.lower()
                raw_l = str(output.get("raw_output", "") or "").lower()
                thought_l = str(output.get("thought", "") or "").lower()

                tool_errors: list[str] = []
                for call in getattr(res, "tool_calls", []) or []:
                    if not isinstance(call, dict):
                        continue
                    out = call.get("output", {}) if isinstance(call.get("output"), dict) else {}
                    err = str(out.get("error", "") or "").strip()
                    if out.get("success") is False:
                        tool_errors.append(err or "tool returned success=false")
                    elif err:
                        tool_errors.append(err)

                explicit_failure = (
                    ans_l in {"", "error", "timeout"}
                    or "unable to" in ans_l
                    or "insufficient" in ans_l
                    or "not available" in ans_l
                    or bool(tool_errors)
                )
                if explicit_failure:
                    reason = "; ".join(dict.fromkeys([e for e in tool_errors if e]))
                    if not reason:
                        reason = ans or str(output.get("thought", "") or "agent failed without details")
                    failing_agents[name] = reason

            if failing_agents and len(failing_agents) == len(agent_results):
                unique_reasons = list(dict.fromkeys([r for r in failing_agents.values() if r]))
                reason_summary = " | ".join(unique_reasons[:3]) if unique_reasons else "no recoverable evidence was produced"
                return (
                    "Task could not be completed because all executed agents failed to produce usable evidence "
                    f"({', '.join(sorted(failing_agents.keys()))}). Primary causes: {reason_summary}."
                )

        messages = [
            {
                "role": "system",
                "content": (
                    self._get_system_prompt()
                    + "\n\nSynthesize the FINAL answer using only provided evidence. "
                    + "Return strict JSON: {\"final_answer\":\"...\"}. "
                    + "Do not include markdown. Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "objective": objective,
                        "agent_results": packed_results,
                        "merged_state": merged_state,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        resp = self._call_llm(messages, tools=[])
        raw = self._strip_thinking_tokens(str(resp.get("content", "")))
        parsed = self._extract_json(raw)
        if isinstance(parsed, dict):
            for key in ("final_answer", "ans", "answer", "response"):
                val = parsed.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()

        if raw.strip():
            return raw.strip()

        for key in ("ans", "answer", "final_answer", "response"):
            val = merged_state.get(key) if isinstance(merged_state, dict) else None
            if val is not None and str(val).strip():
                return str(val).strip()

        return _compact(merged_state)

    def _flatten_output_keys(self, value: Any, prefix: str = "") -> set[str]:
        keys: set[str] = set()
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                keys.add(key)
                keys.update(self._flatten_output_keys(v, key))
        elif isinstance(value, list):
            for idx, item in enumerate(value[:5]):
                key = f"{prefix}[{idx}]"
                keys.update(self._flatten_output_keys(item, key))
        return keys

    def _has_overlap(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        ka = {k.split(".")[0] for k in self._flatten_output_keys(a)}
        kb = {k.split(".")[0] for k in self._flatten_output_keys(b)}
        shared = (ka & kb) - {"objective", "scene_id", "region", "raw_output", "thought", "actions"}
        return len(shared) > 0

    def aggregate(self, results: list[AgentResult]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for r in results:
            if hasattr(r, "output") and isinstance(r.output, dict):
                merged.update(r.output)

        if len(results) >= 2:
            out_a = results[0].output if isinstance(results[0].output, dict) else {}
            out_b = results[1].output if isinstance(results[1].output, dict) else {}
            if self._has_overlap(out_a, out_b):
                best = resolve_conflict(results[0].__dict__, results[1].__dict__)
                merged["selected_by_conflict"] = best.get("output", {})
                if isinstance(best.get("conflict_resolution"), dict):
                    merged["conflict_resolutions"] = best["conflict_resolution"]

        ok, missing = validate_safety_payload(merged)
        merged["safety_valid"] = ok
        merged["safety_missing"] = missing
        return merged
