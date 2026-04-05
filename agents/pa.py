from __future__ import annotations
import json
import math
from datetime import datetime
from typing import Any

from agents.observation_distance import extract_observation_distance_evidence, is_placeholder_entity_name
from agents.base_agent import AgentResult, BaseAgent, append_trace_line
from framework.mpc.message import Message

class PlanningAgent(BaseAgent):
    name = "pa"

    PA_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "EvacuationRoutePlanner",
                "description": "Compute optimal evacuation routes on a road graph, avoiding blocked segments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "graph_path": {"type": "string"},
                        "origins": {"type": "array", "items": {"type": "string"}},
                        "destinations": {"type": "array", "items": {"type": "string"}},
                        "blocked_segments": {"type": "array", "items": {"type": "string"}},
                        "mode": {"type": "string", "enum": ["evacuation", "supply", "conflict_check"]},
                    },
                    "required": ["graph_path", "origins", "destinations"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ComputeDistance",
                "description": "Compute geodesic distance between two points (accepts [lat, lon] or [lon, lat]) as a planning sanity check.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "point_a": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "point_b": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "required": ["point_a", "point_b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Calculator",
                "description": "Evaluate a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Solver",
                "description": "Solve symbolic or numeric equations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equation": {"type": "string"},
                    },
                    "required": ["equation"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Plot",
                "description": "Execute Python code to generate a matplotlib visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x_values": {"type": "array", "items": {"type": "number"}},
                        "y_values": {"type": "array", "items": {"type": "number"}},
                        "output_path": {"type": "string"},
                    },
                    "required": ["x_values", "y_values", "output_path"],
                },
            },
        },
    ]

    def _get_system_prompt(self) -> str:
        return (
            "You are PA, the Planning Agent in MAGRF. You compute routes, distances, "
            "and final decision-grade answers from upstream GA/VRA evidence.\n\n"
            "OUTPUT (mandatory, single JSON object, no markdown):\n"
            "{\"thought\":\"...\",\"actions\":[{\"name\":\"ToolName\",\"arguments\":{...}}]}\n"
            "Finalize: {\"thought\":\"...\",\"actions\":[{\"name\":\"Terminate\",\"arguments\":{\"ans\":\"<concrete answer>\"}}]}\n\n"
            "REACT LOOP: Think -> call one tool -> observe -> think -> repeat or Terminate.\n\n"
            "TOOLS: EvacuationRoutePlanner, ComputeDistance, Calculator, Solver, Plot.\n\n"
            "USE UPSTREAM CONTEXT FIRST:\n"
            "- If payload.episodic_context.ga_result contains closest_pair + distance_meters, use them directly and Terminate. Do not redo GA's work.\n"
            "- If VRA/GA gave bboxes, coordinates, POI names, counts, or index statistics, wire them into the answer.\n"
            "- If context has two candidate points as [lat,lon], call ComputeDistance(point_a, point_b) to verify.\n\n"
            "PLAYBOOKS:\n\n"
            "[A] PROXIMITY / DISTANCE:\n"
            "  - If GA gave distance_meters -> Terminate with '<Name1> and <Name2> at <N> m'.\n"
            "  - Else if pairs of coords exist -> call ComputeDistance on each pair, pick the min.\n"
            "  - Express distances with units ('m' or 'km') and one decimal when useful.\n\n"
            "[B] EVACUATION / SAFE ROUTE:\n"
            "  - EvacuationRoutePlanner(graph_path, origins, destinations, mode='evacuation', blocked_segments=<list>).\n"
            "  - Prefer the route with highest safety score. Surface blocked segments.\n\n"
            "[C] MATH / ARITHMETIC (from VRA measurements):\n"
            "  - Calculator(expression) for numeric derivations (pixel_distance * GSD, area = w*h, etc.).\n"
            "  - Solver(equation) for symbolic / algebraic needs.\n\n"
            "[D] COMPARATIVE PLOT: Plot(x_values, y_values, output_path).\n\n"
            "TERMINATE ANSWER FORMAT:\n"
            "- Concrete: names + number + unit (e.g. 'Central Fire Station and 3rd Precinct: 412.7 m').\n"
            "- Quote exact values from upstream evidence when available, do not round away precision.\n"
            "- Only say 'insufficient' if upstream evidence is truly empty AND you tried to reason with what exists.\n\n"
            "GROUND RULES:\n"
            "- Never fabricate coordinates, route graphs, or distances.\n"
            "- If a tool errors, try alternate arguments or a different tool once before conceding."
        )

    def _is_proximity_query(self, payload: dict[str, Any]) -> bool:
        query_type = str(payload.get("query_type", "")).lower()
        objective = str(payload.get("objective", "")).lower()
        # Pixel/image distance tasks are visual reasoning, not geospatial proximity.
        if query_type in {"visual_qa", "change_detection", "spectral_change"}:
            return False
        if any(k in objective for k in ("pixel", "m/pixel", "meters per pixel", "aerial image", "satellite image")):
            return False
        return (
            "proximity" in query_type
            or "distance" in query_type
            or any(k in objective for k in ("closest", "nearest", "distance", "proximity"))
        )

    def _extract_ga_evidence(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        ctx = payload.get("episodic_context")
        ga_result = ctx.get("ga_result") if isinstance(ctx, dict) else None
        ga_evidence: dict[str, Any] | None = None

        if isinstance(ga_result, dict):
            closest_pair = ga_result.get("closest_pair")
            distance_meters = ga_result.get("distance_meters")
            if isinstance(closest_pair, dict) and isinstance(distance_meters, (int, float)):
                d = float(distance_meters)
                if math.isfinite(d):
                    ga_evidence = {
                        "closest_pair": closest_pair,
                        "distance_meters": d,
                        "fire_pois_count": ga_result.get("fire_pois_count"),
                        "police_pois_count": ga_result.get("police_pois_count"),
                        "distance_source": (
                            str(
                                (
                                    ga_result.get("distance_evidence")
                                    if isinstance(ga_result.get("distance_evidence"), dict)
                                    else {}
                                ).get("source", "ga_result_shared_context")
                            )
                            or "ga_result_shared_context"
                        ),
                    }

        observation_evidence = extract_observation_distance_evidence(
            payload,
            raw_output=(ga_result.get("raw_output") if isinstance(ga_result, dict) and isinstance(ga_result.get("raw_output"), str) else None),
        )
        obs_closest = (
            observation_evidence.get("closest_pair")
            if isinstance(observation_evidence, dict) and isinstance(observation_evidence.get("closest_pair"), dict)
            else None
        )

        if ga_evidence is not None:
            cp = ga_evidence["closest_pair"]
            fire_name = str((cp.get("fire_station", {}) if isinstance(cp.get("fire_station"), dict) else {}).get("name", "")).strip()
            police_name = str((cp.get("police_station", {}) if isinstance(cp.get("police_station"), dict) else {}).get("name", "")).strip()
            has_placeholder_pair = is_placeholder_entity_name(fire_name) and is_placeholder_entity_name(police_name)
            degenerate = has_placeholder_pair and ga_evidence["distance_meters"] <= 0.0
            if not (degenerate and isinstance(obs_closest, dict)):
                return ga_evidence

        if isinstance(obs_closest, dict):
            obs_dist = float(obs_closest.get("distance_meters", 0.0))
            return {
                "closest_pair": {
                    "fire_station": {"name": str(obs_closest.get("entity_a", "entity_a"))},
                    "police_station": {"name": str(obs_closest.get("entity_b", "entity_b"))},
                    "distance_meters": obs_dist,
                },
                "distance_meters": obs_dist,
                "fire_pois_count": None,
                "police_pois_count": None,
                "distance_source": "observation_payload",
            }

        return None

    def _build_context_driven_result(self, message: Message, evidence: dict[str, Any]) -> AgentResult:
        closest_pair = evidence["closest_pair"]
        distance_meters = float(evidence["distance_meters"])

        fire_name = str(
            (closest_pair.get("fire_station", {}) if isinstance(closest_pair.get("fire_station"), dict) else {}).get(
                "name", "fire station"
            )
        )
        police_name = str(
            (
                closest_pair.get("police_station", {})
                if isinstance(closest_pair.get("police_station"), dict)
                else {}
            ).get("name", "police station")
        )

        thought = (
            "I have validated GA evidence from shared episodic context and can finalize the closest pair "
            "without fabricating additional coordinates."
        )
        ans = f"Closest pair: {fire_name} and {police_name} ({distance_meters:.2f} m)."
        actions = [{"name": "Terminate", "arguments": {"ans": ans}}]
        raw_output = json.dumps({"thought": thought, "actions": actions}, ensure_ascii=False)

        trace_step: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.name,
            "task_id": message.payload.get("scene_id", "unknown"),
            "turn": 1,
            "llm_latency_ms": 0.0,
            "assistant_raw_content": raw_output,
            "thought": thought,
            "actions_in_text": actions,
            "tool_calls": [],
            "terminated": True,
            "terminated_no_tool_call": True,
        }
        append_trace_line(trace_step)

        output = {
            "raw_output": raw_output,
            "thought": thought,
            "actions": actions,
            "ans": ans,
            "closest_pair": closest_pair,
            "distance_meters": distance_meters,
            "distance_evidence": {
                "source": str(evidence.get("distance_source", "ga_result_shared_context")),
                "fire_pois_count": evidence.get("fire_pois_count"),
                "police_pois_count": evidence.get("police_pois_count"),
            },
            "tool_call_count": 0,
        }
        for k in ("objective", "scene_id", "region"):
            if k in message.payload:
                output[k] = message.payload[k]

        return AgentResult(
            output=output,
            confidence=0.95,
            tool_calls=[],
            n_turns=1,
            trace=[trace_step],
        )

    def _has_coordinate_context(self, payload: dict[str, Any]) -> bool:
        ctx = payload.get("episodic_context")
        if not isinstance(ctx, dict):
            return False

        def _scan(value: Any) -> bool:
            if isinstance(value, dict):
                lat = value.get("lat")
                lon = value.get("lon")
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    return True
                pa = value.get("point_a")
                pb = value.get("point_b")
                if isinstance(pa, list) and len(pa) >= 2 and isinstance(pb, list) and len(pb) >= 2:
                    return True
                for child in value.values():
                    if _scan(child):
                        return True
            elif isinstance(value, list):
                for child in value:
                    if _scan(child):
                        return True
            return False

        return _scan(ctx)

    def _build_insufficient_data_result(self, message: Message) -> AgentResult:
        thought = (
            "This is a proximity query, but there is no trustworthy upstream coordinate evidence. "
            "Terminating instead of fabricating coordinates."
        )
        ans = "Unable to determine - insufficient geospatial data from upstream agents."
        actions = [{"name": "Terminate", "arguments": {"ans": ans}}]
        raw_output = json.dumps({"thought": thought, "actions": actions}, ensure_ascii=False)

        trace_step: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.name,
            "task_id": message.payload.get("scene_id", "unknown"),
            "turn": 1,
            "llm_latency_ms": 0.0,
            "assistant_raw_content": raw_output,
            "thought": thought,
            "actions_in_text": actions,
            "tool_calls": [],
            "terminated": True,
            "terminated_no_tool_call": True,
        }
        append_trace_line(trace_step)

        output = {
            "raw_output": raw_output,
            "thought": thought,
            "actions": actions,
            "ans": ans,
            "confidence": 0.1,
            "tool_call_count": 0,
        }
        for k in ("objective", "scene_id", "region"):
            if k in message.payload:
                output[k] = message.payload[k]

        return AgentResult(
            output=output,
            confidence=0.1,
            tool_calls=[],
            n_turns=1,
            trace=[trace_step],
        )

    def _augment_output_from_tools(
        self,
        output_dict: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        # Prefer GA evidence from shared context for proximity queries.
        ga_evidence = self._extract_ga_evidence(payload)
        if ga_evidence is not None:
            output_dict["closest_pair"] = ga_evidence["closest_pair"]
            output_dict["distance_meters"] = float(ga_evidence["distance_meters"])
            output_dict["distance_evidence"] = {
                "source": str(ga_evidence.get("distance_source", "ga_result_shared_context")),
                "fire_pois_count": ga_evidence.get("fire_pois_count"),
                "police_pois_count": ga_evidence.get("police_pois_count"),
            }
            if not isinstance(output_dict.get("ans"), str) or not output_dict.get("ans", "").strip():
                cp = ga_evidence["closest_pair"]
                fire_name = str(
                    (cp.get("fire_station", {}) if isinstance(cp.get("fire_station"), dict) else {}).get(
                        "name", "fire station"
                    )
                )
                police_name = str(
                    (cp.get("police_station", {}) if isinstance(cp.get("police_station"), dict) else {}).get(
                        "name", "police station"
                    )
                )
                output_dict["ans"] = (
                    f"Closest pair: {fire_name} and {police_name} ({float(ga_evidence['distance_meters']):.2f} m)."
                )
            return output_dict

        # Fallback: synthesize nearest pair from PA ComputeDistance calls.
        observations: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict) or str(call.get("tool")) != "ComputeDistance":
                continue
            args = call.get("args", {}) if isinstance(call.get("args"), dict) else {}
            out = call.get("output", {}) if isinstance(call.get("output"), dict) else {}
            pa = args.get("point_a") if isinstance(args.get("point_a"), list) else None
            pb = args.get("point_b") if isinstance(args.get("point_b"), list) else None
            dist = out.get("distance_meters")
            if (
                isinstance(pa, list)
                and isinstance(pb, list)
                and len(pa) >= 2
                and len(pb) >= 2
                and isinstance(dist, (int, float))
            ):
                if float(pa[0]) == float(pb[0]) and float(pa[1]) == float(pb[1]) and float(dist) == 0.0:
                    output_dict.setdefault("degenerate_computation", True)
                    continue
                observations.append(
                    {
                        "point_a": [float(pa[0]), float(pa[1])],
                        "point_b": [float(pb[0]), float(pb[1])],
                        "distance_meters": float(dist),
                    }
                )

        if observations:
            best = min(observations, key=lambda x: x["distance_meters"])
            output_dict.setdefault("distance_observations", observations[:20])
            output_dict["distance_meters"] = best["distance_meters"]
            output_dict.setdefault(
                "closest_pair",
                {
                    "fire_station": {"name": "candidate_a", "lat": best["point_a"][0], "lon": best["point_a"][1]},
                    "police_station": {"name": "candidate_b", "lat": best["point_b"][0], "lon": best["point_b"][1]},
                    "distance_meters": best["distance_meters"],
                },
            )
            output_dict.setdefault(
                "distance_evidence",
                {"source": "pa_compute_distance_calls", "distance_measurements": len(observations)},
            )

        return output_dict

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.PA_TOOLS

        ga_evidence = self._extract_ga_evidence(message.payload)
        if self._is_proximity_query(message.payload) and ga_evidence is not None:
            return self._build_context_driven_result(message, ga_evidence)

        # Do NOT short-circuit when coordinate context is missing; let the LLM
        # try to reason with GA's partial output (landmarks, bbox, POI names).
        # The base ReAct loop will still gracefully terminate if truly stuck.
        return super().handle_task(message, tools_schema)
