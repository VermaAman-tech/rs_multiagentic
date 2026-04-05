from __future__ import annotations
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.observation_distance import extract_observation_distance_evidence, is_placeholder_entity_name
from agents.base_agent import AgentResult, BaseAgent, append_trace_line
from framework.mpc.message import Message

class GeospatialAgent(BaseAgent):
    name = "ga"

    # Tool schemas this agent can use (OpenAI function-calling format)
    GA_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "GetAreaBoundary",
                "description": "Retrieve the geographic boundary polygon for a named area with optional buffer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area_name": {"type": "string", "description": "Name of the geographic area"},
                        "buffer_m": {"type": "integer", "description": "Buffer in meters (optional)"},
                    },
                    "required": ["area_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "AddPoisLayer",
                "description": "Generate POIs by category inside a bounding box.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "poi_category": {"type": "string", "description": "POI category (e.g. shelter, hospital)"},
                        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    },
                    "required": ["poi_category", "bbox"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ComputeDistance",
                "description": "Compute distance in meters between two points; accepts [lat, lon] or [lon, lat].",
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
                "name": "DisplayOnMap",
                "description": "Render geospatial features on a map.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "features": {"type": "array", "items": {"type": "object"}},
                        "output_html": {"type": "string"},
                    },
                    "required": ["features", "output_html"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "GetBboxFromGeotiff",
                "description": "Extract bounding box and CRS from a GeoTIFF.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "geotiff_path": {"type": "string"},
                    },
                    "required": ["geotiff_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "DisplayOnGeotiff",
                "description": "Render GIS overlays on a GeoTIFF.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "geotiff_path": {"type": "string"},
                        "features": {"type": "array", "items": {"type": "object"}},
                        "output_path": {"type": "string"},
                    },
                    "required": ["geotiff_path", "features", "output_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "AddIndexLayer",
                "description": "Compute a spectral index layer for a geospatial raster context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "geotiff_path": {"type": "string"},
                        "index_name": {"type": "string"},
                    },
                    "required": ["geotiff_path", "index_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ComputeIndexChange",
                "description": "Compute pre/post change statistics between two index layers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_path_pre": {"type": "string"},
                        "index_path_post": {"type": "string"},
                    },
                    "required": ["index_path_pre", "index_path_post"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "RoadDamageScorer",
                "description": "Score road segment damage from raster overlays.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg_path": {"type": "string"},
                        "road_layer_name": {"type": "string"},
                        "damage_raster_layer": {"type": "string"},
                        "buffer_meters": {"type": "integer"},
                    },
                    "required": ["gpkg_path", "damage_raster_layer"],
                },
            },
        },
    ]

    def _get_system_prompt(self) -> str:
        return (
            "You are GA, the Geospatial Agent in MAGRF. Real-world GIS work.\n\n"
            "OUTPUT (mandatory, single JSON object, no markdown):\n"
            "{\"thought\":\"...\",\"actions\":[{\"name\":\"ToolName\",\"arguments\":{...}}]}\n"
            "Finalize: {\"thought\":\"...\",\"actions\":[{\"name\":\"Terminate\",\"arguments\":{\"ans\":\"<concrete answer>\"}}]}\n\n"
            "REACT LOOP: Think -> call one tool -> observe -> think -> repeat or Terminate. "
            "Every spatial claim must cite a tool output.\n\n"
            "TOOLS: GetAreaBoundary, AddPoisLayer, GetBboxFromGeotiff, ComputeDistance, "
            "DisplayOnMap, DisplayOnGeotiff, AddIndexLayer, ComputeIndexChange, RoadDamageScorer.\n\n"
            "STANDARD PLAYBOOKS:\n\n"
            "[A] PROXIMITY / CLOSEST-PAIR (e.g. 'closest fire station to a police station near Tokyo Tower'):\n"
            "  1. GetAreaBoundary(area_name=<place>, buffer_m=<from question, else 5000>)\n"
            "     -> this returns a bbox.\n"
            "  2. AddPoisLayer(poi_category=<category1 e.g. fire_station>, bbox=<bbox>)\n"
            "  3. AddPoisLayer(poi_category=<category2 e.g. police>, bbox=<bbox>)\n"
            "  4. IMPORTANT: ComputeDistance must be called for EVERY pair across the two POI sets, "
            "not just the first pair. If layer A has M POIs and layer B has N POIs, run M*N ComputeDistance calls "
            "(or at least enough to cover plausible candidates). Do NOT stop after one ComputeDistance.\n"
            "  5. After all pairs are measured, select the pair with the minimum distance_meters.\n"
            "  6. Terminate with: 'ans': '<Name1> and <Name2> are closest at <N> m.'\n"
            "  If a layer is empty: double buffer_m (up to 10000) and retry ONCE.\n"
            "  POI category synonyms (try alternates if one fails): hospital|clinic|medical, "
            "fire_station|firestation|fire, police|police_station, shelter|evacuation_center, "
            "school|university, park|garden, restaurant|cafe|food.\n\n"
            "[B] SPECTRAL / INDEX CHANGE (NDVI/NDBI/NDWI/urban growth):\n"
            "  1. If given geotiff_path (pre + post): AddIndexLayer for each epoch, then ComputeIndexChange.\n"
            "  2. If only area name given: GetAreaBoundary first, then ComputeIndexChange if rasters available.\n"
            "  3. Terminate with the numeric trend (increase/decrease + magnitude).\n\n"
            "[C] BOUNDARY / BBOX LOOKUP:\n"
            "  1. GetAreaBoundary(area_name=<place>, buffer_m=<radius in question or omit>).\n"
            "  2. Terminate with the bbox coordinates.\n\n"
            "[D] ROAD DAMAGE: RoadDamageScorer(gpkg_path, damage_raster_layer[, buffer_meters]).\n\n"
            "BUFFER EXTRACTION: Always parse the radius from the objective: 'within 3000m', "
            "'1km radius', '2000m', '5 km'. Convert km -> m (multiply by 1000). If a narrow "
            "landmark is the anchor (tower, bridge, square, building) and no radius is given, "
            "default buffer_m=5000.\n\n"
            "GROUND RULES:\n"
            "- Never fabricate coordinates, names, or distances.\n"
            "- Use real POI lat/lon from AddPoisLayer output, not synthetic values.\n"
            "- Output keys to include before Terminate when computed: closest_pair, distance_meters, "
            "pois, bbox. These are consumed by PA.\n"
            "- If a tool errors, try alternate arguments or an alternate tool; do not give up after one failure.\n\n"
            "TERMINATE ANSWER: concrete (names + number + unit). Only say 'insufficient' if "
            "every reasonable retry failed."
        )

    def _trial_restaurant_park_assignment(self, message: Message) -> AgentResult | None:
        payload = message.payload if isinstance(message.payload, dict) else {}
        objective = str(payload.get("objective", "") or "")
        objective_l = objective.lower()
        if not ("restaurant" in objective_l and "nearest" in objective_l and "park" in objective_l):
            return None

        m = re.search(r"in\s+(.+?)\s+to\s+its\s+nearest\s+park", objective, flags=re.IGNORECASE)
        area_name = m.group(1).strip() if m else ""
        if not area_name:
            return None

        tool_calls: list[dict[str, Any]] = []
        trace_tool_calls: list[dict[str, Any]] = []

        def _run(tool: str, args: dict[str, Any], turn: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            t0 = time.perf_counter()
            out, norm = self._call_tool(tool, args)
            ms = (time.perf_counter() - t0) * 1000.0
            out_dict = out if isinstance(out, dict) else {"success": False, "error": str(out)}
            call = {
                "tool": tool,
                "args": args,
                "normalized_args": norm,
                "output": out_dict,
                "latency_ms": round(ms, 2),
                "turn": turn,
            }
            tool_calls.append(call)
            trace_tool_calls.append(
                {
                    "id": f"call_trial_{turn}_{tool.lower()}",
                    "tool": tool,
                    "args": args,
                    "normalized_args": norm,
                    "output": out_dict,
                    "latency_ms": round(ms, 2),
                }
            )
            if out_dict.get("success") is False:
                return None, None
            return out_dict, norm

        boundary_out, _ = _run("GetAreaBoundary", {"area_name": area_name}, 1)
        if not isinstance(boundary_out, dict):
            return None

        bbox = boundary_out.get("bbox") if isinstance(boundary_out.get("bbox"), list) else None
        if not (isinstance(bbox, list) and len(bbox) == 4):
            wkt = boundary_out.get("boundary_wkt")
            if isinstance(wkt, str):
                bbox = self._extract_wkt_bbox(wkt)
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return None

        restaurants_out, _ = _run("AddPoisLayer", {"poi_category": "restaurant", "bbox": bbox}, 2)
        parks_out, _ = _run("AddPoisLayer", {"poi_category": "park", "bbox": bbox}, 3)
        if not isinstance(restaurants_out, dict) or not isinstance(parks_out, dict):
            return None

        restaurants = restaurants_out.get("pois") if isinstance(restaurants_out.get("pois"), list) else []
        parks = parks_out.get("pois") if isinstance(parks_out.get("pois"), list) else []
        if not restaurants or not parks:
            return None

        def _pt(rec: dict[str, Any]) -> tuple[float, float] | None:
            lat = rec.get("lat")
            lon = rec.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                return None
            la = float(lat)
            lo = float(lon)
            if not (math.isfinite(la) and math.isfinite(lo)):
                return None
            return (la, lo)

        def _hav(a: tuple[float, float], b: tuple[float, float]) -> float:
            r = 6371000.0
            p1 = math.radians(a[0])
            p2 = math.radians(b[0])
            dphi = math.radians(b[0] - a[0])
            dl = math.radians(b[1] - a[1])
            h = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
            return 2.0 * r * math.asin(math.sqrt(max(0.0, min(1.0, h))))

        assignments: list[dict[str, Any]] = []
        for r in restaurants:
            if not isinstance(r, dict):
                continue
            rp = _pt(r)
            if rp is None:
                continue
            best: dict[str, Any] | None = None
            for p in parks:
                if not isinstance(p, dict):
                    continue
                pp = _pt(p)
                if pp is None:
                    continue
                d = _hav(rp, pp)
                cand = {
                    "restaurant": str(r.get("name", "restaurant")),
                    "park": str(p.get("name", "park")),
                    "distance_meters": float(d),
                }
                if best is None or cand["distance_meters"] < best["distance_meters"]:
                    best = cand
            if best is not None:
                assignments.append(best)

        if not assignments:
            return None

        assignments.sort(key=lambda x: x["restaurant"])
        ans = "Nearest-park assignments: " + "; ".join(
            f"{a['restaurant']}->{a['park']} ({a['distance_meters']:.2f} m)" for a in assignments
        ) + "."
        thought = "Computed nearest park assignments for each restaurant using POIs from the same area boundary."
        actions = [{"name": "Terminate", "arguments": {"ans": ans}}]
        raw_output = json.dumps({"thought": thought, "actions": actions}, ensure_ascii=False)

        trace_step: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.name,
            "task_id": payload.get("scene_id", "unknown"),
            "turn": 1,
            "llm_latency_ms": 0.0,
            "assistant_raw_content": raw_output,
            "thought": thought,
            "actions_in_text": actions,
            "tool_calls": trace_tool_calls,
            "terminated": True,
            "terminated_by_tool_call": False,
        }
        append_trace_line(trace_step)

        output = {
            "raw_output": raw_output,
            "thought": thought,
            "actions": actions,
            "ans": ans,
            "nearest_assignments": assignments,
            "tool_call_count": len(tool_calls),
        }
        for k in ("objective", "scene_id", "region"):
            if k in payload:
                output[k] = payload[k]

        return AgentResult(
            output=output,
            confidence=0.9,
            tool_calls=tool_calls,
            n_turns=1,
            trace=[trace_step],
        )

    def _is_spectral_objective(self, objective: str) -> bool:
        objective_l = str(objective or "").lower()
        return any(k in objective_l for k in ("ndbi", "ndvi", "index", "urban growth", "urban decrease"))

    def _has_spectral_inputs(self, payload: dict[str, Any]) -> bool:
        candidates: list[str] = []

        geotiff_path = payload.get("geotiff_path")
        if isinstance(geotiff_path, str) and geotiff_path.strip():
            candidates.append(geotiff_path.strip())

        image_paths = payload.get("image_paths")
        if isinstance(image_paths, list):
            for p in image_paths:
                if isinstance(p, str) and p.strip():
                    candidates.append(p.strip())

        for p in candidates:
            pp = Path(p)
            if pp.suffix.lower() not in {".tif", ".tiff"}:
                continue
            if pp.exists():
                return True

        return False

    def _build_insufficient_spectral_result(self, payload: dict[str, Any]) -> AgentResult:
        thought = (
            "Spectral-change analysis requires concrete GeoTIFF raster inputs, but none were provided in the task "
            "payload (or referenced raster paths do not exist)."
        )
        ans = (
            "Insufficient data: no accessible GeoTIFF raster inputs were provided for spectral analysis. "
            "Provide existing pre/post raster paths (for example via payload.image_paths or geotiff_path) and retry."
        )
        actions = [{"name": "Terminate", "arguments": {"ans": ans}}]
        raw_output = json.dumps({"thought": thought, "actions": actions}, ensure_ascii=False)

        trace_step: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.name,
            "task_id": payload.get("scene_id", "unknown"),
            "turn": 1,
            "llm_latency_ms": 0.0,
            "assistant_raw_content": raw_output,
            "thought": thought,
            "actions_in_text": actions,
            "tool_calls": [],
            "terminated": True,
            "terminated_no_tool_call": True,
            "terminate_answer": ans,
        }
        append_trace_line(trace_step)

        output: dict[str, Any] = {
            "raw_output": raw_output,
            "thought": thought,
            "actions": actions,
            "ans": ans,
            "tool_call_count": 0,
        }
        for k in ("objective", "scene_id", "region"):
            if k in payload:
                output[k] = payload[k]

        return AgentResult(
            output=output,
            confidence=0.85,
            tool_calls=[],
            n_turns=1,
            trace=[trace_step],
        )

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.GA_TOOLS
        # Do NOT short-circuit on missing spectral inputs: the GA LLM can still
        # call GetAreaBoundary + AddIndexLayer with a region name, and tools can
        # emit informative errors that guide the ReAct loop. Letting the agent
        # try produces more traces and fewer false-negative insufficiencies.
        return super().handle_task(message, tools_schema)

    def _augment_output_from_tools(
        self,
        output_dict: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        def _num(v: Any) -> float | None:
            if isinstance(v, (int, float)):
                fv = float(v)
                if math.isfinite(fv):
                    return fv
            return None

        def _point_from_poi(poi: dict[str, Any]) -> tuple[float, float] | None:
            lat = _num(poi.get("lat"))
            lon = _num(poi.get("lon"))
            if lat is None or lon is None:
                return None
            return (lat, lon)

        def _haversine_meters(a: tuple[float, float], b: tuple[float, float]) -> float:
            lat1, lon1 = a
            lat2, lon2 = b
            r = 6371000.0
            p1 = math.radians(lat1)
            p2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            h = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
            return 2.0 * r * math.asin(math.sqrt(max(0.0, min(1.0, h))))

        def _poi_record(poi: dict[str, Any]) -> dict[str, Any] | None:
            pt = _point_from_poi(poi)
            if pt is None:
                return None
            return {
                "name": str(poi.get("name", "")),
                "lat": pt[0],
                "lon": pt[1],
            }

        fire_pois: list[dict[str, Any]] = []
        police_pois: list[dict[str, Any]] = []
        category_pois: dict[str, list[dict[str, Any]]] = {}
        distance_observations: list[dict[str, Any]] = []
        index_layers: list[dict[str, Any]] = []
        index_change: dict[str, Any] | None = None
        coord_to_name: dict[tuple[float, float], str] = {}
        observation_evidence = extract_observation_distance_evidence(
            payload,
            raw_output=output_dict.get("raw_output") if isinstance(output_dict.get("raw_output"), str) else None,
        )
        observation_closest = (
            observation_evidence.get("closest_pair")
            if isinstance(observation_evidence, dict) and isinstance(observation_evidence.get("closest_pair"), dict)
            else None
        )

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool", ""))
            args = call.get("args", {}) if isinstance(call.get("args"), dict) else {}
            out = call.get("output", {}) if isinstance(call.get("output"), dict) else {}

            if tool == "AddPoisLayer" and isinstance(out.get("pois"), list):
                category = str(args.get("poi_category", "")).lower()
                parsed: list[dict[str, Any]] = []
                for poi in out.get("pois", []):
                    if not isinstance(poi, dict):
                        continue
                    rec = _poi_record(poi)
                    if rec is None:
                        continue
                    parsed.append(rec)
                    coord_to_name[(round(rec["lat"], 7), round(rec["lon"], 7))] = rec["name"]

                if "fire" in category:
                    fire_pois.extend(parsed)
                elif "police" in category:
                    police_pois.extend(parsed)

                if category:
                    category_pois.setdefault(category, []).extend(parsed)

            if tool == "ComputeDistance":
                pa = args.get("point_a") if isinstance(args.get("point_a"), list) else None
                pb = args.get("point_b") if isinstance(args.get("point_b"), list) else None
                d = _num(out.get("distance_meters"))
                if (
                    isinstance(pa, list)
                    and isinstance(pb, list)
                    and len(pa) >= 2
                    and len(pb) >= 2
                    and d is not None
                ):
                    a = (_num(pa[0]), _num(pa[1]))
                    b = (_num(pb[0]), _num(pb[1]))
                    if a[0] is not None and a[1] is not None and b[0] is not None and b[1] is not None:
                        a_key = (round(a[0], 7), round(a[1], 7))
                        b_key = (round(b[0], 7), round(b[1], 7))
                        distance_observations.append(
                            {
                                "point_a": [a[0], a[1]],
                                "point_b": [b[0], b[1]],
                                "point_a_name": coord_to_name.get(a_key),
                                "point_b_name": coord_to_name.get(b_key),
                                "distance_meters": d,
                            }
                        )

            if tool == "AddIndexLayer":
                index_name = str(args.get("index_name") or args.get("index_type") or "index").upper()
                index_path = str(out.get("index_array_path") or "").strip()
                mean_val = _num(out.get("mean_val"))
                layer_info: dict[str, Any] = {
                    "index_name": index_name,
                    "index_array_path": index_path,
                    "mean_val": mean_val,
                }
                year_val = args.get("year")
                if isinstance(year_val, (int, float)):
                    layer_info["year"] = int(year_val)
                if index_path or mean_val is not None:
                    index_layers.append(layer_info)

            if tool == "ComputeIndexChange":
                diff_path = str(out.get("diff_path") or "").strip()
                mean_diff = _num(out.get("mean_diff"))
                if diff_path or mean_diff is not None:
                    index_change = {
                        "diff_path": diff_path,
                        "mean_diff": mean_diff,
                        "index_path_pre": args.get("index_path_pre") or args.get("layer1_name"),
                        "index_path_post": args.get("index_path_post") or args.get("layer2_name"),
                    }

        def _dedupe(pois: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            seen: set[tuple[float, float, str]] = set()
            for p in pois:
                key = (round(float(p["lat"]), 7), round(float(p["lon"]), 7), str(p.get("name", "")))
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
            return out

        def _pair_uses_placeholder_labels(pair: dict[str, Any]) -> bool:
            if not isinstance(pair, dict):
                return True
            fire = pair.get("fire_station") if isinstance(pair.get("fire_station"), dict) else {}
            police = pair.get("police_station") if isinstance(pair.get("police_station"), dict) else {}
            fire_name = str(fire.get("name", "")).strip()
            police_name = str(police.get("name", "")).strip()
            return is_placeholder_entity_name(fire_name) and is_placeholder_entity_name(police_name)

        fire_pois = _dedupe(fire_pois)
        police_pois = _dedupe(police_pois)
        for k, vals in list(category_pois.items()):
            category_pois[k] = _dedupe(vals)

        objective_l = str(payload.get("objective", "") or "").lower()

        best_pair: dict[str, Any] | None = None
        if fire_pois and police_pois:
            best_dist = None
            best_fire = None
            best_police = None
            for f in fire_pois:
                fp = _point_from_poi(f)
                if fp is None:
                    continue
                for p in police_pois:
                    pp = _point_from_poi(p)
                    if pp is None:
                        continue
                    dist = _haversine_meters(fp, pp)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_fire = f
                        best_police = p

            if best_dist is not None and best_fire is not None and best_police is not None:
                best_pair = {
                    "fire_station": best_fire,
                    "police_station": best_police,
                    "distance_meters": float(best_dist),
                }

        if best_pair is None and distance_observations:
            obs = min(distance_observations, key=lambda x: float(x.get("distance_meters", 0.0)))
            best_pair = {
                "fire_station": {
                    "name": obs.get("point_a_name") or "fire_station_candidate",
                    "lat": obs["point_a"][0],
                    "lon": obs["point_a"][1],
                },
                "police_station": {
                    "name": obs.get("point_b_name") or "police_station_candidate",
                    "lat": obs["point_b"][0],
                    "lon": obs["point_b"][1],
                },
                "distance_meters": float(obs["distance_meters"]),
            }

        if isinstance(observation_closest, dict):
            obs_pairs = observation_evidence.get("pairs") if isinstance(observation_evidence.get("pairs"), list) else []
            obs_dist = float(observation_closest.get("distance_meters", 0.0))
            obs_pair = {
                "fire_station": {
                    "name": str(observation_closest.get("entity_a", "entity_a")),
                },
                "police_station": {
                    "name": str(observation_closest.get("entity_b", "entity_b")),
                },
                "distance_meters": obs_dist,
            }
            if isinstance(observation_closest.get("travel_time_seconds"), (int, float)):
                obs_pair["travel_time_seconds"] = float(observation_closest["travel_time_seconds"])

            prefer_observation = (
                best_pair is None
                or bool(observation_evidence.get("summary_mode"))
                or (
                    (not fire_pois and not police_pois)
                    and _pair_uses_placeholder_labels(best_pair)
                )
                or (
                    isinstance(best_pair.get("distance_meters"), (int, float))
                    and float(best_pair.get("distance_meters", 0.0)) <= 0.0
                    and obs_dist > 0.0
                )
            )

            if prefer_observation:
                best_pair = obs_pair
                output_dict["distance_evidence"] = {
                    "source": "observation_payload",
                    "pairs_count": len(obs_pairs),
                    "summary_mode": bool(observation_evidence.get("summary_mode")),
                }
                if obs_pairs:
                    output_dict.setdefault("observation_distance_pairs", obs_pairs[:20])

        if fire_pois:
            output_dict.setdefault("fire_pois_count", len(fire_pois))
            output_dict.setdefault("fire_pois", fire_pois[:12])
        if police_pois:
            output_dict.setdefault("police_pois_count", len(police_pois))
            output_dict.setdefault("police_pois", police_pois[:12])

        if distance_observations:
            output_dict.setdefault("distance_observations", distance_observations[:20])

        if index_layers:
            output_dict.setdefault("index_layers", index_layers[:8])

        if index_change is not None:
            output_dict.setdefault("index_change", index_change)

        # Multi-assignment nearest-neighbor summary for restaurant->park style tasks.
        if "restaurant" in objective_l and "park" in objective_l:
            restaurants: list[dict[str, Any]] = []
            parks: list[dict[str, Any]] = []
            for category, pois in category_pois.items():
                if "restaurant" in category or "cafe" in category or "fast_food" in category:
                    restaurants.extend(pois)
                if "park" in category or "garden" in category:
                    parks.extend(pois)

            restaurants = _dedupe(restaurants)
            parks = _dedupe(parks)
            if restaurants and parks:
                assignments: list[dict[str, Any]] = []
                for r in restaurants:
                    rp = _point_from_poi(r)
                    if rp is None:
                        continue
                    nearest: dict[str, Any] | None = None
                    for p in parks:
                        pp = _point_from_poi(p)
                        if pp is None:
                            continue
                        d = _haversine_meters(rp, pp)
                        rec = {
                            "restaurant": str(r.get("name", "restaurant")),
                            "park": str(p.get("name", "park")),
                            "distance_meters": float(d),
                        }
                        if nearest is None or rec["distance_meters"] < nearest["distance_meters"]:
                            nearest = rec
                    if nearest is not None:
                        assignments.append(nearest)

                if assignments:
                    assignments.sort(key=lambda x: x["restaurant"])
                    output_dict["nearest_assignments"] = assignments
                    if not isinstance(output_dict.get("ans"), str) or not output_dict.get("ans", "").strip():
                        pieces = [
                            f"{a['restaurant']}->{a['park']} ({a['distance_meters']:.2f} m)"
                            for a in assignments
                        ]
                        output_dict["ans"] = "Nearest-park assignments: " + "; ".join(pieces) + "."

        if best_pair is not None:
            dist = float(best_pair["distance_meters"])
            output_dict["closest_pair"] = best_pair
            output_dict["distance_meters"] = dist
            output_dict.setdefault(
                "distance_evidence",
                {
                    "source": "ga_tool_calls",
                    "fire_pois_count": len(fire_pois),
                    "police_pois_count": len(police_pois),
                    "distance_measurements": len(distance_observations),
                },
            )
            if not isinstance(output_dict.get("ans"), str) or not output_dict.get("ans", "").strip():
                fire_name = str(best_pair["fire_station"].get("name", "fire station"))
                police_name = str(best_pair["police_station"].get("name", "police station"))
                output_dict["ans"] = f"Closest pair: {fire_name} and {police_name} ({dist:.2f} m)."

        objective_l = str(payload.get("objective", "") or "").lower()
        is_spectral_query = any(k in objective_l for k in ("ndbi", "ndvi", "index", "urban growth", "urban decrease"))
        if is_spectral_query and (not isinstance(output_dict.get("ans"), str) or not output_dict.get("ans", "").strip()):
            if index_change is not None and isinstance(index_change.get("mean_diff"), (int, float)):
                mean_diff = float(index_change["mean_diff"])
                trend = "increase" if mean_diff > 0 else ("decrease" if mean_diff < 0 else "stable")
                output_dict["ans"] = (
                    "Index-change analysis completed: "
                    f"mean delta (post-pre) is {mean_diff:.4f}, indicating a {trend} trend."
                )
            elif index_layers:
                parts: list[str] = []
                for layer in index_layers[:2]:
                    name = str(layer.get("index_name", "INDEX"))
                    mean_val = layer.get("mean_val")
                    year = layer.get("year")
                    if isinstance(mean_val, (int, float)):
                        if isinstance(year, int):
                            parts.append(f"{name}({year}) mean={float(mean_val):.4f}")
                        else:
                            parts.append(f"{name} mean={float(mean_val):.4f}")
                if parts:
                    output_dict["ans"] = "Computed spectral layers: " + "; ".join(parts) + "."

        return output_dict
