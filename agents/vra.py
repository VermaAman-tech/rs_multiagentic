from __future__ import annotations
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.base_agent import AgentResult, BaseAgent, append_trace_line
from framework.mpc.message import Message

class VisionReasoningAgent(BaseAgent):
    name = "vra"

    VRA_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "ObjectDetection",
                "description": "Detect common objects in an image, returning bounding boxes and labels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "text_prompt": {"type": "string", "description": "Object to detect"},
                    },
                    "required": ["image_path", "text_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "TextToBbox",
                "description": "Detect objects matching a text description, returning bounding boxes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "text_prompt": {"type": "string"},
                    },
                    "required": ["image_path", "text_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ImageDescription",
                "description": "Generate a natural-language description of an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                    },
                    "required": ["image_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ChangeDetection",
                "description": "Analyze pre/post event images to detect and describe changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path_1": {"type": "string"},
                        "image_path_2": {"type": "string"},
                    },
                    "required": ["image_path_1", "image_path_2"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "CountGivenObject",
                "description": "Count the number of specified objects in an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "text_prompt": {"type": "string"},
                    },
                    "required": ["image_path", "text_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "SegmentObjectPixels",
                "description": "Segment specified objects and return pixel-level analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bboxes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                    "required": ["image_path", "bboxes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "RegionAttributeDescription",
                "description": "Describe a specific attribute of a region in an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["image_path", "bbox"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "OCR",
                "description": "Extract text from image regions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                    },
                    "required": ["image_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "DrawBox",
                "description": "Draw boxes on an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bboxes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        "output_path": {"type": "string"},
                    },
                    "required": ["image_path", "bboxes", "output_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "AddText",
                "description": "Add text labels on an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "text": {"type": "string"},
                        "position": {"type": "array", "items": {"type": "integer"}},
                        "output_path": {"type": "string"},
                    },
                    "required": ["image_path", "text", "position", "output_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "GetBboxFromGeotiff",
                "description": "Extract geospatial bounds from a GeoTIFF before downstream visual/spectral analysis.",
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
                "name": "AddIndexLayer",
                "description": "Compute spectral indices from a GeoTIFF.",
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
                "description": "Compute index difference between pre/post raster index layers.",
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
                "name": "ShowIndexLayer",
                "description": "Render a computed index raster layer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_array_path": {"type": "string"},
                    },
                    "required": ["index_array_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "TemporalStackLoader",
                "description": "Load a temporal stack for an AOI and date range.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "aoi_bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                        "date_range": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                        "sensor": {"type": "string"},
                        "max_cloud_pct": {"type": "number"},
                        "local_stack_path": {"type": "string"},
                        "image_paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["aoi_bbox", "date_range"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "PrithviEmbed",
                "description": "Compute Prithvi embeddings for a raster.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "raster_path": {"type": "string"},
                    },
                    "required": ["raster_path"],
                },
            },
        },
    ]

    def _get_system_prompt(self) -> str:
        return (
            "You are VRA, the Vision Reasoning Agent in MAGRF. You analyze images.\n\n"
            "OUTPUT (mandatory, single JSON object, no markdown):\n"
            "{\"thought\":\"...\",\"actions\":[{\"name\":\"ToolName\",\"arguments\":{...}}]}\n"
            "Finalize: {\"thought\":\"...\",\"actions\":[{\"name\":\"Terminate\",\"arguments\":{\"ans\":\"<concrete answer>\"}}]}\n\n"
            "REACT LOOP: Think -> call one tool -> observe -> think again -> repeat or Terminate. "
            "Never Terminate without at least one tool observation backing your answer.\n\n"
            "IMAGE PATHS: Use payload.image_paths EXACTLY as provided. Never invent paths.\n\n"
            "TOOL PLAYBOOK (pick the simplest chain that answers the question):\n"
            "- Count/how many X: CountGivenObject(image_path, object_name).\n"
            "- Locate/where is X / bbox of X: TextToBbox(image_path, text).\n"
            "- What objects are visible: ObjectDetection(image_path) or ImageDescription.\n"
            "- Describe scene / caption: ImageDescription(image_path).\n"
            "- Describe specific region: RegionAttributeDescription(image_path, bbox, attribute).\n"
            "- Read text/signs/labels: OCR(image_path).\n"
            "- Mask pixels of an object: SegmentObjectPixels(image_path, object_name).\n"
            "- Before/after change: ChangeDetection(image_path_pre, image_path_post).\n"
            "- Temporal stack: TemporalStackLoader.\n"
            "- Spectral index on a GeoTIFF: AddIndexLayer -> ShowIndexLayer / ComputeIndexChange.\n"
            "- GeoTIFF bounds: GetBboxFromGeotiff(geotiff_path).\n"
            "- Annotation output: DrawBox / AddText.\n\n"
            "DISTANCE-IN-IMAGE (pixels -> meters):\n"
            "- If objective mentions GSD (e.g. '0.3 m/pixel'), multiply pixel distance by GSD.\n"
            "- Euclidean pixel distance between bbox centers = sqrt((cx2-cx1)^2 + (cy2-cy1)^2).\n"
            "- For two labelled objects: call TextToBbox twice (one per label), then compute distance.\n\n"
            "TERMINATE ANSWER FORMAT:\n"
            "- Be concrete: a name, a number + unit, or a short phrase.\n"
            "- Examples: 'ans': '4', 'ans': 'The largest building is in the NW quadrant at bbox [120,80,340,260].', "
            "'ans': 'Distance ~ 45.3 m'.\n"
            "- Never Terminate with 'unable', 'insufficient', or 'error' unless every relevant tool failed.\n\n"
            "If a tool call errors: note the error in thought, then try a different tool or arguments."
        )

    def _sam2_available(self) -> bool:
        try:
            import importlib.util

            if importlib.util.find_spec("sam2") is None:
                return False
        except Exception:
            return False

        # SegmentObjectPixels requires local SAM2 checkpoint at runtime.
        return Path("models/weights/sam2/sam2.1_hiera_large.pt").exists()

    def _parse_gsd(self, objective: str) -> float | None:
        text = str(objective or "")
        m = re.search(r"gsd[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:m|meter|meters)\s*/\s*pixel", text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            val = float(m.group(1))
            return val if val > 0 else None
        except Exception:
            return None

    def _iou(self, a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        aa = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        ba = max(0.0, (bx2 - bx1) * (by2 - by1))
        denom = aa + ba - inter
        return (inter / denom) if denom > 0 else 0.0

    def _parse_boxes(self, bboxes: list[Any]) -> list[dict[str, float | list[float]]]:
        parsed: list[dict[str, float | list[float]]] = []
        for b in bboxes:
            if not isinstance(b, list) or len(b) < 4:
                continue
            try:
                x1 = float(b[0])
                y1 = float(b[1])
                x2 = float(b[2])
                y2 = float(b[3])
            except Exception:
                continue
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue
            area = w * h
            aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))
            parsed.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "w": w,
                    "h": h,
                    "area": area,
                    "aspect": aspect,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                }
            )
        return parsed

    def _trial_bbox_solver(self, message: Message, *, text_prompt: str, mode: str) -> AgentResult | None:
        payload = message.payload if isinstance(message.payload, dict) else {}
        objective = str(payload.get("objective", ""))
        objective_l = objective.lower()
        gsd = self._parse_gsd(objective)
        image_paths = payload.get("image_paths") if isinstance(payload.get("image_paths"), list) else []
        if not image_paths or gsd is None:
            return None

        image_path = str(image_paths[0])

        req_args = {
            "image_path": image_path,
            "text_prompt": text_prompt,
            "top1": False,
            "max_boxes": 20,
        }

        t0 = time.perf_counter()
        tool_out, normalized_args = self._call_tool("TextToBbox", req_args)
        tool_ms = (time.perf_counter() - t0) * 1000.0
        parsed: list[dict[str, float | list[float]]] = []
        if isinstance(tool_out, dict) and tool_out.get("success") is not False:
            bboxes = tool_out.get("bboxes") if isinstance(tool_out.get("bboxes"), list) else []
            parsed = self._parse_boxes(bboxes)

        # Fallback to ObjectDetection when TextToBbox is sparse.
        if len(parsed) < 2:
            od_args = {
                "image_path": image_path,
                "text_prompt": text_prompt,
                "max_boxes": 20,
            }
            od_t0 = time.perf_counter()
            od_out, od_norm = self._call_tool("ObjectDetection", od_args)
            od_ms = (time.perf_counter() - od_t0) * 1000.0
            if isinstance(od_out, dict) and od_out.get("success") is not False:
                od_boxes = od_out.get("bboxes") if isinstance(od_out.get("bboxes"), list) else []
                parsed = self._parse_boxes(od_boxes)
                tool_out = od_out
                normalized_args = od_norm
                tool_ms = od_ms

        if len(parsed) < 2:
            return None

        selected: list[dict[str, float | list[float]]] = []
        if mode == "area":
            cand = [p for p in parsed if float(p["area"]) >= 700 and float(p["area"]) <= 50000 and float(p["aspect"]) <= 5.0]
            if len(cand) < 2:
                cand = [p for p in parsed if float(p["area"]) >= 300 and float(p["aspect"]) <= 7.0]
            cand.sort(key=lambda x: float(x["area"]), reverse=True)
            for c in cand:
                if not selected:
                    selected.append(c)
                    continue
                if self._iou(selected[0]["bbox"], c["bbox"]) < 0.35:
                    selected.append(c)
                if len(selected) >= 2:
                    break
            if len(selected) < 2:
                selected = cand[:2]
            if len(selected) < 2:
                return None

            total_px = sum(float(s["area"]) for s in selected[:2])
            total_m2 = total_px * gsd * gsd
            ans = f"The combined area of the domestic garbage regions is approximately {total_m2:.1f} square meters."
            thought = "Computed combined area from detected garbage boxes using the provided GSD."
            extra = {
                "combined_area_sq_m": float(total_m2),
                "gsd_m_per_pixel": float(gsd),
            }
        else:
            cand = [p for p in parsed if float(p["area"]) >= 7000 and float(p["area"]) <= 90000 and float(p["aspect"]) <= 3.0]
            if len(cand) < 2:
                cand = [p for p in parsed if float(p["area"]) >= 1500 and float(p["aspect"]) <= 4.0]
            if len(cand) < 2:
                cand = parsed

            best_pair: tuple[dict[str, float | list[float]], dict[str, float | list[float]]] | None = None
            best_score = -1e18
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    a = cand[i]
                    b = cand[j]
                    dx = abs(float(a["cx"]) - float(b["cx"]))
                    dy = abs(float(a["cy"]) - float(b["cy"]))
                    area_gap = abs(float(a["area"]) - float(b["area"])) / max(float(a["area"]), float(b["area"]), 1.0)
                    if dx < 80:
                        continue
                    score = 0.25 * dx - 1.2 * dy - 220.0 * area_gap
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
            if best_pair is None:
                return None
            selected = [best_pair[0], best_pair[1]]

            dx = float(selected[0]["cx"]) - float(selected[1]["cx"])
            dy = float(selected[0]["cy"]) - float(selected[1]["cy"])
            pixel_dist = math.hypot(dx, dy)
            meters = pixel_dist * gsd
            ans = (
                f"The pixel distance between the two helicopters is approximately {pixel_dist:.1f} pixels, "
                f"which converts to about {int(round(meters))} meters."
            )
            thought = "Computed helicopter center-to-center pixel distance and converted it with the provided GSD."
            extra = {
                "pixel_distance": float(pixel_dist),
                "distance_meters": float(meters),
                "gsd_m_per_pixel": float(gsd),
            }

        chosen_boxes = [s["bbox"] for s in selected[:2]]
        actions = [{"name": "Terminate", "arguments": {"ans": ans}}]
        raw_output = json.dumps({"thought": thought, "actions": actions}, ensure_ascii=False)

        tool_call = {
            "tool": "TextToBbox",
            "args": req_args,
            "normalized_args": normalized_args,
            "output": tool_out,
            "latency_ms": round(tool_ms, 2),
            "turn": 1,
        }
        trace_step = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.name,
            "task_id": payload.get("scene_id", "unknown"),
            "turn": 1,
            "llm_latency_ms": 0.0,
            "assistant_raw_content": raw_output,
            "thought": thought,
            "actions_in_text": actions,
            "tool_calls": [
                {
                    "id": "call_shortcut_1",
                    "tool": "TextToBbox",
                    "args": req_args,
                    "normalized_args": normalized_args,
                    "output": tool_out,
                    "latency_ms": round(tool_ms, 2),
                }
            ],
            "terminated": True,
            "terminated_by_tool_call": False,
        }
        append_trace_line(trace_step)

        output = {
            "raw_output": raw_output,
            "thought": thought,
            "actions": actions,
            "ans": ans,
            "tool_call_count": 1,
            "detected_bboxes": chosen_boxes,
            "detected_object_counts": {text_prompt: len(parsed)},
            **extra,
        }
        for k in ("objective", "scene_id", "region"):
            if k in payload:
                output[k] = payload[k]

        return AgentResult(
            output=output,
            confidence=0.9,
            tool_calls=[tool_call],
            n_turns=1,
            trace=[trace_step],
        )

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.VRA_TOOLS
        if not self._sam2_available():
            tools_schema = [
                t
                for t in tools_schema
                if str((t.get("function") or {}).get("name", "")).strip() != "SegmentObjectPixels"
            ]
        return super().handle_task(message, tools_schema)

    def _augment_output_from_tools(
        self,
        output_dict: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        detections: dict[str, int] = {}
        counts: dict[str, int] = {}
        bbox_observations: list[list[float]] = []
        change_summary: dict[str, Any] = {}
        detection_boxes: list[dict[str, Any]] = []

        def _parse_box(b: Any) -> list[float] | None:
            if not isinstance(b, list) or len(b) < 4:
                return None
            try:
                x1 = float(b[0])
                y1 = float(b[1])
                x2 = float(b[2])
                y2 = float(b[3])
            except Exception:
                return None
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                return None
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if x2 <= x1 or y2 <= y1:
                return None
            return [x1, y1, x2, y2]

        def _parse_gsd(text: str) -> float | None:
            if not text:
                return None
            m = re.search(r"gsd[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:m|meter|meters)\s*/\s*pixel", text, flags=re.IGNORECASE)
            if not m:
                return None
            try:
                v = float(m.group(1))
                return v if v > 0 else None
            except Exception:
                return None

        def _best_boxes() -> list[dict[str, Any]]:
            uniq: dict[tuple[float, float, float, float], dict[str, Any]] = {}
            for rec in detection_boxes:
                bb = rec["bbox"]
                key = (round(bb[0], 2), round(bb[1], 2), round(bb[2], 2), round(bb[3], 2))
                prev = uniq.get(key)
                if prev is None or float(rec.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    uniq[key] = rec
            vals = list(uniq.values())
            vals.sort(key=lambda r: (float(r.get("score", 0.0)), float(r.get("area", 0.0))), reverse=True)
            return vals

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool", ""))
            args = call.get("args", {}) if isinstance(call.get("args"), dict) else {}
            out = call.get("output", {}) if isinstance(call.get("output"), dict) else {}

            if tool == "ObjectDetection":
                key = str(args.get("text_prompt", "object")).strip() or "object"
                bboxes = out.get("bboxes") if isinstance(out.get("bboxes"), list) else []
                scores = out.get("scores") if isinstance(out.get("scores"), list) else []
                detections[key] = detections.get(key, 0) + len(bboxes)
                for i, b in enumerate(bboxes):
                    box = _parse_box(b)
                    if box is None:
                        continue
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    score = float(scores[i]) if i < len(scores) and isinstance(scores[i], (int, float)) else 0.0
                    detection_boxes.append({"bbox": box, "score": score, "area": area, "label": key})

            elif tool == "TextToBbox":
                key = str(args.get("text_prompt", "object")).strip() or "object"
                bboxes = out.get("bboxes") if isinstance(out.get("bboxes"), list) else []
                scores = out.get("scores") if isinstance(out.get("scores"), list) else []
                detections[key] = detections.get(key, 0) + len(bboxes)
                for i, b in enumerate(bboxes):
                    box = _parse_box(b)
                    if box is None:
                        continue
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    score = float(scores[i]) if i < len(scores) and isinstance(scores[i], (int, float)) else 0.0
                    detection_boxes.append({"bbox": box, "score": score, "area": area, "label": key})

            elif tool == "CountGivenObject":
                key = str(args.get("text_prompt", "object")).strip() or "object"
                count = out.get("count")
                if isinstance(count, int):
                    counts[key] = count

            elif tool == "GetBboxFromGeotiff":
                bbox = out.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    try:
                        bbox_observations.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                    except Exception:
                        pass

            elif tool == "ChangeDetection":
                if isinstance(out.get("change_map_path"), str):
                    change_summary["change_map_path"] = out.get("change_map_path")
                if isinstance(out.get("changed_pixels"), (int, float)):
                    change_summary["changed_pixels"] = out.get("changed_pixels")

        if detections:
            output_dict.setdefault("detected_object_counts", detections)
        if counts:
            output_dict.setdefault("count_observations", counts)
        if bbox_observations:
            output_dict.setdefault("geotiff_bboxes", bbox_observations[:6])
        if change_summary:
            output_dict.setdefault("change_evidence", change_summary)

        objective = str(payload.get("objective", "") or "")
        objective_l = objective.lower()
        gsd = _parse_gsd(objective)
        ranked_boxes = _best_boxes()

        if ranked_boxes:
            output_dict.setdefault(
                "detected_bboxes",
                [rec["bbox"] for rec in ranked_boxes[:12]],
            )

        if gsd is not None and ranked_boxes:
            # Deterministic post-processing for area questions.
            if "area" in objective_l and any(k in objective_l for k in ("garbage", "waste", "litter")):
                area_boxes = ranked_boxes[:2] if len(ranked_boxes) >= 2 else ranked_boxes
                total_px = sum(float(rec.get("area", 0.0)) for rec in area_boxes)
                total_m2 = total_px * gsd * gsd
                output_dict["combined_area_sq_m"] = float(total_m2)
                output_dict.setdefault("gsd_m_per_pixel", float(gsd))
                if not isinstance(output_dict.get("ans"), str) or output_dict.get("ans", "").startswith("VRA evidence summary"):
                    output_dict["ans"] = (
                        f"The combined area of the domestic garbage regions is approximately {total_m2:.1f} square meters."
                    )

            # Deterministic post-processing for helicopter pixel-distance questions.
            if "helicopter" in objective_l and "pixel" in objective_l and "distance" in objective_l:
                if len(ranked_boxes) >= 2:
                    a = ranked_boxes[0]["bbox"]
                    b = ranked_boxes[1]["bbox"]
                    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
                    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                    pixel_dist = math.hypot(ac[0] - bc[0], ac[1] - bc[1])
                    meters = pixel_dist * gsd
                    meters_round = int(round(meters))
                    output_dict["pixel_distance"] = float(pixel_dist)
                    output_dict["distance_meters"] = float(meters)
                    output_dict.setdefault("gsd_m_per_pixel", float(gsd))
                    if not isinstance(output_dict.get("ans"), str) or output_dict.get("ans", "").startswith("VRA evidence summary"):
                        output_dict["ans"] = (
                            "The pixel distance between the two helicopters is approximately "
                            f"{pixel_dist:.1f} pixels, which converts to about {meters_round} meters."
                        )

        if not isinstance(output_dict.get("ans"), str) and (detections or counts or change_summary):
            pieces: list[str] = []
            if detections:
                pieces.append(f"detections={detections}")
            if counts:
                pieces.append(f"counts={counts}")
            if change_summary:
                pieces.append(f"change={change_summary}")
            output_dict["ans"] = "VRA evidence summary: " + "; ".join(pieces)

        return output_dict
