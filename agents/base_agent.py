from __future__ import annotations
import json
import httpx
import time
import logging
import re
import os
from typing import Any
from dataclasses import dataclass, field
from framework.mpc.message import Message

from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    output: dict[str, Any]
    confidence: float
    tool_calls: list[dict[str, Any]]
    n_turns: int
    trace: list[dict[str, Any]] = field(default_factory=list)

def append_trace_line(trace_step: dict[str, Any]) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "agent_trace.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace_step) + "\n")

class BaseAgent:
    name = "base"

    _MAX_TOOL_MSG_CHARS = 1600
    _MAX_MESSAGES = 10
    _CONTEXT_BUDGET_RATIO = 0.70

    _SMALLTALK_PREFIXES = (
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "what's up",
        "how are you",
    )

    def __init__(
        self,
        model_id: str = "mock-model",
        base_url: str = "http://localhost:8002/v1",
        tool_server: str = "http://localhost:9000",
        max_turns: int = 15,
        api_key: str = "sk-mock",
        allow_mock_fallback: bool = True,
        model_max_context_tokens: int = 32768,
        thinking_mode: str = "selective",
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.tool_server = tool_server
        self.max_turns = max_turns
        self.api_key = api_key
        self.allow_mock_fallback = allow_mock_fallback
        self.model_max_context_tokens = int(model_max_context_tokens)
        self.thinking_mode = str(thinking_mode or "selective")
        self._tokenizer_ready = False
        self._tokenizer = None
        self._global_token_budget = int(self.model_max_context_tokens * max(4, int(self.max_turns)))
        self._token_usage_prompt = 0
        self._token_usage_completion = 0
        self._token_usage_total = 0
        self._runtime_context_limit = int(self.model_max_context_tokens)

    def _init_tokenizer(self) -> None:
        if self._tokenizer_ready:
            return
        self._tokenizer_ready = True
        try:
            from transformers import AutoTokenizer  # type: ignore

            model_candidates = [self.model_id]
            if "qwen" not in str(self.model_id).lower():
                model_candidates.append("Qwen/Qwen2.5-7B-Instruct")
            for cand in model_candidates:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(cand, local_files_only=True, trust_remote_code=True)
                    break
                except Exception:
                    continue
        except Exception:
            self._tokenizer = None

    def _register_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._token_usage_prompt += max(0, int(prompt_tokens))
        self._token_usage_completion += max(0, int(completion_tokens))
        self._token_usage_total = self._token_usage_prompt + self._token_usage_completion

    def _reset_token_usage(self) -> None:
        # Budget accounting should be scoped to one task execution, not process lifetime.
        self._token_usage_prompt = 0
        self._token_usage_completion = 0
        self._token_usage_total = 0

    def _estimate_tokens(self, text: str) -> int:
        self._init_tokenizer()
        if self._tokenizer is not None:
            try:
                return max(1, len(self._tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return max(1, len(enc.encode(text)))
        except Exception:
            return max(1, len(text) // 4)

    def _estimate_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for m in messages:
            try:
                txt = json.dumps(m, ensure_ascii=False)
            except TypeError:
                txt = str(m)
            total += self._estimate_tokens(txt)
        return total

    def _get_system_prompt(self) -> str:
        return (
            f"You are the {self.name} agent. "
            "Use strict ReAct loops: Think, call one tool at a time, observe output, then decide next action. "
            "Return results grounded in tool evidence and avoid unsupported assumptions."
        )

    def _chat_completions_url(self) -> str:
        return f"{str(self.base_url).rstrip('/')}/chat/completions"

    def _is_hf_router(self) -> bool:
        base = str(self.base_url or "").lower()
        return "huggingface.co" in base or "hf.space" in base

    def _resolve_api_key(self) -> str:
        key = str(self.api_key or "").strip()
        if key and key != "sk-mock":
            return key
        for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env_val = str(os.getenv(env_name, "")).strip()
            if env_val:
                return env_val
        return key

    def _effective_context_limit(self) -> int:
        return max(1024, int(self._runtime_context_limit or self.model_max_context_tokens))

    def _update_runtime_context_limit_from_error(self, error_text: str) -> None:
        m = re.search(r"maximum context length is\s*(\d+)", str(error_text or ""), flags=re.IGNORECASE)
        if not m:
            return
        try:
            discovered = int(m.group(1))
        except Exception:
            return
        if discovered <= 0:
            return
        self._runtime_context_limit = min(self._effective_context_limit(), discovered)

    def _extract_server_token_counts(self, error_text: str) -> tuple[int | None, int | None, int | None]:
        """
        Parse OpenAI/vLLM overflow errors like:
        "maximum context length is 8192 ... requested 8707 (6659 in messages, 2048 in completion)"
        Returns: (max_context, message_tokens, completion_tokens)
        """
        txt = str(error_text or "")
        m_max = re.search(r"maximum context length is\s*(\d+)", txt, flags=re.IGNORECASE)
        m_split = re.search(
            r"requested\s*(\d+)\s*tokens\s*\((\d+)\s*in\s*the\s*messages?,\s*(\d+)\s*in\s*the\s*completion\)",
            txt,
            flags=re.IGNORECASE,
        )
        max_ctx = int(m_max.group(1)) if m_max else None
        msg_toks = int(m_split.group(2)) if m_split else None
        comp_toks = int(m_split.group(3)) if m_split else None
        return max_ctx, msg_toks, comp_toks

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Call the LLM via OpenAI-compatible API.
        Falls back to a mock response if the server is unreachable.
        """
        prepared_messages = self._prepare_messages_for_llm(messages)
        if prepared_messages is not messages:
            messages[:] = prepared_messages

        resolved_key = self._resolve_api_key()
        headers = {"Content-Type": "application/json"}
        if resolved_key and resolved_key != "sk-mock":
            headers["Authorization"] = f"Bearer {resolved_key}"

        if self._is_hf_router() and "Authorization" not in headers:
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "thought": (
                            "Hugging Face token is missing. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) "
                            "to use remote inference."
                        ),
                        "actions": [{"name": "Terminate", "arguments": {"ans": "error"}}],
                    }
                ),
                "tool_calls": [],
            }

        context_limit = self._effective_context_limit()

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.0,
            # Keep completions conservative for 8k-context local models to reduce overflow retries.
            "max_tokens": max(256, min(1024, context_limit // 6)),
        }

        # HF router may reject OpenAI tools schema on some providers/models.
        # In that case we rely on strict JSON actions and inline parsing.
        if tools and not self._is_hf_router():
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        estimated_prompt_tokens = self._estimate_messages_tokens(messages)
        # Keep completion budget within model context headroom to avoid 400 overflow.
        # Reserve generous slack because client-side token estimation tends to undercount vs server accounting.
        available_for_completion = int(context_limit - estimated_prompt_tokens - 1024)
        payload["max_tokens"] = max(64, min(int(payload["max_tokens"]), max(64, available_for_completion)))
        if self._token_usage_total + estimated_prompt_tokens > self._global_token_budget:
            logger.warning(
                "Global token budget pressure for agent=%s used=%s next_prompt=%s budget=%s; applying aggressive shrink",
                self.name,
                self._token_usage_total,
                estimated_prompt_tokens,
                self._global_token_budget,
            )
            shrunk = self._aggressive_shrink_messages(messages)
            messages[:] = shrunk
            payload["messages"] = messages
            estimated_prompt_tokens = self._estimate_messages_tokens(messages)
            if self._token_usage_total + estimated_prompt_tokens > self._global_token_budget:
                logger.error(
                    "Global token budget exhausted for agent=%s used=%s next_prompt=%s budget=%s",
                    self.name,
                    self._token_usage_total,
                    estimated_prompt_tokens,
                    self._global_token_budget,
                )
                return {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "thought": "Global token budget exhausted before LLM call.",
                            "actions": [{"name": "Terminate", "arguments": {"ans": "error"}}],
                        }
                    ),
                    "tool_calls": [],
                }

        try:
            resp = httpx.post(
                self._chat_completions_url(),
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            resp.raise_for_status()
            body = resp.json()
            msg = body["choices"][0]["message"]
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            prompt_tokens = int(usage.get("prompt_tokens", estimated_prompt_tokens or 0))
            completion_tokens = int(
                usage.get(
                    "completion_tokens",
                    self._estimate_tokens(json.dumps(msg, ensure_ascii=False)) if isinstance(msg, dict) else 0,
                )
            )
            self._register_token_usage(prompt_tokens, completion_tokens)
            return msg
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            body = e.response.text if e.response is not None else ""
            logger.error("LLM HTTP error for agent=%s status=%s body=%s", self.name, status, body[:400])
            lower = body.lower()

            if (
                self._is_hf_router()
                and status in (400, 422)
                and tools
                and ("tool" in lower or "schema" in lower or "function" in lower)
            ):
                retry_payload = dict(payload)
                retry_payload.pop("tools", None)
                retry_payload.pop("tool_choice", None)
                try:
                    retry_resp = httpx.post(
                        self._chat_completions_url(),
                        json=retry_payload,
                        headers=headers,
                        timeout=120.0,
                    )
                    retry_resp.raise_for_status()
                    body_retry = retry_resp.json()
                    msg = body_retry["choices"][0]["message"]
                    usage = body_retry.get("usage", {}) if isinstance(body_retry, dict) else {}
                    prompt_tokens = int(usage.get("prompt_tokens", estimated_prompt_tokens or 0))
                    completion_tokens = int(
                        usage.get(
                            "completion_tokens",
                            self._estimate_tokens(json.dumps(msg, ensure_ascii=False)) if isinstance(msg, dict) else 0,
                        )
                    )
                    self._register_token_usage(prompt_tokens, completion_tokens)
                    return msg
                except Exception as retry_exc:
                    logger.error("HF retry without tools failed for agent=%s error=%s", self.name, retry_exc)

            is_context_overflow = (
                status in (400, 413)
                and (
                    "maximum context length" in lower
                    or "reduce the length" in lower
                    or "context" in lower
                    or "token" in lower
                    or status == 413
                )
            )

            if is_context_overflow:
                logger.warning("LLM context overflow detected for agent=%s; retrying with compressed context", self.name)
                self._update_runtime_context_limit_from_error(body)
                shrunk = self._aggressive_shrink_messages(messages)
                messages[:] = shrunk
                retry_payload = dict(payload)
                retry_payload["messages"] = shrunk
                retry_prompt_tokens = self._estimate_messages_tokens(shrunk)
                retry_context_limit = self._effective_context_limit()
                retry_available = max(64, int(retry_context_limit - retry_prompt_tokens - 1024))
                retry_max_tokens = max(64, min(int(retry_payload.get("max_tokens", 512)), retry_available))

                # Prefer server-reported message-token accounting when available.
                srv_max_ctx, srv_msg_toks, _ = self._extract_server_token_counts(body)
                if isinstance(srv_max_ctx, int) and srv_max_ctx > 0:
                    retry_context_limit = min(retry_context_limit, srv_max_ctx)
                if isinstance(srv_max_ctx, int) and isinstance(srv_msg_toks, int):
                    server_available = max(64, int(srv_max_ctx - srv_msg_toks - 64))
                    retry_max_tokens = max(64, min(retry_max_tokens, server_available, 512))
                else:
                    retry_max_tokens = max(64, min(retry_max_tokens, 512))

                retry_payload["max_tokens"] = retry_max_tokens
                try:
                    retry_resp = httpx.post(
                        self._chat_completions_url(),
                        json=retry_payload,
                        headers=headers,
                        timeout=120.0,
                    )
                    retry_resp.raise_for_status()
                    return retry_resp.json()["choices"][0]["message"]
                except Exception as retry_exc:
                    logger.error("Context-overflow retry failed for agent=%s error=%s", self.name, retry_exc)
                    return {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "thought": (
                                    f"LLM context overflow persisted after compression: {retry_exc}"
                                ),
                                "actions": [{"name": "Terminate", "arguments": {"ans": "error"}}],
                            }
                        ),
                        "tool_calls": [],
                    }

            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "thought": f"LLM HTTP error {status}: {body[:300]}",
                        "actions": [{"name": "Terminate", "arguments": {"ans": "error"}}],
                    }
                ),
                "tool_calls": [],
            }
        except httpx.ConnectError:
            if self.name == "vra":
                logger.error("VRA LLM endpoint unreachable at %s; cannot fallback to mock for vision tasks", self.base_url)
                return {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "thought": f"VRA model endpoint unavailable at {self.base_url}.",
                            "actions": [{"name": "Terminate", "arguments": {"ans": "error"}}],
                        }
                    ),
                    "tool_calls": [],
                }
            if not self.allow_mock_fallback:
                raise RuntimeError(
                    f"LLM server unreachable at {self.base_url}. "
                    "Set allow_mock_fallback=True only for offline tests."
                )
            logger.warning("LLM server unreachable for agent=%s base_url=%s; using mock fallback", self.name, self.base_url)
            # Server not running — return mock for testing
            return {
                "role": "assistant",
                "content": json.dumps({
                    "thought": "No LLM server available. Returning mock output.",
                    "actions": [{"name": "Terminate", "arguments": {"ans": "mock result"}}]
                }),
                "tool_calls": []
            }
        except httpx.TimeoutException:
            logger.warning("LLM timeout for agent=%s base_url=%s", self.name, self.base_url)
            return {
                "role": "assistant",
                "content": json.dumps({
                    "thought": "LLM request timed out.",
                    "actions": [{"name": "Terminate", "arguments": {"ans": "timeout"}}]
                }),
                "tool_calls": []
            }
        except Exception as e:
            logger.exception("Unhandled LLM call failure for agent=%s error=%s", self.name, e)
            return {
                "role": "assistant",
                "content": json.dumps({
                    "thought": f"LLM call failed: {e}",
                    "actions": [{"name": "Terminate", "arguments": {"ans": "error"}}]
                }),
                "tool_calls": []
            }

    def _prepare_messages_for_llm(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        msgs = self._trim_message_history(messages)
        budget = int(self.model_max_context_tokens * self._CONTEXT_BUDGET_RATIO)

        # First pass: summarize old tool responses.
        if self._estimate_messages_tokens(msgs) > budget:
            for i in range(2, max(2, len(msgs) - 2)):
                m = msgs[i]
                if m.get("role") != "tool":
                    continue
                content = str(m.get("content", ""))
                if len(content) > 800:
                    m["content"] = self._tool_result_for_llm({"content": content})

        # Second pass: drop oldest non-critical turns if still over budget.
        while self._estimate_messages_tokens(msgs) > budget and len(msgs) > 4:
            removed = False
            for i in range(2, len(msgs) - 2):
                role = str(msgs[i].get("role", ""))
                if role in {"tool", "assistant", "user"}:
                    del msgs[i]
                    removed = True
                    break
            if not removed:
                break

        return msgs

    def _aggressive_shrink_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        msgs = self._trim_message_history(messages)
        for m in msgs:
            if m.get("role") == "tool":
                m["content"] = self._tool_result_for_llm({"content": m.get("content", "")})
        while self._estimate_messages_tokens(msgs) > int(self.model_max_context_tokens * 0.55) and len(msgs) > 4:
            del msgs[2]
        return msgs

    def _normalize_tool_arguments(self, tool_name: str, arguments: dict | None) -> dict:
        """
        Best-effort compatibility layer for OpenEarthAgent-style arguments.
        Keeps agent-side logged args unchanged while sending server-compatible payloads.
        """
        args = dict(arguments or {})
        norm = dict(args)
        t = (tool_name or "").strip().lower()

        def _to_float_pair(value: Any) -> list[float] | None:
            """Normalize heterogeneous point payloads to [x, y]/[lat, lon] float pairs."""
            if isinstance(value, dict):
                if "lat" in value and "lon" in value:
                    try:
                        return [float(value["lat"]), float(value["lon"])]
                    except Exception:
                        return None
                if "x" in value and "y" in value:
                    try:
                        return [float(value["x"]), float(value["y"])]
                    except Exception:
                        return None
                for key in ("point", "coord", "coords", "coordinate", "coordinates", "location"):
                    nested = value.get(key)
                    pair = _to_float_pair(nested)
                    if pair is not None:
                        return pair
                return None

            if isinstance(value, (list, tuple)):
                if len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
                    try:
                        return [float(value[0]), float(value[1])]
                    except Exception:
                        return None
                for item in value:
                    pair = _to_float_pair(item)
                    if pair is not None:
                        return pair
                return None

            return None

        def move(src: str, dst: str) -> None:
            if src in norm and dst not in norm:
                norm[dst] = norm[src]

        # Cross-tool aliases commonly seen in OpenEarthAgent traces.
        move("image", "image_path")
        move("text", "text_prompt")
        move("geotiff", "geotiff_path")
        move("pre_image", "image_path_1")
        move("post_image", "image_path_2")

        if t == "getareaboundary":
            move("area", "area_name")

        elif t == "addpoislayer":
            if "poi_category" not in norm:
                q = norm.get("query")
                if isinstance(q, dict):
                    amenity = q.get("amenity")
                    if isinstance(amenity, list) and amenity:
                        norm["poi_category"] = str(amenity[0])
                    elif amenity is not None:
                        norm["poi_category"] = str(amenity)
                    else:
                        norm["poi_category"] = json.dumps(q)
                elif q is not None:
                    norm["poi_category"] = str(q)
                else:
                    norm["poi_category"] = "poi"
            if "bbox" not in norm:
                if isinstance(norm.get("boundary_wkt"), str):
                    b = self._extract_wkt_bbox(str(norm.get("boundary_wkt", "")))
                    if isinstance(b, list) and len(b) == 4:
                        norm["bbox"] = b

        elif t == "computedistance":
            # Native mode.
            if "point_a" in norm and "point_b" in norm:
                pa = _to_float_pair(norm.get("point_a"))
                pb = _to_float_pair(norm.get("point_b"))
                if pa is not None:
                    norm["point_a"] = pa
                if pb is not None:
                    norm["point_b"] = pb
            elif isinstance(norm.get("points"), list):
                pts = norm.get("points")
                if isinstance(pts, list) and len(pts) >= 2:
                    pa = _to_float_pair(pts[0])
                    pb = _to_float_pair(pts[1])
                    if pa is not None and pb is not None:
                        norm["point_a"] = pa
                        norm["point_b"] = pb
            # Layer-to-layer mode is unsupported by this tool contract.
            # Do not fabricate placeholder coordinates.

        elif t == "texttobbox":
            move("image", "image_path")
            move("text", "text_prompt")
            norm.setdefault("text_prompt", "object")

        elif t == "objectdetection":
            move("image", "image_path")
            move("text", "text_prompt")
            norm.setdefault("text_prompt", "object")

        elif t == "countgivenobject":
            move("image", "image_path")
            move("text", "text_prompt")
            norm.setdefault("text_prompt", "object")

        elif t == "imagedescription":
            move("image", "image_path")

        elif t == "regionattributedescription":
            move("image", "image_path")
            if "bbox" not in norm and isinstance(norm.get("bboxes"), list) and norm["bboxes"]:
                first = norm["bboxes"][0]
                if isinstance(first, list):
                    norm["bbox"] = first
            norm.setdefault("bbox", [0.0, 0.0, 1.0, 1.0])

        elif t == "ocr":
            move("image", "image_path")

        elif t == "changedetection":
            move("pre_image", "image_path_1")
            move("post_image", "image_path_2")

        elif t == "segmentobjectpixels":
            move("image", "image_path")
            if "bboxes" not in norm:
                if isinstance(norm.get("bbox"), list):
                    norm["bboxes"] = [norm["bbox"]]
                else:
                    norm["bboxes"] = [[0.0, 0.0, 1.0, 1.0]]

        elif t == "drawbox":
            move("image", "image_path")
            if "bboxes" not in norm:
                if isinstance(norm.get("bbox"), list):
                    norm["bboxes"] = [norm["bbox"]]
                else:
                    norm["bboxes"] = [[0.0, 0.0, 1.0, 1.0]]
            norm.setdefault("output_path", "data/tmp/draw_box.png")

        elif t == "addtext":
            move("image", "image_path")
            norm.setdefault("output_path", "data/tmp/add_text.png")

        elif t == "getbboxfromgeotiff":
            move("geotiff", "geotiff_path")

        elif t == "displayonmap":
            norm.setdefault("features", [])
            norm.setdefault("output_html", "data/tmp/map.png")

        elif t in {"displayongeotiff", "displaygeotiff"}:
            move("geotiff", "geotiff_path")
            norm.setdefault("features", [])
            norm.setdefault("output_path", "data/tmp/display_geotiff.png")

        elif t == "addindexlayer":
            move("index_type", "index_name")
            move("geotiff", "geotiff_path")
            if "geotiff_path" not in norm and "gpkg" in norm:
                norm["geotiff_path"] = str(norm["gpkg"])
            norm.setdefault("index_name", "NDVI")

        elif t == "computeindexchange":
            move("layer1_name", "index_path_pre")
            move("layer2_name", "index_path_post")

        elif t == "temporalstackloader":
            if "image_path" in norm and "image_paths" not in norm:
                norm["image_paths"] = [str(norm["image_path"])]

        elif t == "showindexlayer":
            move("out_file", "index_array_path")
            move("layer_name", "index_array_path")

        elif t == "solver":
            move("command", "equation")

        elif t == "plot":
            if "command" in norm and not all(k in norm for k in ("x_values", "y_values", "output_path")):
                norm.setdefault("x_values", [0.0, 1.0])
                norm.setdefault("y_values", [0.0, 1.0])
                norm.setdefault("output_path", "data/tmp/plot.png")

        return norm

    # POI synonym fallback chain: when AddPoisLayer returns empty pois, retry with alternates.
    _POI_SYNONYM_CHAINS: dict[str, list[str]] = {
        "mall": ["shopping_mall", "shopping_centre", "shopping", "supermarket"],
        "shopping_mall": ["mall", "shopping_centre", "shopping", "supermarket"],
        "shopping_centre": ["mall", "shopping_mall", "shopping", "supermarket"],
        "shopping": ["mall", "shopping_mall", "shopping_centre", "supermarket"],
        "garden": ["park", "leisure_park", "nature_reserve"],
        "park": ["garden", "leisure_park", "nature_reserve"],
        "nature_reserve": ["park", "garden"],
        "shelter": ["hospital", "clinic", "emergency_shelter"],
        "hospital": ["clinic", "shelter"],
        "clinic": ["hospital", "shelter", "pharmacy"],
        "fire_station": ["fire", "fireStation"],
        "fire": ["fire_station"],
        "police": ["police_station"],
        "police_station": ["police"],
        "school": ["university", "college", "education"],
        "university": ["college", "school"],
        "restaurant": ["cafe", "fast_food", "food"],
        "cafe": ["restaurant", "fast_food"],
        "fast_food": ["restaurant", "cafe"],
        "food": ["restaurant", "cafe", "fast_food"],
        "bank": ["atm"],
        "atm": ["bank"],
        "hotel": ["motel", "hostel", "guest_house"],
        "fuel": ["fuel_station", "gas_station", "gas"],
        "fuel_station": ["fuel", "gas_station"],
        "gas_station": ["fuel", "fuel_station"],
        "parking": ["parking_space", "parking_entrance"],
        "bus_stop": ["bus"],
        "bus": ["bus_stop"],
        "train": ["train_station", "railway"],
        "airport": ["aerodrome", "terminal"],
        "supermarket": ["convenience", "marketplace", "mall"],
        "marketplace": ["market", "supermarket"],
        "market": ["marketplace", "supermarket"],
        "pharmacy": ["clinic", "hospital"],
        "court": ["courthouse"],
        "courthouse": ["court"],
        "post_office": ["post"],
        "post": ["post_office"],
    }

    def _call_tool(self, tool_name: str, arguments: dict) -> tuple[Any, dict]:
        """Call a tool on the tool server. Returns (result_or_error, normalized_args)."""
        url = f"{self.tool_server}/tools/{tool_name}"
        normalized_args = self._normalize_tool_arguments(tool_name, arguments)
        tool_l = str(tool_name or "").strip().lower()

        # Pre-create parent folders for tools that persist outputs to disk.
        for out_key in ("output_path", "output_html"):
            out_val = normalized_args.get(out_key)
            if isinstance(out_val, str) and out_val.strip():
                try:
                    Path(out_val).parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

        if tool_l == "addpoislayer":
            timeout_s = 180.0
        elif tool_l == "getareaboundary":
            timeout_s = 120.0
        else:
            timeout_s = 60.0

        def _post_once(payload: dict) -> tuple[Any, Exception | None]:
            try:
                r = httpx.post(url, json=payload, timeout=timeout_s)
                r.raise_for_status()
                return r.json(), None
            except BaseException as exc:  # noqa: BLE001
                return None, exc

        try:
            resp_json, err = _post_once(normalized_args)
            if err is not None:
                raise err

            # Empty-POI synonym-retry: if AddPoisLayer returned zero pois, try chained synonyms.
            if (
                tool_l == "addpoislayer"
                and isinstance(resp_json, dict)
                and resp_json.get("success") is not False
                and isinstance(resp_json.get("pois"), list)
                and len(resp_json.get("pois", [])) == 0
            ):
                original_cat = str(normalized_args.get("poi_category", "")).strip().lower().replace(" ", "_")
                tried: set[str] = {original_cat}
                synonyms = list(self._POI_SYNONYM_CHAINS.get(original_cat, []))
                for alt in synonyms:
                    alt_key = alt.strip().lower().replace(" ", "_")
                    if alt_key in tried:
                        continue
                    tried.add(alt_key)
                    retry_args = dict(normalized_args)
                    retry_args["poi_category"] = alt
                    retry_json, retry_err = _post_once(retry_args)
                    if retry_err is not None:
                        continue
                    if (
                        isinstance(retry_json, dict)
                        and isinstance(retry_json.get("pois"), list)
                        and len(retry_json.get("pois", [])) > 0
                    ):
                        retry_json = dict(retry_json)
                        retry_json["synonym_retry"] = {
                            "original_category": original_cat,
                            "matched_category": alt,
                            "tried": list(tried),
                        }
                        return retry_json, retry_args
                # No synonym matched: return original empty result (still success).
                if isinstance(resp_json, dict):
                    resp_json = dict(resp_json)
                    resp_json["synonym_retry"] = {
                        "original_category": original_cat,
                        "matched_category": None,
                        "tried": list(tried),
                    }
                return resp_json, normalized_args

            return resp_json, normalized_args
        except httpx.ConnectError:
            return {"success": False, "error": f"Tool server not reachable at {self.tool_server}"}, normalized_args
        except httpx.TimeoutException:
            if tool_l == "addpoislayer":
                # Soft fallback keeps the pipeline alive; downstream agent can still reason about missing POIs.
                return {
                    "success": True,
                    "pois": [],
                    "warning": "AddPoisLayer timed out; returned empty POI set fallback.",
                }, normalized_args
            return {"success": False, "error": f"Tool {tool_name} timed out"}, normalized_args
        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response is not None else ""
            return {
                "success": False,
                "error": f"Tool {tool_name} HTTP {e.response.status_code if e.response is not None else 'error'}: {body[:300]}",
            }, normalized_args
        except BaseException as e:
            return {"success": False, "error": str(e)}, normalized_args

    def _bbox_list(self, args: dict[str, Any]) -> list[list[float]]:
        boxes: list[list[float]] = []
        bboxes = args.get("bboxes") if isinstance(args.get("bboxes"), list) else None
        if bboxes:
            for b in bboxes:
                if not isinstance(b, list) or len(b) < 4:
                    continue
                try:
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                except Exception:
                    continue
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                boxes.append([x1, y1, x2, y2])
        bbox = args.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4 and not boxes:
            try:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                boxes.append([x1, y1, x2, y2])
            except Exception:
                pass
        return boxes

    def _emulate_unsupported_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        n = str(tool_name or "").strip().lower()
        boxes = self._bbox_list(args if isinstance(args, dict) else {})
        gsd = args.get("gsd") if isinstance(args, dict) else None
        if not isinstance(gsd, (int, float)):
            gsd = args.get("gsd_m_per_pixel") if isinstance(args, dict) else None
        gsd_f = float(gsd) if isinstance(gsd, (int, float)) and float(gsd) > 0 else 1.0

        if n in {"computearea", "computepixelarea"}:
            if not boxes:
                return None
            total_px = 0.0
            for x1, y1, x2, y2 in boxes:
                total_px += max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_m2 = total_px * gsd_f * gsd_f
            return {
                "success": True,
                "emulated": True,
                "pixel_area": float(total_px),
                "area_sq_meters": float(area_m2),
                "gsd_m_per_pixel": float(gsd_f),
            }

        if n == "computecanopydiameter":
            if not boxes:
                return None
            best = max(boxes, key=lambda b: max(b[2] - b[0], b[3] - b[1]))
            diameter_px = max(best[2] - best[0], best[3] - best[1])
            diameter_m = diameter_px * gsd_f
            return {
                "success": True,
                "emulated": True,
                "diameter_pixels": float(diameter_px),
                "diameter_meters": float(diameter_m),
                "gsd_m_per_pixel": float(gsd_f),
            }

        if n == "segmentobjectpixels":
            if not boxes:
                return {
                    "success": True,
                    "emulated": True,
                    "polygons": [],
                    "warning": "SAM2 unavailable; returned empty polygon set.",
                }
            polys = []
            for x1, y1, x2, y2 in boxes:
                polys.append([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
            return {
                "success": True,
                "emulated": True,
                "polygons": polys,
                "warning": "SAM2 unavailable; approximated polygons from bounding boxes.",
            }

        return None

    def _strip_thinking_tokens(self, content: str) -> str:
        if not content:
            return ""
        try:
            cleaned = re.sub(r"<think>.*?</think>", "", str(content), flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        except Exception:
            cleaned = str(content)
        return cleaned.strip()

    def _extract_json_dict(self, text: str) -> dict[str, Any] | None:
        cleaned = self._strip_thinking_tokens(text)
        if not cleaned:
            return None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _extract_inline_tool_call(self, content: str) -> dict | None:
        """
        Extract a tool call from assistant text output (for models that
        don't use the OpenAI tool_calls format but embed JSON in text).
        """
        parsed = self._extract_json_dict(content)
        if not isinstance(parsed, dict):
            return None
        # Check for {actions: [{name, arguments}]} format
        if "actions" in parsed and isinstance(parsed["actions"], list):
            for act in parsed["actions"]:
                if isinstance(act, dict) and act.get("name"):
                    return act
        # Check for {name, arguments} format
        if parsed.get("name"):
            return parsed
        return None

    def _extract_thought_actions(self, content: str) -> tuple[str | None, list[dict[str, Any]]]:
        parsed = self._extract_json_dict(content)
        if isinstance(parsed, dict):
            thought = parsed.get("thought")
            actions = parsed.get("actions") if isinstance(parsed.get("actions"), list) else []
            return thought, actions
        return None, []

    def _extract_terminate_answer_from_actions(self, actions: list[dict[str, Any]]) -> str | None:
        for act in actions:
            if not isinstance(act, dict):
                continue
            name = str(act.get("name", "")).strip().lower()
            if name != "terminate":
                continue
            args = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
            for key in ("ans", "final_answer", "answer", "response"):
                val = args.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
        return None

    def _extract_terminate_answer_from_content(self, content: str) -> str | None:
        _, actions = self._extract_thought_actions(content)
        ans = self._extract_terminate_answer_from_actions(actions)
        if ans:
            return ans
        inline = self._extract_inline_tool_call(content)
        if not isinstance(inline, dict):
            return None
        if str(inline.get("name", "")).strip().lower() != "terminate":
            return None
        args = inline.get("arguments", {}) if isinstance(inline.get("arguments"), dict) else {}
        for key in ("ans", "final_answer", "answer", "response"):
            val = args.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
        return None

    def _tool_result_for_llm(self, value: Any) -> str:
        """
        Compact oversized tool payloads before feeding them back to the model.
        This avoids request validation failures when geospatial outputs are huge.
        """
        if isinstance(value, dict):
            if isinstance(value.get("boundary_wkt"), str):
                wkt_text = str(value.get("boundary_wkt", ""))
                bounds = self._extract_wkt_bbox(wkt_text)
                compact = {
                    "boundary_summary": {
                        "has_geometry": True,
                        "bbox": bounds,
                        "gpkg_path": value.get("gpkg_path"),
                    }
                }
                return json.dumps(compact, ensure_ascii=False)

            pois = value.get("pois")
            if isinstance(pois, list):
                compact_pois = []
                for item in pois[:5]:
                    if isinstance(item, dict):
                        compact_pois.append(
                            {
                                "name": item.get("name"),
                                "lat": item.get("lat"),
                                "lon": item.get("lon"),
                            }
                        )
                return json.dumps(
                    {
                        "pois_summary": {
                            "count": len(pois),
                            "sample": compact_pois,
                        }
                    },
                    ensure_ascii=False,
                )

        try:
            raw = json.dumps(value, ensure_ascii=False)
        except TypeError:
            raw = str(value)

        if len(raw) <= self._MAX_TOOL_MSG_CHARS:
            return raw

        preview_head = raw[: self._MAX_TOOL_MSG_CHARS // 2]
        preview_tail = raw[-(self._MAX_TOOL_MSG_CHARS // 4) :]
        summary: dict[str, Any] = {
            "truncated": True,
            "original_chars": len(raw),
            "preview_head": preview_head,
            "preview_tail": preview_tail,
        }

        if isinstance(value, dict):
            key_summary: dict[str, Any] = {}
            for k, v in value.items():
                if isinstance(v, (bool, int, float)):
                    key_summary[str(k)] = v
                elif isinstance(v, str):
                    key_summary[str(k)] = v[:120] + ("..." if len(v) > 120 else "")
                elif isinstance(v, list):
                    key_summary[str(k)] = f"<list len={len(v)}>"
                elif isinstance(v, dict):
                    key_summary[str(k)] = f"<dict keys={len(v)}>"
                else:
                    key_summary[str(k)] = f"<{type(v).__name__}>"
            summary["key_summary"] = key_summary

        return json.dumps(summary, ensure_ascii=False)

    def _extract_wkt_bbox(self, wkt_text: str) -> list[float] | None:
        try:
            from shapely import wkt as shapely_wkt  # type: ignore

            geom = shapely_wkt.loads(wkt_text)
            minx, miny, maxx, maxy = geom.bounds
            return [float(minx), float(miny), float(maxx), float(maxy)]
        except Exception:
            return None

    def _trim_message_history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) <= self._MAX_MESSAGES:
            return messages
        # Keep system and original user instruction, then tail recent turns.
        prefix = messages[:2]
        tail = messages[-(self._MAX_MESSAGES - 2) :]
        return prefix + tail

    def _is_smalltalk_objective(self, objective: str) -> bool:
        txt = (objective or "").strip().lower()
        if not txt:
            return False
        n_words = len(txt.split())
        if n_words <= 5:
            for p in self._SMALLTALK_PREFIXES:
                if txt == p or txt.startswith(p + " "):
                    return True
        return False

    def _smalltalk_terminate_answer(self) -> str:
        return (
            f"Hello! I am {self.name.upper()}. "
            "Please share a concrete geospatial or remote-sensing task and I will reason step-by-step before tool use."
        )

    def _synthesize_react_thought(self, objective: str, tool_name: str, t_args: dict[str, Any]) -> str:
        t = str(tool_name or "").strip()
        if not t:
            return "I need to reason about the next best action before responding."
        if objective:
            return (
                f"I should call {t} now to gather evidence for objective: {objective}. "
                "After observing the output, I will decide the next step."
            )
        return f"I should call {t} now, inspect the result, and then decide the next action."

    def _augment_output_from_tools(
        self,
        output_dict: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Agent-specific post-processing hook.

        Subclasses can derive structured evidence from executed tool calls and add
        it to the final output. Base implementation is a no-op.
        """
        return output_dict

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = []

        self._reset_token_usage()

        objective = str(message.payload.get("objective", "")).strip()

        if self._is_smalltalk_objective(objective):
            thought = "This is conversational smalltalk and does not require tool usage."
            answer = self._smalltalk_terminate_answer()
            actions = [{"name": "Terminate", "arguments": {"ans": answer}}]
            content_obj = {"thought": thought, "actions": actions}
            raw_content = json.dumps(content_obj)

            trace_step: dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent": self.name,
                "task_id": message.payload.get("scene_id", "unknown"),
                "turn": 1,
                "llm_latency_ms": 0.0,
                "assistant_raw_content": raw_content,
                "thought": thought,
                "actions_in_text": actions,
                "tool_calls": [],
                "terminated": True,
                "terminated_no_tool_call": True,
            }
            append_trace_line(trace_step)
            logger.info("[AGENT:%s | TURN 1 | 0.0ms] THOUGHT=%s TERMINATED=smalltalk", self.name, json.dumps(thought))

            output_dict: dict[str, Any] = {
                "raw_output": raw_content,
                "thought": thought,
                "actions": actions,
                "ans": answer,
            }
            for k in ("objective", "scene_id", "region"):
                if k in message.payload:
                    output_dict.setdefault(k, message.payload[k])

            return AgentResult(
                output=output_dict,
                confidence=0.8,
                tool_calls=[],
                n_turns=1,
                trace=[trace_step],
            )

        allowed_tool_names = {
            str((t.get("function") or {}).get("name", "")).strip()
            for t in tools_schema
            if isinstance(t, dict)
        }
        allowed_tool_names.discard("")

        prompt_payload = dict(message.payload)
        prompt_payload["enabled_tools"] = sorted(allowed_tool_names)

        messages = [
            {"role": "system", "content": self._build_task_adaptive_system_prompt(prompt_payload)},
            {"role": "user", "content": f"Task instructions:\n{json.dumps(message.payload)}"}
        ]

        if isinstance(message.payload.get("episodic_context"), dict) and message.payload.get("episodic_context"):
            messages.append(
                {
                    "role": "user",
                    "content": "Relevant episodic context:\n" + json.dumps(message.payload.get("episodic_context"), ensure_ascii=False),
                }
            )

        turns = 0
        all_tool_calls_logged = []
        trace: list[dict[str, Any]] = []
        response_msg: dict[str, Any] = {"content": ""}
        terminate_answer: str | None = None

        while turns < self.max_turns:
            turns += 1
            llm_t0 = time.perf_counter()
            response_msg = self._call_llm(messages, tools_schema)
            llm_ms = (time.perf_counter() - llm_t0) * 1000.0
            messages.append(response_msg)
            content = response_msg.get("content", "")
            thought, actions = self._extract_thought_actions(content)

            # Check for OpenAI-style tool_calls
            tool_calls = response_msg.get("tool_calls") or []

            # Also check for inline tool calls in text content
            if not tool_calls:
                inline = self._extract_inline_tool_call(content)
                if inline:
                    inline_name = str(inline.get("name", "")).strip()
                    inline_args = inline.get("arguments", {}) if isinstance(inline.get("arguments"), dict) else {}
                    if inline_name.lower() == "terminate":
                        terminate_answer = self._extract_terminate_answer_from_actions(
                            [{"name": inline_name, "arguments": inline_args}]
                        )
                    else:
                        tool_calls = [{
                            "id": f"call_inline_{turns}",
                            "function": {
                                "name": inline_name,
                                "arguments": json.dumps(inline_args)
                            }
                        }]

            if terminate_answer is None:
                terminate_answer = self._extract_terminate_answer_from_actions(actions)
            if terminate_answer is None:
                terminate_answer = self._extract_terminate_answer_from_content(content)

            if (thought is None or str(thought).strip() == "") and tool_calls:
                fn = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
                t_name = fn.get("name")
                t_args_str = fn.get("arguments", "{}")
                try:
                    t_args = json.loads(t_args_str) if isinstance(t_args_str, str) else t_args_str
                except (json.JSONDecodeError, TypeError):
                    t_args = {}
                thought = self._synthesize_react_thought(objective, str(t_name or ""), t_args if isinstance(t_args, dict) else {})
            elif (thought is None or str(thought).strip() == "") and isinstance(content, str) and content.strip():
                thought = content.strip()

            trace_step: dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent": self.name,
                "task_id": message.payload.get("scene_id", "unknown"),
                "turn": turns,
                "llm_latency_ms": round(llm_ms, 2),
                "assistant_raw_content": content,
                "thought": thought,
                "actions_in_text": actions,
                "tool_calls": [],
                "terminated": False
            }

            if not tool_calls:
                trace_step["terminated_no_tool_call"] = True
                trace_step["terminated"] = True
                if terminate_answer:
                    trace_step["terminate_answer"] = terminate_answer
                trace.append(trace_step)
                append_trace_line(trace_step)

                logger.info(
                    "[AGENT:%s | TURN %s | %sms] THOUGHT=%s TERMINATED=no_tools",
                    self.name,
                    turns,
                    round(llm_ms, 2),
                    json.dumps(thought) if thought else "",
                )
                break

            terminate_called = False
            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{turns}")
                fn = tc.get("function", {})
                t_name = fn.get("name")
                t_args_str = fn.get("arguments", "{}")
                try:
                    t_args = json.loads(t_args_str) if isinstance(t_args_str, str) else t_args_str
                except (json.JSONDecodeError, TypeError):
                    t_args = {}

                if (
                    isinstance(t_name, str)
                    and t_name.strip()
                    and str(t_name).strip().lower() != "terminate"
                    and allowed_tool_names
                    and str(t_name).strip() not in allowed_tool_names
                ):
                    emulated = self._emulate_unsupported_tool(t_name, t_args if isinstance(t_args, dict) else {})
                    if emulated is not None:
                        t_res = emulated
                    else:
                        t_res = {
                            "success": True,
                            "skipped_tool": str(t_name),
                            "warning": f"Tool {t_name} is not enabled for this agent/task. Choose one of the enabled tools.",
                            "observation_type": "unsupported_tool",
                        }
                    all_tool_calls_logged.append(
                        {
                            "tool": t_name,
                            "args": t_args if isinstance(t_args, dict) else {},
                            "normalized_args": t_args if isinstance(t_args, dict) else {},
                            "output": t_res,
                            "latency_ms": 0.0,
                            "turn": turns,
                        }
                    )
                    trace_step["tool_calls"].append(
                        {
                            "id": tc_id,
                            "tool": t_name,
                            "args": t_args if isinstance(t_args, dict) else {},
                            "normalized_args": t_args if isinstance(t_args, dict) else {},
                            "output": t_res,
                            "latency_ms": 0.0,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": str(t_name),
                            "content": json.dumps(t_res, ensure_ascii=False),
                        }
                    )
                    messages = self._trim_message_history(messages)
                    continue

                if str(t_name or "").strip().lower() == "terminate":
                    terminate_answer = self._extract_terminate_answer_from_actions(
                        [{"name": "Terminate", "arguments": t_args if isinstance(t_args, dict) else {}}]
                    )
                    term_output = {
                        "status": "terminated",
                        "final_answer": terminate_answer,
                        "success": True,
                    }
                    all_tool_calls_logged.append(
                        {
                            "tool": "Terminate",
                            "args": t_args if isinstance(t_args, dict) else {},
                            "normalized_args": t_args if isinstance(t_args, dict) else {},
                            "output": term_output,
                            "latency_ms": 0.0,
                            "turn": turns,
                        }
                    )
                    trace_step["tool_calls"].append(
                        {
                            "id": tc_id,
                            "tool": "Terminate",
                            "args": t_args if isinstance(t_args, dict) else {},
                            "normalized_args": t_args if isinstance(t_args, dict) else {},
                            "output": term_output,
                            "latency_ms": 0.0,
                        }
                    )
                    trace_step["terminated"] = True
                    trace_step["terminated_by_tool_call"] = True
                    if terminate_answer:
                        trace_step["terminate_answer"] = terminate_answer
                    terminate_called = True
                    logger.info(
                        "[AGENT:%s | TURN %s | %sms] THOUGHT=%s TERMINATED=tool_call",
                        self.name,
                        turns,
                        round(llm_ms, 2),
                        json.dumps(thought) if thought else "",
                    )
                    break

                tool_t0 = time.perf_counter()
                t_res, normalized_args = self._call_tool(t_name, t_args)
                tool_ms = (time.perf_counter() - tool_t0) * 1000.0
                all_tool_calls_logged.append({
                    "tool": t_name,
                    "args": t_args,
                    "normalized_args": normalized_args,
                    "output": t_res,
                    "latency_ms": round(tool_ms, 2),
                    "turn": turns,
                })
                trace_step["tool_calls"].append({
                    "id": tc_id,
                    "tool": t_name,
                    "args": t_args,
                    "normalized_args": normalized_args,
                    "output": t_res,
                    "latency_ms": round(tool_ms, 2),
                })

                res_str = json.dumps(t_res)
                if len(res_str) > 150:
                    res_str = res_str[:147] + "..."
                logger.info(
                    "[AGENT:%s | TURN %s | %sms] THOUGHT=%s TOOL=%s ARGS=%s TOOL_RES=%s TOOL_LATENCY_MS=%s",
                    self.name,
                    turns,
                    round(llm_ms, 2),
                    json.dumps(thought) if thought else "",
                    t_name,
                    json.dumps(t_args),
                    res_str,
                    round(tool_ms, 2),
                )

                tool_content = self._tool_result_for_llm(t_res)
                if isinstance(t_res, dict) and t_res.get("success") is False:
                    err_text = str(t_res.get("error") or "tool returned success=false")
                    tool_content = json.dumps(
                        {
                            "success": False,
                            "error": err_text,
                            "observation_type": "tool_error",
                        },
                        ensure_ascii=False,
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": t_name,
                    "content": tool_content
                })
                messages = self._trim_message_history(messages)
            trace.append(trace_step)
            append_trace_line(trace_step)
            if terminate_called:
                break

        # Final extraction
        final_content = response_msg.get("content", "")
        if final_content is None:
            final_content = ""
        clean_final_content = self._strip_thinking_tokens(final_content)
        output_dict = {"raw_output": final_content}

        # Try to parse structured output from the final response
        try:
            parsed_final = json.loads(clean_final_content)
            if isinstance(parsed_final, dict):
                output_dict.update(parsed_final)
        except (json.JSONDecodeError, TypeError):
            pass

        if not isinstance(output_dict.get("ans"), str) or not str(output_dict.get("ans", "")).strip():
            if terminate_answer:
                output_dict["ans"] = terminate_answer

        if not isinstance(output_dict.get("ans"), str) or not str(output_dict.get("ans", "")).strip():
            for tr in reversed(trace):
                actions = tr.get("actions_in_text") if isinstance(tr.get("actions_in_text"), list) else []
                ans = self._extract_terminate_answer_from_actions(actions)
                if ans:
                    output_dict["ans"] = ans
                    break

        # Preserve task metadata from payload
        for k in ("objective", "scene_id", "region"):
            if k in message.payload:
                output_dict.setdefault(k, message.payload[k])

        output_dict.setdefault("tool_call_count", len(all_tool_calls_logged))
        output_dict = self._augment_output_from_tools(output_dict, all_tool_calls_logged, message.payload)

        return AgentResult(
            output=output_dict,
            confidence=0.8,
            tool_calls=all_tool_calls_logged,
            n_turns=turns,
            trace=trace,
        )

    def _build_task_adaptive_system_prompt(self, payload: dict[str, Any]) -> str:
        prompt = self._get_system_prompt()
        task_description = str(payload.get("task_description", "")).strip()
        query_type = str(payload.get("query_type", "")).strip()
        if task_description:
            prompt += f"\n\nAssigned sub-task from ORC: {task_description}"
        if query_type:
            prompt += f"\nQuery type: {query_type}"
        wm_summary = str(payload.get("working_memory_summary", "")).strip()
        if wm_summary:
            prompt += f"\nWorking memory summary:\n{wm_summary}"
        upstream_notes = str(payload.get("upstream_notes", "")).strip()
        if upstream_notes:
            prompt += f"\nUpstream notes: {upstream_notes}"
        image_paths = payload.get("image_paths")
        if isinstance(image_paths, list) and image_paths:
            available = [str(p) for p in image_paths if str(p).strip()][:12]
            prompt += f"\nAvailable images for this task: {json.dumps(available, ensure_ascii=False)}"
            prompt += "\nUse these exact paths when calling vision tools. Do not fabricate image paths."
        enabled_tools = payload.get("enabled_tools")
        if isinstance(enabled_tools, list) and enabled_tools:
            tool_names = [str(t) for t in enabled_tools if str(t).strip()]
            prompt += f"\nEnabled tools for this turn: {json.dumps(tool_names, ensure_ascii=False)}"
            prompt += "\nOnly call tools from this enabled list."
        prompt += (
            "\n\nRESPONSE FORMAT (mandatory): "
            "Return exactly one JSON object with keys thought and actions. "
            "Example: {\"thought\":\"...\",\"actions\":[{\"name\":\"ToolName\",\"arguments\":{}}]} "
            "When finished, call Terminate as: "
            "{\"thought\":\"...\",\"actions\":[{\"name\":\"Terminate\",\"arguments\":{\"ans\":\"final answer\"}}]}. "
            "Do not use markdown code fences and do not add text outside JSON. "
            "Call exactly one tool per turn and never fabricate data."
        )
        if self.thinking_mode.lower() == "always":
            prompt += "\nAlways provide an explicit thought before any tool call."
        return prompt
