from __future__ import annotations

from typing import Any
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import json
import time
import logging
import os
from agents.vra import VisionReasoningAgent
from agents.ga import GeospatialAgent
from agents.pa import PlanningAgent
from agents.orc import Orchestrator
from framework.memory.episodic_memory import EpisodicMemory
from framework.memory.working_memory import WorkingMemory
from framework.memory.trajectory_log import TrajectoryLog
from framework.mpc.message import Message
from framework.mpc.broker import MessageBroker
from framework.protocols.compression import compress_context
from framework.protocols.safety import validate_safety_payload
from framework.protocols.deadlock import DeadlockDetector
from evaluation import metrics as eval_metrics

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)

class EpisodeRunner:
    _LEAKY_TURN_HINTS = (
        "observation:",
        "saved to line layer",
        "distance=",
        "travel_time=",
        "distances =",
        "travel_times =",
    )

    _ANSWER_ARG_KEYS = ("ans", "answer", "final_answer", "response")

    def __init__(
        self,
        model_id: str = "mock-model",
        base_url: str = "http://localhost:8002/v1",
        tool_server: str | None = None,
        memory: EpisodicMemory | None = None,
        trajectory_log: TrajectoryLog | None = None,
        agent_model_map: dict[str, str] | None = None,
        agent_base_url_map: dict[str, str] | None = None,
        allow_mock_fallback: bool = True,
        max_turns: int = 15,
    ) -> None:
        self.tool_server = tool_server or "http://localhost:9000"
        self.memory = memory or EpisodicMemory()
        self.trajectory_log = trajectory_log or TrajectoryLog()
        self.broker = MessageBroker()
        self._agents_cfg = self._load_yaml(Path("configs/agents.yaml"))
        self._models_cfg = self._load_yaml(Path("configs/models.yaml"))

        # All agents share the same model and tool server
        model_map = agent_model_map or {}
        url_map = agent_base_url_map or {}

        def _default_base_url_for(agent_name: str) -> str:
            model_cfg = self._models_cfg.get(agent_name, {}) if isinstance(self._models_cfg, dict) else {}
            cfg_base_url = str(model_cfg.get("base_url", "")).strip()
            if cfg_base_url:
                return cfg_base_url
            port = model_cfg.get("vllm_port")
            if isinstance(port, int):
                return f"http://127.0.0.1:{port}/v1"
            return base_url

        def _default_api_key_for(agent_name: str, resolved_base_url: str) -> str:
            model_cfg = self._models_cfg.get(agent_name, {}) if isinstance(self._models_cfg, dict) else {}
            api_key_env = str(model_cfg.get("api_key_env", "")).strip()
            if api_key_env:
                val = str(os.getenv(api_key_env, "")).strip()
                if val:
                    return val

            if "huggingface.co" in str(resolved_base_url).lower() or "hf.space" in str(resolved_base_url).lower():
                for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                    val = str(os.getenv(env_name, "")).strip()
                    if val:
                        return val

            return "sk-mock"

        def _mk(agent_name: str) -> dict[str, Any]:
            agent_cfg = self._agents_cfg.get(agent_name, {}) if isinstance(self._agents_cfg, dict) else {}
            model_cfg = self._models_cfg.get(agent_name, {}) if isinstance(self._models_cfg, dict) else {}
            configured_turns = int(agent_cfg.get("max_react_turns", max_turns))
            effective_turns = configured_turns if max_turns == 15 else max_turns
            resolved_base_url = url_map.get(agent_name, _default_base_url_for(agent_name))
            return {
                "model_id": model_map.get(agent_name, model_cfg.get("model_id", model_id)),
                "base_url": resolved_base_url,
                "api_key": _default_api_key_for(agent_name, resolved_base_url),
                "tool_server": self.tool_server,
                "allow_mock_fallback": allow_mock_fallback,
                "max_turns": int(effective_turns),
                "thinking_mode": str(model_cfg.get("thinking_mode", "selective")),
                "model_max_context_tokens": int(model_cfg.get("max_model_len", 32768)),
            }

        self.vra = VisionReasoningAgent(**_mk("vra"))
        self.ga = GeospatialAgent(**_mk("ga"))
        self.pa = PlanningAgent(**_mk("pa"))
        self.orc = Orchestrator(**_mk("orc"))
        self.deadlock_detector = DeadlockDetector()
        self.results_root = Path("results/agent_runs")
        self.results_root.mkdir(parents=True, exist_ok=True)
        self._message_trace: list[dict[str, Any]] = []
        self._orc_backlog: list[Message] = []
        self._flow_trace: list[dict[str, Any]] = []
        self._flow_seq: int = 0
        self._orc_trace: list[dict[str, Any]] = []
        self._orc_turn: int = 0
        orc_cfg = self._agents_cfg.get("orc", {}) if isinstance(self._agents_cfg, dict) else {}
        self.ack_timeout_seconds = int(orc_cfg.get("deadlock_timeout_seconds", 30))
        self.deadlock_detector_mode = str(orc_cfg.get("deadlock_detector_mode", "parallel_only")).strip().lower()
        self.deadlock_detector_enabled = False
        self.em_soft_limit_tokens = int(orc_cfg.get("episodic_memory_soft_limit_tokens", 32000))
        self.default_parallel_vra_ga = False
        self.max_agent_retries = 2
        self.retry_backoff_base_seconds = 1.0
        self.working_memory = {
            "vra": WorkingMemory(token_limit=int(self._agents_cfg.get("vra", {}).get("working_memory_tokens", 4096))),
            "ga": WorkingMemory(token_limit=int(self._agents_cfg.get("ga", {}).get("working_memory_tokens", 4096))),
            "pa": WorkingMemory(token_limit=int(self._agents_cfg.get("pa", {}).get("working_memory_tokens", 4096))),
            "orc": WorkingMemory(token_limit=int(self._agents_cfg.get("orc", {}).get("working_memory_tokens", 4096))),
        }

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if yaml is None or not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _tool_names(self, tools_schema: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for tool in tools_schema:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name")
            if isinstance(name, str):
                out.append(name)
        return out

    def _estimate_tokens(self, payload: dict[str, Any]) -> int:
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except TypeError:
            text = str(payload)
        return max(1, len(text) // 4)

    def _build_message(
        self,
        *,
        msg_type: str,
        sender: str,
        recipient: str,
        payload: dict[str, Any],
        thread_id: str,
        parent_msg_id: str | None = None,
        confidence: float = 1.0,
        epoch_ref: int | None = None,
        priority: int = 1,
        ttl_ms: int = 60_000,
    ) -> Message:
        now_ms = int(time.time() * 1000)
        return Message(
            message_type=msg_type,
            sender=sender,
            recipient=recipient,
            payload=payload,
            priority=priority,
            ttl_seconds=max(1, int(round(ttl_ms / 1000.0))),
            thread_id=thread_id,
            parent_msg_id=parent_msg_id,
            confidence=confidence,
            epoch_ref=epoch_ref,
            token_count=self._estimate_tokens(payload),
            timestamp_ms=now_ms,
            ttl_ms=ttl_ms,
        )

    def _record_message_event(self, event: str, msg: Message) -> None:
        row = {
            "event": event,
            "message": msg.to_dict(),
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        self._message_trace.append(row)
        self.trajectory_log.append("orc", row)

    def _publish(self, msg: Message) -> None:
        self.broker.publish(msg)
        self._record_message_event("mpc_publish", msg)

    def _drain_for(self, recipient: str) -> list[Message]:
        msgs = self.broker.drain_for(recipient)
        for m in msgs:
            self._record_message_event("mpc_deliver", m)
        return msgs

    def _collect_orc_mailbox(self) -> list[Message]:
        msgs = list(self._orc_backlog)
        self._orc_backlog.clear()
        msgs.extend(self._drain_for("orc"))
        return msgs

    def _deadlock_detection_active(self) -> bool:
        if self.deadlock_detector_mode == "off":
            return False
        if self.deadlock_detector_mode == "always":
            return True
        return bool(self.deadlock_detector_enabled)

    def _add_wait_edge(self, source: str, target: str) -> None:
        if self._deadlock_detection_active():
            self.deadlock_detector.add_wait(source, target)

    def _remove_wait_edge(self, source: str, target: str) -> None:
        if self._deadlock_detection_active():
            self.deadlock_detector.remove_wait(source, target)

    def _emit_ack(self, task_msg: Message, agent_name: str) -> None:
        ack = self._build_message(
            msg_type="ACK",
            sender=agent_name,
            recipient="orc",
            payload={"acknowledged_msg_id": task_msg.msg_id},
            thread_id=task_msg.thread_id,
            parent_msg_id=task_msg.msg_id,
            confidence=1.0,
            epoch_ref=task_msg.epoch_ref,
            priority=1,
            ttl_ms=30_000,
        )
        self._publish(ack)

    def _await_ack(self, task_msg: Message, agent_name: str) -> bool:
        deadline_ms = int(time.time() * 1000) + self.ack_timeout_seconds * 1000
        while int(time.time() * 1000) <= deadline_ms:
            inbox = self._drain_for("orc")
            if not inbox:
                break
            for msg in inbox:
                if (
                    msg.msg_type == "ACK"
                    and msg.sender == agent_name
                    and msg.payload.get("acknowledged_msg_id") == task_msg.msg_id
                ):
                    self._remove_wait_edge("orc", agent_name)
                    return True
                self._orc_backlog.append(msg)
        return False

    def _persist_agent_output(self, agent_name: str, output: dict[str, Any], epoch_ref: int | None) -> None:
        self.memory.write(f"{agent_name}_result", output)
        if epoch_ref is not None:
            self.memory.write(f"{agent_name}_result_T{epoch_ref}", output)
        for k, v in output.items():
            self.memory.write(f"{agent_name}_{k}", v)

    def _update_working_memory(self, agent_name: str, summary: str) -> None:
        wm = self.working_memory.get(agent_name)
        if wm is None:
            return
        snapshot = wm.view()
        flushed = wm.append(summary)
        if flushed:
            text = snapshot
            if summary:
                text = f"{snapshot}\n{summary}" if snapshot else summary
            if text:
                self.memory.write(f"{agent_name}_working_memory_flush", text)

    def _detect_error_result(self, result: Any) -> tuple[bool, str | None]:
        output = result.output if hasattr(result, "output") and isinstance(result.output, dict) else {}
        ans_raw = str(output.get("ans", "")).strip()
        ans = ans_raw.lower()
        if ans in {"error", "timeout", "mock result"}:
            return True, f"ans={ans}"
        has_usable_answer = bool(ans_raw)

        thought = str(output.get("thought", "")).lower()
        raw = str(output.get("raw_output", "")).lower()
        if any(k in thought for k in ("llm call failed", "http error", "context overflow")):
            return True, output.get("thought")
        if any(k in raw for k in ("llm call failed", "http error", "context overflow")):
            return True, output.get("raw_output")

        tool_calls = getattr(result, "tool_calls", []) if hasattr(result, "tool_calls") else []
        tool_error_reason: str | None = None
        for c in tool_calls if isinstance(tool_calls, list) else []:
            out = c.get("output", {}) if isinstance(c, dict) else {}
            if isinstance(out, dict) and out.get("error"):
                tool_error_reason = str(out.get("error"))
                break
            if isinstance(out, dict) and out.get("success") is False:
                tool_error_reason = str(out.get("error") or "tool_success_false")
                break

        if tool_error_reason:
            if has_usable_answer:
                return False, None
            return True, tool_error_reason

        if not tool_calls and not output.get("ans") and not output.get("raw_output"):
            return True, "empty_result"

        return False, None

    def _summarize_context_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if "boundary_wkt" in value and isinstance(value.get("boundary_wkt"), str):
                wkt = str(value.get("boundary_wkt", ""))
                return {
                    "boundary_summary": {
                        "wkt_chars": len(wkt),
                        "gpkg_path": value.get("gpkg_path"),
                    }
                }
            if "pois" in value and isinstance(value.get("pois"), list):
                pois = value.get("pois", [])
                sample = []
                for item in pois[:5]:
                    if isinstance(item, dict):
                        sample.append(
                            {
                                "name": item.get("name"),
                                "lat": item.get("lat"),
                                "lon": item.get("lon"),
                            }
                        )
                return {"pois_count": len(pois), "pois_sample": sample}
            out: dict[str, Any] = {}
            for k, v in value.items():
                if k in {"raw_output", "trace", "tool_calls"}:
                    continue
                out[str(k)] = self._summarize_context_value(v)
            return out
        if isinstance(value, list):
            if len(value) > 8:
                return {"list_len": len(value), "sample": [self._summarize_context_value(v) for v in value[:3]]}
            return [self._summarize_context_value(v) for v in value]
        if isinstance(value, str) and len(value) > 240:
            return value[:240] + "..."
        return value

    def _episodic_context_for_agent(self, keys: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        items = self.memory.items()
        for k in keys:
            if k in items:
                out[k] = self._summarize_context_value(items[k])
        return out

    def _run_agent_from_task(
        self,
        *,
        agent_name: str,
        agent,
        task_msg: Message,
        tools_schema: list[dict[str, Any]],
    ) -> tuple[Any, float, bool, str | None]:
        # Single-process mode: record synthetic readiness instead of self-pub/sub ACK.
        synthetic_ack = self._build_message(
            msg_type="ACK",
            sender=agent_name,
            recipient="orc",
            payload={"acknowledged_msg_id": task_msg.msg_id, "mode": "single_process_synthetic"},
            thread_id=task_msg.thread_id,
            parent_msg_id=task_msg.msg_id,
            confidence=1.0,
            epoch_ref=task_msg.epoch_ref,
            priority=1,
            ttl_ms=30_000,
        )
        self._record_message_event("mpc_ack_synthetic", synthetic_ack)
        self._remove_wait_edge("orc", agent_name)

        t0 = time.perf_counter()
        result = agent.handle_task(task_msg, tools_schema=tools_schema)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        error_result, error_reason = self._detect_error_result(result)

        if hasattr(result, "output") and isinstance(result.output, dict):
            self._persist_agent_output(agent_name, result.output, task_msg.epoch_ref)
            self._update_working_memory(agent_name, self._preview(result.output, max_len=600))

        result_payload = {
            "output_type": "agent_result",
            "data_path_or_inline": result.output,
            "quality_score": float(getattr(result, "confidence", 0.0)),
            "reasoning_summary": str(result.output.get("raw_output", ""))[:500],
            "error_result": error_result,
            "error_reason": error_reason,
        }
        result_msg = self._build_message(
            msg_type="RESULT",
            sender=agent_name,
            recipient="orc",
            payload=result_payload,
            thread_id=task_msg.thread_id,
            parent_msg_id=task_msg.msg_id,
            confidence=float(getattr(result, "confidence", 0.0)),
            epoch_ref=task_msg.epoch_ref,
            priority=1,
            ttl_ms=120_000,
        )
        self._publish(result_msg)
        return result, elapsed_ms, error_result, error_reason

    def _context_keys_for(self, agent_name: str) -> list[str]:
        keys = sorted(self.memory.items().keys())
        if agent_name == "vra":
            return [k for k in keys if "episode_metadata" in k][:8]
        if agent_name == "ga":
            preferred = [
                "episode_metadata",
                "vra_damage_polygons",
                "vra_flood_extent",
                "vra_result",
            ]
            selected = [k for k in keys if any(p in k for p in preferred)]
            return selected[:12]
        if agent_name == "pa":
            preferred = [
                "episode_metadata",
                "vra_damage_polygons",
                "vra_flood_extent",
                "ga_road_scores",
                "ga_accessibility",
                "conflict_resolutions",
                "vra_result",
                "ga_result",
            ]
            selected = [k for k in keys if any(p in k for p in preferred)]
            dynamic = [k for k in keys if k.startswith(("vra_", "ga_"))]
            for k in dynamic:
                if k not in selected:
                    selected.append(k)
            if not selected:
                selected = [k for k in keys if k.startswith(("vra_", "ga_", "episode_metadata"))]
            return selected
        return []

    def _task_payload(
        self,
        *,
        agent_name: str,
        task: dict[str, Any],
        task_description: str,
        tools_schema_subset: list[str],
        em_context_keys: list[str],
        episodic_context: dict[str, Any],
        query_type: str,
        deadline_ms: int,
    ) -> dict[str, Any]:
        payload = {
            "task_description": task_description,
            "tool_schema_subset": tools_schema_subset,
            "em_context_keys": em_context_keys,
            "episodic_context": episodic_context,
            "query_type": query_type,
            "deadline_ms": deadline_ms,
            # Keep legacy keys expected by existing agent output extraction.
            "objective": task.get("objective"),
            "scene_id": task.get("scene_id"),
            "region": task.get("region"),
            "benchmark": task.get("benchmark"),
            "dataset_row_id": task.get("dataset_row_id"),
            "source": task.get("source"),
        }
        if "epoch_ref" in task:
            payload["epoch_ref"] = task["epoch_ref"]
        if isinstance(task.get("human_inputs"), list):
            inputs = [str(x) for x in task.get("human_inputs", []) if str(x).strip()]
            payload["human_inputs"] = inputs[:1]
        if isinstance(task.get("image_paths"), list):
            payload["image_paths"] = [str(x) for x in task.get("image_paths", []) if str(x).strip()]
        wm = self.working_memory.get(agent_name)
        if wm is not None:
            wm_text = wm.view()
            if isinstance(wm_text, str) and wm_text.strip():
                payload["working_memory_summary"] = wm_text[-1600:]

        if agent_name == "pa":
            ga_result = self.memory.items().get("ga_result")
            has_ga_distance = isinstance(ga_result, dict) and isinstance(ga_result.get("distance_meters"), (int, float))
            has_ga_pair = isinstance(ga_result, dict) and isinstance(ga_result.get("closest_pair"), dict)
            has_ga_pois = isinstance(ga_result, dict) and (
                isinstance(ga_result.get("fire_pois"), list)
                or isinstance(ga_result.get("police_pois"), list)
            )
            if not (has_ga_distance or has_ga_pair or has_ga_pois):
                payload["upstream_notes"] = "GA produced no valid output for this query."
        return payload

    def _decompose_task(self, task: dict[str, Any]) -> dict[str, str]:
        objective = task.get("objective", "analyze disaster scene")
        return {
            "vra": f"Run visual-temporal evidence extraction for: {objective}",
            "ga": f"Run geospatial and road-condition analysis for: {objective}",
            "pa": f"Generate safe evacuation/supply routes for: {objective}",
        }

    def _resolve_agent_sequence(self, task: dict[str, Any]) -> list[str]:
        allowed = ["vra", "ga", "pa"]
        requested = task.get("agent_sequence")
        seq: list[str] = []
        if isinstance(requested, list):
            for item in requested:
                name = str(item).lower().strip()
                if name in allowed and name not in seq:
                    seq.append(name)
        if not seq:
            seq = ["vra", "ga", "pa"]
        return seq

    def _allow_parallel_vra_ga(self, task: dict[str, Any]) -> bool:
        # Parallel VRA+GA is opt-in and only allowed when caller asserts disjoint subtasks.
        requested_parallel = bool(task.get("parallel_vra_ga", self.default_parallel_vra_ga))
        disjoint = bool(task.get("disjoint_subtasks", False))
        return requested_parallel and disjoint

    def _preview(self, value: Any, max_len: int = 240) -> str:
        try:
            txt = json.dumps(value, ensure_ascii=False)
        except TypeError:
            txt = str(value)
        if len(txt) > max_len:
            return txt[: max_len - 3] + "..."
        return txt

    def _clean_turn_text(self, value: Any) -> str:
        text = str(value or "").replace("<AGENT_PROMPT>", " ").strip()
        return " ".join(text.split())

    def _is_leaky_turn(self, text: str) -> bool:
        lowered = text.lower()
        return any(hint in lowered for hint in self._LEAKY_TURN_HINTS)

    def _pick_primary_question(self, task: dict[str, Any]) -> str:
        objective = self._clean_turn_text(task.get("objective", ""))
        candidates: list[str] = []
        if objective:
            candidates.append(objective)

        human_inputs = task.get("human_inputs")
        if isinstance(human_inputs, list):
            for item in human_inputs:
                cleaned = self._clean_turn_text(item)
                if cleaned:
                    candidates.append(cleaned)

        if not candidates:
            return ""

        for text in candidates:
            if not self._is_leaky_turn(text):
                return text

        return candidates[0]

    def _sanitize_task(self, task: dict[str, Any]) -> dict[str, Any]:
        safe_task = dict(task or {})
        primary_question = self._pick_primary_question(safe_task)

        if primary_question:
            safe_task["objective"] = primary_question
            safe_task["human_inputs"] = [primary_question]
        elif "human_inputs" in safe_task:
            safe_task["human_inputs"] = []

        for key in (
            "ground_truth",
            "gt_answer",
            "answer",
            "expected_answer",
            "reference_answer",
        ):
            safe_task.pop(key, None)

        return safe_task

    def _sanitize_logged_action(self, action: dict[str, Any]) -> dict[str, Any]:
        name = str(action.get("name", ""))
        args = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
        sanitized_args = dict(args)

        return {
            "name": name,
            "arguments": sanitized_args,
        }

    def _record_flow_event(self, event: str, payload: dict[str, Any]) -> None:
        self._flow_seq += 1
        self._flow_trace.append(
            {
                "seq": self._flow_seq,
                "ts": datetime.utcnow().isoformat() + "Z",
                "event": event,
                **payload,
            }
        )

    def _record_orc_reason(
        self,
        *,
        task: dict[str, Any],
        stage: str,
        thought: str,
        decision: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        terminated: bool = False,
    ) -> None:
        self._orc_turn += 1
        row = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": "orc",
            "task_id": task.get("scene_id", "unknown"),
            "turn": self._orc_turn,
            "stage": stage,
            "thought": thought,
            "decision": decision or {},
            "context": context or {},
            "tool_calls": [],
            "terminated": terminated,
        }
        self._orc_trace.append(row)
        self._record_flow_event(
            "orc_reason",
            {
                "turn": row["turn"],
                "stage": stage,
                "thought": thought,
                "decision": row["decision"],
                "terminated": terminated,
            },
        )

    def _record_agent_flow(
        self,
        *,
        agent_name: str,
        task_msg: Message,
        result,
        elapsed_ms: float,
    ) -> None:
        turns: list[dict[str, Any]] = []
        for tr in result.trace:
            calls = tr.get("tool_calls", []) if isinstance(tr.get("tool_calls"), list) else []
            turns.append(
                {
                    "turn": tr.get("turn"),
                    "thought": tr.get("thought"),
                    "terminated": tr.get("terminated", False),
                    "tool_calls": [
                        {
                            "tool": c.get("tool"),
                            "args": c.get("args", {}),
                            "latency_ms": c.get("latency_ms"),
                            "output_preview": self._preview(c.get("output")),
                        }
                        for c in calls
                    ],
                }
            )

        self._record_flow_event(
            "agent_run_complete",
            {
                "agent": agent_name,
                "task_msg_id": task_msg.msg_id,
                "task_description": task_msg.payload.get("task_description"),
                "elapsed_ms": round(elapsed_ms, 2),
                "n_turns": result.n_turns,
                "n_tool_calls": len(result.tool_calls),
                "turns": turns,
                "final_output_preview": self._preview(result.output),
            },
        )

    def _build_flow_log(
        self,
        *,
        thread_id: str,
        task: dict[str, Any],
        execution_mode: str,
        agent_sequence: list[str],
    ) -> dict[str, Any]:
        return {
            "thread_id": thread_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "execution_mode": execution_mode,
            "agent_sequence": agent_sequence,
            "task": {
                "scene_id": task.get("scene_id"),
                "region": task.get("region"),
                "objective": task.get("objective"),
            },
            "events": self._flow_trace,
        }

    def _planned_sequence_items(
        self,
        *,
        plan: dict[str, str],
        agent_sequence: list[str],
    ) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for agent in agent_sequence:
            items.append(
                {
                    "agent": agent,
                    "sub_task": str(plan.get(agent, "")),
                }
            )
        return items

    def _sequence_from_flow(self, event_name: str) -> list[str]:
        seq: list[str] = []
        for ev in self._flow_trace:
            if ev.get("event") == event_name and isinstance(ev.get("agent"), str):
                seq.append(ev["agent"])
        return seq

    def _compute_plan_follow_metrics(
        self,
        planned_sequence: list[str],
        dispatched_sequence: list[str],
        executed_sequence: list[str],
    ) -> dict[str, Any]:
        n_planned = max(1, len(planned_sequence))
        order_matches = sum(
            1
            for i, agent in enumerate(planned_sequence)
            if i < len(dispatched_sequence) and dispatched_sequence[i] == agent
        )
        order_score = order_matches / n_planned
        coverage_score = (
            sum(1 for agent in planned_sequence if agent in dispatched_sequence) / n_planned
        )

        loop_count = sum(max(0, dispatched_sequence.count(a) - 1) for a in set(dispatched_sequence))
        loop_penalty = min(0.5, loop_count / n_planned)
        plan_accuracy = max(0.0, min(1.0, (0.7 * order_score + 0.3 * coverage_score) - loop_penalty))

        off_plan_steps = [a for a in dispatched_sequence if a not in planned_sequence]
        return {
            "plan_accuracy": round(plan_accuracy, 4),
            "plan_followed": dispatched_sequence == planned_sequence,
            "plan_loop_count": int(loop_count),
            "off_plan_steps": off_plan_steps,
            "planned_sequence": planned_sequence,
            "dispatched_sequence": dispatched_sequence,
            "executed_sequence": executed_sequence,
        }

    def _build_agents_output(
        self,
        *,
        task: dict[str, Any],
        planned_items: list[dict[str, str]],
        dispatched_sequence: list[str],
        results_by_agent: dict[str, Any],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        # Keep a single primary human question to avoid benchmark answer leakage.
        human_inputs = task.get("human_inputs", [])
        if isinstance(human_inputs, list) and human_inputs:
            out.append({"from": "human", "value": str(human_inputs[0])})
        elif isinstance(task.get("objective"), str) and task.get("objective", "").strip():
            out.append({"from": "human", "value": str(task.get("objective", ""))})

        # ORC explicit plan action (requested by user).
        out.append(
            {
                "from": "gpt",
                "agent": "orc",
                "value": json.dumps(
                    {
                        "actions": [
                            {
                                "name": "plan",
                                "arguments": {
                                    "sequence": planned_items,
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        )

        for tr in self._orc_trace:
            out.append(
                {
                    "from": "gpt",
                    "agent": "orc",
                    "turn": tr.get("turn"),
                    "value": json.dumps(
                        {
                            "thought": tr.get("thought"),
                            "actions": [
                                {
                                    "name": "orchestrate",
                                    "arguments": tr.get("decision", {}),
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        for agent_name in dispatched_sequence:
            result = results_by_agent.get(agent_name)
            if result is None:
                continue
            for tr in result.trace:
                actions: list[dict[str, Any]] = []
                seen_action_keys: set[tuple[str, str]] = set()
                turn_tool_calls = tr.get("tool_calls", []) if isinstance(tr.get("tool_calls"), list) else []
                observation_rows: list[dict[str, Any]] = []

                for tc in turn_tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    sanitized = self._sanitize_logged_action(
                        {
                            "name": tc.get("tool"),
                            "arguments": tc.get("args", {}) if isinstance(tc.get("args"), dict) else {},
                        }
                    )
                    name_norm = str(sanitized.get("name", "")).strip().lower()
                    args_key = json.dumps(sanitized.get("arguments", {}), sort_keys=True, ensure_ascii=False)
                    key = (name_norm, args_key)
                    if name_norm and key not in seen_action_keys:
                        actions.append(sanitized)
                        seen_action_keys.add(key)

                    observation_payload = {
                        "observation": tc.get("output"),
                        "latency_ms": tc.get("latency_ms"),
                    }
                    observation_rows.append(
                        {
                            "from": "observation",
                            "agent": agent_name,
                            "turn": tr.get("turn"),
                            "tool": tc.get("tool"),
                            "value": json.dumps(observation_payload, ensure_ascii=False),
                        }
                    )

                for act in tr.get("actions_in_text", []) if isinstance(tr.get("actions_in_text"), list) else []:
                    if not isinstance(act, dict) or not act.get("name"):
                        continue
                    sanitized = self._sanitize_logged_action(
                        {
                            "name": act.get("name"),
                            "arguments": act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {},
                        }
                    )
                    name_norm = str(sanitized.get("name", "")).strip().lower()
                    args_key = json.dumps(sanitized.get("arguments", {}), sort_keys=True, ensure_ascii=False)
                    key = (name_norm, args_key)
                    if name_norm and key not in seen_action_keys:
                        actions.append(sanitized)
                        seen_action_keys.add(key)

                if not actions:
                    continue

                out.append(
                    {
                        "from": "gpt",
                        "agent": agent_name,
                        "turn": tr.get("turn"),
                        "value": json.dumps(
                            {
                                "thought": tr.get("thought"),
                                "actions": actions,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                out.extend(observation_rows)

        return out

    def _write_verbose_agents_output_log(self, run_dir: Path, agents_output: list[dict[str, Any]]) -> Path:
        path = run_dir / "agents_verbose_ordered.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, row in enumerate(agents_output, start=1):
                rec = {"seq": i, **row}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def _next_run_dir(self) -> Path:
        nums: list[int] = []
        for p in self.results_root.glob("run_*"):
            if p.is_dir():
                try:
                    nums.append(int(p.name.split("_")[-1]))
                except ValueError:
                    continue
        nxt = (max(nums) + 1) if nums else 1
        run_dir = self.results_root / f"run_{nxt:03d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _save_run_bundle(self, run_dir: Path, payload: dict[str, Any]) -> None:
        (run_dir / "bundle.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        md = [
            f"# MAGRF 4-Agent Run {run_dir.name}",
            "",
            f"- created_at: {payload.get('created_at')}",
            f"- objective: {payload.get('task', {}).get('objective')}",
            "",
            "## Agent Timings",
        ]
        for ag, meta in payload.get("agent_timings", {}).items():
            if isinstance(meta, dict):
                md.append(f"- {ag}: {meta.get('elapsed_ms')} ms")
            else:
                md.append(f"- {ag}: {meta} ms")
        md.append("")
        md.append("## Agent Tool Calls")
        for ag, calls in payload.get("agent_tool_calls", {}).items():
            md.append(f"- {ag}: {len(calls)} calls")
        md.append("")
        (run_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    def run_single_episode(self, task: dict[str, Any]) -> dict[str, Any]:
        task = self._sanitize_task(task)
        run_dir = self._next_run_dir()
        run_started = time.perf_counter()
        thread_id = str(uuid4())
        self._message_trace = []
        self._orc_backlog = []
        self._flow_trace = []
        self._flow_seq = 0
        self._orc_trace = []
        self._orc_turn = 0

        # Reset per-episode token accounting for all agents.
        for _agent in (self.orc, self.vra, self.ga, self.pa):
            if hasattr(_agent, "_reset_token_usage"):
                _agent._reset_token_usage()

        # Avoid leaking prior episode state into the current task.
        self.memory.set_items({}, 1.0)

        episode_metadata = {
            "thread_id": thread_id,
            "objective": task.get("objective"),
            "scene_id": task.get("scene_id"),
            "region": task.get("region"),
            "started_at": datetime.utcnow().isoformat() + "Z",
        }
        self.memory.write("episode_metadata", episode_metadata)

        now_ms = int(time.time() * 1000)
        deadline_ms = now_ms + 180_000

        available_agents = self._resolve_agent_sequence(task)
        plan_payload = self.orc.plan(
            objective=str(task.get("objective", "")),
            available_agents=available_agents,
            context={"scene_id": task.get("scene_id"), "region": task.get("region")},
        )
        query_type = str(plan_payload.get("query_type", "mixed_analysis"))
        agent_sequence = [a for a in plan_payload.get("agent_sequence", []) if a in {"vra", "ga", "pa"}]
        if not agent_sequence:
            agent_sequence = available_agents

        sub_tasks = plan_payload.get("sub_tasks", {}) if isinstance(plan_payload.get("sub_tasks"), dict) else {}
        plan = {
            "vra": str(sub_tasks.get("vra", f"Run visual-temporal evidence extraction for: {task.get('objective', '')}")),
            "ga": str(sub_tasks.get("ga", f"Run geospatial analysis for: {task.get('objective', '')}")),
            "pa": str(sub_tasks.get("pa", f"Run planning and distance analysis for: {task.get('objective', '')}")),
        }

        agent_registry: dict[str, dict[str, Any]] = {
            "vra": {"agent": self.vra, "tools": self.vra.VRA_TOOLS, "plan": plan["vra"]},
            "ga": {"agent": self.ga, "tools": self.ga.GA_TOOLS, "plan": plan["ga"]},
            "pa": {"agent": self.pa, "tools": self.pa.PA_TOOLS, "plan": plan["pa"]},
        }
        planned_items = self._planned_sequence_items(plan=plan, agent_sequence=agent_sequence)
        parallel_vra_ga = self._allow_parallel_vra_ga(task)
        execution_mode = "parallel_vra_ga" if parallel_vra_ga else "sequential"
        self.deadlock_detector_enabled = execution_mode == "parallel_vra_ga"

        self._record_flow_event(
            "deadlock_detector_mode",
            {
                "configured_mode": self.deadlock_detector_mode,
                "active": self._deadlock_detection_active(),
                "execution_mode": execution_mode,
            },
        )

        self._record_flow_event(
            "orc_decompose",
            {
                "thread_id": thread_id,
                "execution_mode": execution_mode,
                "agent_sequence": agent_sequence,
                "plan": plan,
                "plan_sequence": planned_items,
            },
        )
        self._record_orc_reason(
            task=task,
            stage="decompose",
            thought=(
                f"Decomposed objective into {len(agent_sequence)} sub-tasks and selected "
                f"execution mode '{execution_mode}'."
            ),
            decision={
                "agent_sequence": agent_sequence,
                "execution_mode": execution_mode,
                "query_type": query_type,
            },
            context={
                "plan_sequence": planned_items,
                "planner_rationale": str(plan_payload.get("rationale", "")),
            },
        )

        results_by_agent: dict[str, Any] = {}
        timings_by_agent: dict[str, float] = {}
        result_msg_count = 0
        agent_errors: dict[str, dict[str, Any]] = {}

        def _dispatch_task(agent_name: str, sync_from: str | None = None) -> Message:
            if sync_from is not None:
                sync_msg = self._build_message(
                    msg_type="SYNC",
                    sender="orc",
                    recipient=agent_name,
                    payload={"barrier_id": f"{thread_id}:{sync_from}_done"},
                    thread_id=thread_id,
                    confidence=1.0,
                    epoch_ref=task.get("epoch_ref"),
                    priority=2,
                    ttl_ms=120_000,
                )
                self._publish(sync_msg)
                self._record_flow_event(
                    "orc_sync",
                    {
                        "from_agent": sync_from,
                        "to_agent": agent_name,
                        "barrier_id": sync_msg.payload.get("barrier_id"),
                    },
                )

            self._record_orc_reason(
                task=task,
                stage="dispatch",
                thought=(
                    f"Dispatching {agent_name.upper()} with focused episodic context and awaiting ACK/RESULT "
                    "before continuing orchestration."
                ),
                decision={"dispatch_to": agent_name, "sync_from": sync_from},
                context={"context_keys": self._context_keys_for(agent_name)},
            )

            reg = agent_registry[agent_name]
            selected_keys = self._context_keys_for(agent_name)
            episodic_context = self._episodic_context_for_agent(selected_keys)
            msg = self._build_message(
                msg_type="TASK",
                sender="orc",
                recipient=agent_name,
                payload=self._task_payload(
                    agent_name=agent_name,
                    task=task,
                    task_description=reg["plan"],
                    tools_schema_subset=self._tool_names(reg["tools"]),
                    em_context_keys=selected_keys,
                    episodic_context=episodic_context,
                    query_type=query_type,
                    deadline_ms=deadline_ms,
                ),
                thread_id=thread_id,
                confidence=1.0,
                epoch_ref=task.get("epoch_ref"),
                priority=1,
                ttl_ms=180_000,
            )
            self._publish(msg)
            self._record_flow_event(
                "orc_dispatch_task",
                {
                    "agent": agent_name,
                    "task_msg_id": msg.msg_id,
                    "task_description": msg.payload.get("task_description"),
                    "context_keys": msg.payload.get("em_context_keys", []),
                },
            )

            self._add_wait_edge("orc", agent_name)
            inbox = self._drain_for(agent_name)
            task_msgs = [m for m in inbox if m.msg_type == "TASK"]
            if not task_msgs:
                raise RuntimeError(f"Failed to route TASK message to {agent_name}")
            return task_msgs[0]

        def _execute_task(agent_name: str, task_msg: Message) -> dict[str, Any]:
            nonlocal result_msg_count
            reg = agent_registry[agent_name]
            result, elapsed_ms, error_result, error_reason = self._run_agent_from_task(
                agent_name=agent_name,
                agent=reg["agent"],
                task_msg=task_msg,
                tools_schema=reg["tools"],
            )
            results_by_agent[agent_name] = result
            timings_by_agent[agent_name] = elapsed_ms
            agent_errors[agent_name] = {
                "error_result": bool(error_result),
                "error_reason": error_reason,
                "turns": int(getattr(result, "n_turns", 0)),
                "tool_calls": int(len(getattr(result, "tool_calls", []))),
            }
            self._record_agent_flow(
                agent_name=agent_name,
                task_msg=task_msg,
                result=result,
                elapsed_ms=elapsed_ms,
            )

            if error_result:
                logger.warning(
                    "agent_error agent=%s reason=%s turns=%s tool_calls=%s",
                    agent_name,
                    error_reason,
                    int(getattr(result, "n_turns", 0)),
                    int(len(getattr(result, "tool_calls", []))),
                )
                self._record_flow_event(
                    "agent_error",
                    {
                        "agent": agent_name,
                        "error_reason": error_reason,
                        "turns": int(getattr(result, "n_turns", 0)),
                        "tool_calls": int(len(getattr(result, "tool_calls", []))),
                    },
                )
                self.trajectory_log.append(
                    "orc",
                    {
                        "event": "agent_error",
                        "thread_id": task_msg.thread_id,
                        "agent": agent_name,
                        "error_reason": error_reason,
                    },
                )

            mail = self._collect_orc_mailbox()
            keep: list[Message] = []
            found_result = False
            for msg in mail:
                if msg.msg_type == "RESULT" and msg.sender == agent_name:
                    found_result = True
                    result_msg_count += 1
                    self._record_flow_event(
                        "orc_receive_result",
                        {
                            "agent": agent_name,
                            "result_msg_id": msg.msg_id,
                            "confidence": msg.confidence,
                            "quality_score": msg.payload.get("quality_score"),
                        },
                    )
                else:
                    keep.append(msg)
            self._orc_backlog.extend(keep)

            if not found_result:
                self._record_flow_event(
                    "orc_missing_result",
                    {
                        "agent": agent_name,
                        "note": "No RESULT message found immediately after execution.",
                    },
                )
                self._record_orc_reason(
                    task=task,
                    stage="post_result_missing",
                    thought=(
                        f"{agent_name.upper()} execution finished but RESULT message was not observed immediately; "
                        "keeping mailbox backlog and proceeding with caution."
                    ),
                    decision={"status": "wait_or_continue", "agent": agent_name},
                )
            else:
                remaining = [a for a in agent_sequence if a not in results_by_agent]
                next_step = remaining[0] if remaining else "merge"
                self._record_orc_reason(
                    task=task,
                    stage="post_result",
                    thought=(
                        f"Received {agent_name.upper()} RESULT (turns={result.n_turns}, tools={len(result.tool_calls)}, "
                        f"error={error_result}). Next orchestration step: {next_step.upper() if isinstance(next_step, str) else next_step}."
                    ),
                    decision={
                        "received_from": agent_name,
                        "next_step": next_step,
                        "remaining_agents": remaining,
                        "error_result": error_result,
                        "error_reason": error_reason,
                    },
                    context={"result_preview": self._preview(result.output)},
                )

            return {
                "result": result,
                "elapsed_ms": elapsed_ms,
                "error_result": error_result,
                "error_reason": error_reason,
            }

        # Dynamic ORC loop: dispatch -> observe -> ORC decides next step.
        pending_queue = list(agent_sequence)
        retry_count: dict[str, int] = {a: 0 for a in agent_sequence}
        prev_agent: str | None = None
        scheduler_steps = 0
        max_scheduler_steps = max(8, len(agent_sequence) * (self.max_agent_retries + 3))
        aborted_by_orc = False

        while pending_queue and scheduler_steps < max_scheduler_steps:
            scheduler_steps += 1
            agent_name = pending_queue.pop(0)
            task_msg = _dispatch_task(agent_name, sync_from=prev_agent)
            exec_out = _execute_task(agent_name, task_msg)
            latest_result = exec_out["result"]
            latest_error = exec_out.get("error_reason") if exec_out.get("error_result") else None

            decision = self.orc.evaluate_next_step(
                objective=str(task.get("objective", "")),
                completed_agents=list(results_by_agent.keys()),
                pending_agents=list(pending_queue),
                latest_agent=agent_name,
                latest_error=latest_error,
                latest_output_preview=self._preview(latest_result.output, max_len=320),
                retry_count=retry_count.get(agent_name, 0),
            )

            action = str(decision.get("action", "continue")).lower().strip()
            next_agent = decision.get("next_agent")
            self._record_orc_reason(
                task=task,
                stage="replan",
                thought=(
                    f"After {agent_name.upper()} completion ORC selected action '{action}'"
                    + (f" with next agent '{next_agent}'." if next_agent else ".")
                ),
                decision=decision,
            )

            if action == "abort":
                if latest_error and pending_queue:
                    aborted_by_orc = True
                    break
                if pending_queue:
                    self._record_orc_reason(
                        task=task,
                        stage="abort_downgraded",
                        thought=(
                            f"Received ABORT from ORC without a hard error; downgrading to SKIP for {agent_name.upper()} "
                            "to keep pipeline progress deterministic."
                        ),
                        decision={"agent": agent_name, "forced_action": "skip"},
                    )
                    action = "skip"
                else:
                    action = "terminate"

            if action == "terminate":
                break

            if action == "retry" and retry_count.get(agent_name, 0) < self.max_agent_retries:
                retry_count[agent_name] = retry_count.get(agent_name, 0) + 1
                backoff = self.retry_backoff_base_seconds * (2 ** (retry_count[agent_name] - 1))
                time.sleep(min(4.0, backoff))
                pending_queue.insert(0, agent_name)
                continue

            if action == "retry" and retry_count.get(agent_name, 0) >= self.max_agent_retries:
                self._record_orc_reason(
                    task=task,
                    stage="retry_cap",
                    thought=(
                        f"Retry cap reached for {agent_name.upper()} "
                        f"({retry_count.get(agent_name, 0)} retries). Forcing SKIP."
                    ),
                    decision={"agent": agent_name, "forced_action": "skip"},
                )
                action = "skip"

            if action == "skip":
                prev_agent = agent_name
                continue

            if isinstance(next_agent, str) and next_agent in pending_queue:
                pending_queue.remove(next_agent)
                pending_queue.insert(0, next_agent)

            prev_agent = agent_name

        if pending_queue and scheduler_steps >= max_scheduler_steps:
            self._record_orc_reason(
                task=task,
                stage="scheduler_guard",
                thought=(
                    "Reached ORC scheduler step guard before completing all pending agents; "
                    "terminating loop to avoid infinite orchestration."
                ),
                decision={"pending_agents": pending_queue, "max_scheduler_steps": max_scheduler_steps},
                terminated=True,
            )

        if aborted_by_orc:
            self._record_orc_reason(
                task=task,
                stage="abort",
                thought="ORC aborted execution due to dynamic replanning decision.",
                decision={"aborted": True, "pending_agents": pending_queue},
                terminated=True,
            )

        vra_r = results_by_agent.get("vra")
        ga_r = results_by_agent.get("ga")
        pa_r = results_by_agent.get("pa")

        if vra_r is not None and ga_r is not None:
            merged_state = self.orc.aggregate([vra_r, ga_r])
        else:
            ordered_partial = [results_by_agent[n] for n in agent_sequence if n in results_by_agent]
            if len(ordered_partial) >= 2:
                merged_state = self.orc.aggregate(ordered_partial[:2])
            elif len(ordered_partial) == 1:
                merged_state = dict(ordered_partial[0].output)
                merged_state.setdefault("safety_valid", True)
                merged_state.setdefault("safety_missing", [])
            else:
                merged_state = {"safety_valid": False, "safety_missing": ["no_agent_results"]}

        self.memory.write("conflict_resolutions", merged_state.get("selected_by_conflict"))
        self._record_flow_event(
            "orc_merge",
            {
                "result_messages": result_msg_count,
                "has_conflict_selection": bool(merged_state.get("selected_by_conflict")),
            },
        )
        self._record_orc_reason(
            task=task,
            stage="merge",
            thought="Merging agent outputs, resolving conflicts, and applying safety validation before final termination.",
            decision={
                "result_messages": result_msg_count,
                "has_conflict_selection": bool(merged_state.get("selected_by_conflict")),
            },
        )

        # Step 5b: memory budget controls.
        em_items = self.memory.items()
        token_estimate = self.memory.token_count()
        ccq = 1.0
        if token_estimate > self.em_soft_limit_tokens:
            em_items, ccq = compress_context(em_items)
            self.memory.set_items(em_items, ccq)
            self.trajectory_log.append("orc", {
                "event": "compression",
                "thread_id": thread_id,
                "token_estimate": token_estimate,
                "ccq": ccq,
            })

        ordered_results = [results_by_agent[n] for n in ("vra", "ga", "pa") if n in results_by_agent]
        if len(ordered_results) >= 2:
            final_state = self.orc.aggregate(ordered_results)
        elif len(ordered_results) == 1:
            final_state = dict(ordered_results[0].output)
            final_state.setdefault("safety_valid", True)
            final_state.setdefault("safety_missing", [])
        else:
            final_state = {"safety_valid": False, "safety_missing": ["no_agent_results"]}

        final_state.setdefault("query_type", query_type)

        final_answer = self.orc.synthesize_answer(
            objective=str(task.get("objective", "")),
            agent_results=results_by_agent,
            merged_state=final_state,
        )
        if isinstance(final_answer, str) and final_answer.strip():
            final_state["ans"] = final_answer.strip()

        dispatched_sequence = self._sequence_from_flow("orc_dispatch_task")
        executed_sequence = self._sequence_from_flow("agent_run_complete")
        plan_follow_metrics = self._compute_plan_follow_metrics(
            planned_sequence=agent_sequence,
            dispatched_sequence=dispatched_sequence,
            executed_sequence=executed_sequence,
        )

        agents_output = self._build_agents_output(
            task=task,
            planned_items=planned_items,
            dispatched_sequence=dispatched_sequence,
            results_by_agent=results_by_agent,
        )
        verbose_agents_output_path = self._write_verbose_agents_output_log(run_dir, agents_output)

        final_state["plan"] = planned_items
        final_state["plan_accuracy"] = plan_follow_metrics["plan_accuracy"]
        final_state["orc_aborted"] = aborted_by_orc

        # Route safety override: if any planned route has RSS < 1.0, force safety failure.
        route_rss: list[float] = []
        if pa_r is not None:
            for call in pa_r.tool_calls:
                out = call.get("output", {})
                if isinstance(out, dict) and isinstance(out.get("routes"), list):
                    for route in out["routes"]:
                        if isinstance(route, dict) and isinstance(route.get("rss"), (int, float)):
                            route_rss.append(float(route["rss"]))
        if route_rss and min(route_rss) < 1.0:
            final_state["safety_valid"] = False
            missing = list(final_state.get("safety_missing", []))
            if "route_safety_score>=1.0" not in missing:
                missing.append("route_safety_score>=1.0")
            final_state["safety_missing"] = missing
            self.trajectory_log.append("orc", {
                "event": "safety_override",
                "thread_id": thread_id,
                "min_rss": min(route_rss),
            })

        # Step 7: safety + termination checks.
        safety_payload = dict(final_state)
        if task.get("enforce_safety_keys"):
            safety_payload["enforce_safety_keys"] = True
        ok, missing = validate_safety_payload(safety_payload)
        if not ok:
            final_state["safety_valid"] = False
            final_state["safety_missing"] = missing

        self._record_orc_reason(
            task=task,
            stage="terminate",
            thought=(
                "Termination check complete. Returning final merged state with safety verdict and plan-follow metrics."
            ),
            decision={
                "safety_valid": final_state.get("safety_valid"),
                "safety_missing": final_state.get("safety_missing", []),
                "plan_accuracy": plan_follow_metrics.get("plan_accuracy"),
            },
            terminated=True,
        )

        # Step 8: trajectory log bundle.
        episode_data = {
            "task": task,
            "vra_result": vra_r.output if vra_r is not None else {},
            "ga_result": ga_r.output if ga_r is not None else {},
            "pa_result": pa_r.output if pa_r is not None else {},
            "merged": final_state,
            "ccq": ccq,
            "total_tool_calls": (
                sum(len(r.tool_calls) for r in results_by_agent.values())
            ),
            "thread_id": thread_id,
        }
        self.trajectory_log.append("orc", {
            "event": "episode_complete", "data": episode_data
        })

        total_ms = (time.perf_counter() - run_started) * 1000.0
        
        # Determine success via a single explicit TSR condition.
        agent_error_count = sum(1 for info in agent_errors.values() if info.get("error_result"))
        safety_ok = bool(final_state.get("safety_valid", True))
        final_answer = str(final_state.get("ans", "") or "").strip()
        has_final_answer = bool(final_answer) and final_answer.lower() not in {"error", "timeout", "mock result"}
        task_success = (
            safety_ok
            and not aborted_by_orc
            and len(results_by_agent) > 0
            and has_final_answer
        )
        
        metrics = {
            "ccq": ccq,
            "safety_valid": final_state.get("safety_valid"),
            "safety_missing": final_state.get("safety_missing"),
            "total_tool_calls": episode_data["total_tool_calls"],
            "task_success": task_success,
            "hrr_proxy": 1 if (vra_r is not None and ga_r is not None and vra_r.n_turns == 0 and ga_r.n_turns == 0) else 0,
            "execution_mode": execution_mode,
            "plan_accuracy": plan_follow_metrics["plan_accuracy"],
            "plan_followed": plan_follow_metrics["plan_followed"],
            "plan_loop_count": plan_follow_metrics["plan_loop_count"],
            "off_plan_steps": plan_follow_metrics["off_plan_steps"],
            "planned_sequence": plan_follow_metrics["planned_sequence"],
            "dispatched_sequence": plan_follow_metrics["dispatched_sequence"],
            "executed_sequence": plan_follow_metrics["executed_sequence"],
            "agent_error_count": agent_error_count,
            "agent_errors": {k: v for k, v in agent_errors.items() if v.get("error_result")},
        }

        logger.info("EPISODE START task_id=%s", task.get("scene_id", "unknown"))
        logger.info("ORC execution mode=%s", execution_mode)
        logger.info("ORC call order=%s", " -> ".join(agent_sequence))
        for agent_name in agent_sequence:
            if agent_name in timings_by_agent and agent_name in results_by_agent:
                agent_res = results_by_agent[agent_name]
                logger.info(
                    "%s completed in %.2fs | turns=%s | tools=%s",
                    agent_name.upper(),
                    timings_by_agent[agent_name] / 1000.0,
                    agent_res.n_turns,
                    len(agent_res.tool_calls),
                )
        if vra_r is not None and ga_r is not None and final_state.get("selected_by_conflict"):
            logger.info("Conflict resolution: ORC resolved conflicting reports")
        if safety_ok:
            logger.info("Safety: PASS")
        else:
            missing_str = ", ".join(final_state.get("safety_missing", []))
            logger.info("Safety: FAIL missing=[%s]", missing_str)
        logger.info(
            "Plan accuracy=%.2f | loops=%s | dispatched=%s",
            metrics["plan_accuracy"],
            metrics["plan_loop_count"],
            " -> ".join(metrics["dispatched_sequence"]) if metrics["dispatched_sequence"] else "none",
        )
        if metrics["plan_accuracy"] < 0.8:
            logger.warning(
                "Plan-follow deviation detected planned=%s dispatched=%s",
                metrics["planned_sequence"],
                metrics["dispatched_sequence"],
            )
        logger.info("Agent errors=%s", metrics["agent_error_count"])
        logger.info(
            "CCQ=%.2f | TSR=%s | total_tool_calls=%s",
            ccq,
            "PASS" if task_success else "FAIL",
            metrics["total_tool_calls"],
        )

        flow_log = self._build_flow_log(
            thread_id=thread_id,
            task=task,
            execution_mode=execution_mode,
            agent_sequence=agent_sequence,
        )
        flow_path = run_dir / "flow_log.json"
        flow_path.write_text(json.dumps(flow_log, indent=2), encoding="utf-8")

        bundle = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_dir": str(run_dir),
            "task": task,
            "thread_id": thread_id,
            "execution_mode": execution_mode,
            "agent_call_order": agent_sequence,
            "dispatched_agent_order": dispatched_sequence,
            "executed_agent_order": executed_sequence,
            "plan": {
                "sequence": planned_items,
                "accuracy": plan_follow_metrics["plan_accuracy"],
                "followed_exactly": plan_follow_metrics["plan_followed"],
                "loop_count": plan_follow_metrics["plan_loop_count"],
                "off_plan_steps": plan_follow_metrics["off_plan_steps"],
            },
            "flow_log_path": str(flow_path),
            "verbose_agents_output_path": str(verbose_agents_output_path),
            "agent_timings": {
                "vra": {
                    "elapsed_ms": round(timings_by_agent["vra"], 2) if "vra" in timings_by_agent else None,
                    "n_turns": results_by_agent["vra"].n_turns if "vra" in results_by_agent else 0,
                },
                "ga": {
                    "elapsed_ms": round(timings_by_agent["ga"], 2) if "ga" in timings_by_agent else None,
                    "n_turns": results_by_agent["ga"].n_turns if "ga" in results_by_agent else 0,
                },
                "pa": {
                    "elapsed_ms": round(timings_by_agent["pa"], 2) if "pa" in timings_by_agent else None,
                    "n_turns": results_by_agent["pa"].n_turns if "pa" in results_by_agent else 0,
                },
                "orc": {
                    "elapsed_ms": None,
                    "n_turns": len(self._orc_trace),
                },
                "total_episode_ms": round(total_ms, 2),
            },
            "agent_outputs": {
                "orc": {
                    "trace_turns": len(self._orc_trace),
                    "latest_decision": (self._orc_trace[-1].get("decision") if self._orc_trace else {}),
                },
                "vra": results_by_agent["vra"].output if "vra" in results_by_agent else {},
                "ga": results_by_agent["ga"].output if "ga" in results_by_agent else {},
                "pa": results_by_agent["pa"].output if "pa" in results_by_agent else {},
                "orc_merged": final_state,
            },
            "agents_output": agents_output,
            "agent_tool_calls": {
                "vra": results_by_agent["vra"].tool_calls if "vra" in results_by_agent else [],
                "ga": results_by_agent["ga"].tool_calls if "ga" in results_by_agent else [],
                "pa": results_by_agent["pa"].tool_calls if "pa" in results_by_agent else [],
            },
            "agent_traces": {
                "orc": self._orc_trace,
                "vra": results_by_agent["vra"].trace if "vra" in results_by_agent else [],
                "ga": results_by_agent["ga"].trace if "ga" in results_by_agent else [],
                "pa": results_by_agent["pa"].trace if "pa" in results_by_agent else [],
            },
            "episode_metrics": {
                "ccq": ccq,
                "safety_valid": final_state.get("safety_valid"),
                "safety_missing": final_state.get("safety_missing"),
                "total_tool_calls": episode_data["total_tool_calls"],
                "execution_mode": execution_mode,
                "plan_accuracy": plan_follow_metrics["plan_accuracy"],
                "plan_followed": plan_follow_metrics["plan_followed"],
                "plan_loop_count": plan_follow_metrics["plan_loop_count"],
                "agent_error_count": metrics["agent_error_count"],
                "agent_errors": metrics["agent_errors"],
            },
            "mpc_messages": self._message_trace,
            "flow_log": flow_log,
            "agent_runtime_config": {
                "vra": {"model_id": self.vra.model_id, "base_url": self.vra.base_url},
                "ga": {"model_id": self.ga.model_id, "base_url": self.ga.base_url},
                "pa": {"model_id": self.pa.model_id, "base_url": self.pa.base_url},
                "orc": {"model_id": self.orc.model_id, "base_url": self.orc.base_url},
            },
        }
        self._save_run_bundle(run_dir, bundle)

        # 13 & 14. Terminate and return
        return {
            "result": final_state, 
            "metrics": metrics, 
            "run_dir": str(run_dir),
            "bundle": bundle
        }

    def run_episode(self, query: str) -> dict[str, Any]:
        ep = self.run_single_episode({
            "objective": query,
            "scene_id": "test-scene",
            "region": "test-region"
        })
        return {"final": {"status": "terminated"}, "episode": ep}
