from __future__ import annotations

from typing import Any
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import json
import time
from agents.vra import VisionReasoningAgent
from agents.ga import GeospatialAgent
from agents.pa import PlanningAgent
from agents.orc import Orchestrator
from framework.memory.episodic_memory import EpisodicMemory
from framework.memory.trajectory_log import TrajectoryLog
from framework.mpc.message import Message
from framework.mpc.broker import MessageBroker
from framework.protocols.compression import compress_context
from framework.protocols.safety import validate_safety_payload
from framework.protocols.deadlock import DeadlockDetector, break_cycle
from evaluation import metrics as eval_metrics

class EpisodeRunner:
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

        # All agents share the same model and tool server
        model_map = agent_model_map or {}
        url_map = agent_base_url_map or {}

        def _mk(agent_name: str) -> dict[str, Any]:
            return {
                "model_id": model_map.get(agent_name, model_id),
                "base_url": url_map.get(agent_name, base_url),
                "tool_server": self.tool_server,
                "allow_mock_fallback": allow_mock_fallback,
                "max_turns": max_turns,
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
        self.ack_timeout_seconds = 30
        self.default_parallel_vra_ga = False

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
                    self.deadlock_detector.remove_wait("orc", agent_name)
                    return True
                self._orc_backlog.append(msg)
        return False

    def _persist_agent_output(self, agent_name: str, output: dict[str, Any], epoch_ref: int | None) -> None:
        self.memory.write(f"{agent_name}_result", output)
        if epoch_ref is not None:
            self.memory.write(f"{agent_name}_result_T{epoch_ref}", output)
        for k, v in output.items():
            self.memory.write(f"{agent_name}_{k}", v)

    def _run_agent_from_task(
        self,
        *,
        agent_name: str,
        agent,
        task_msg: Message,
        tools_schema: list[dict[str, Any]],
    ) -> tuple[Any, float]:
        self._emit_ack(task_msg, agent_name)
        ack_ok = self._await_ack(task_msg, agent_name)
        if not ack_ok:
            cycle = self.deadlock_detector.detect_cycle()
            if cycle:
                break_cycle(self.deadlock_detector)

            timeout_msg = self._build_message(
                msg_type="ERROR",
                sender="orc",
                recipient=agent_name,
                payload={
                    "error_type": "ACK_TIMEOUT",
                    "tool_name": None,
                    "step": "pre_execution",
                    "message": f"No ACK within {self.ack_timeout_seconds}s",
                },
                thread_id=task_msg.thread_id,
                parent_msg_id=task_msg.msg_id,
                confidence=1.0,
                epoch_ref=task_msg.epoch_ref,
                priority=3,
                ttl_ms=30_000,
            )
            self._publish(timeout_msg)

        t0 = time.perf_counter()
        result = agent.handle_task(task_msg, tools_schema=tools_schema)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if hasattr(result, "output") and isinstance(result.output, dict):
            self._persist_agent_output(agent_name, result.output, task_msg.epoch_ref)

        result_payload = {
            "output_type": "agent_result",
            "data_path_or_inline": result.output,
            "quality_score": float(getattr(result, "confidence", 0.0)),
            "reasoning_summary": str(result.output.get("raw_output", ""))[:500],
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
        return result, elapsed_ms

    def _context_keys_for(self, agent_name: str) -> list[str]:
        keys = sorted(self.memory.items().keys())
        if agent_name == "vra":
            return [k for k in keys if "episode_metadata" in k][:8]
        if agent_name == "ga":
            preferred = [
                "episode_metadata",
                "vra_damage_polygons",
                "vra_flood_extent",
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
            ]
            selected = [k for k in keys if any(p in k for p in preferred)]
            if not selected:
                selected = [k for k in keys if k.startswith(("vra_", "ga_", "episode_metadata"))]
            return selected[:24]
        return []

    def _task_payload(
        self,
        *,
        task: dict[str, Any],
        task_description: str,
        tools_schema_subset: list[str],
        em_context_keys: list[str],
        deadline_ms: int,
    ) -> dict[str, Any]:
        payload = {
            "task_description": task_description,
            "tool_schema_subset": tools_schema_subset,
            "em_context_keys": em_context_keys,
            "deadline_ms": deadline_ms,
            # Keep legacy keys expected by existing agent output extraction.
            "objective": task.get("objective"),
            "scene_id": task.get("scene_id"),
            "region": task.get("region"),
        }
        if "epoch_ref" in task:
            payload["epoch_ref"] = task["epoch_ref"]
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
        # Keep all core agents reachable unless explicitly disabled by caller.
        for name in allowed:
            if name not in seq:
                seq.append(name)
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
        run_dir = self._next_run_dir()
        run_started = time.perf_counter()
        thread_id = str(uuid4())
        self._message_trace = []
        self._orc_backlog = []
        self._flow_trace = []
        self._flow_seq = 0

        episode_metadata = {
            "thread_id": thread_id,
            "objective": task.get("objective"),
            "scene_id": task.get("scene_id"),
            "region": task.get("region"),
            "started_at": datetime.utcnow().isoformat() + "Z",
        }
        self.memory.write("episode_metadata", episode_metadata)

        plan = self._decompose_task(task)
        now_ms = int(time.time() * 1000)
        deadline_ms = now_ms + 180_000

        agent_registry: dict[str, dict[str, Any]] = {
            "vra": {"agent": self.vra, "tools": self.vra.VRA_TOOLS, "plan": plan["vra"]},
            "ga": {"agent": self.ga, "tools": self.ga.GA_TOOLS, "plan": plan["ga"]},
            "pa": {"agent": self.pa, "tools": self.pa.PA_TOOLS, "plan": plan["pa"]},
        }
        agent_sequence = self._resolve_agent_sequence(task)
        parallel_vra_ga = self._allow_parallel_vra_ga(task)
        execution_mode = "parallel_vra_ga" if parallel_vra_ga else "sequential"

        self._record_flow_event(
            "orc_decompose",
            {
                "thread_id": thread_id,
                "execution_mode": execution_mode,
                "agent_sequence": agent_sequence,
                "plan": plan,
            },
        )

        results_by_agent: dict[str, Any] = {}
        timings_by_agent: dict[str, float] = {}
        result_msg_count = 0

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

            reg = agent_registry[agent_name]
            msg = self._build_message(
                msg_type="TASK",
                sender="orc",
                recipient=agent_name,
                payload=self._task_payload(
                    task=task,
                    task_description=reg["plan"],
                    tools_schema_subset=self._tool_names(reg["tools"]),
                    em_context_keys=self._context_keys_for(agent_name),
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

            self.deadlock_detector.add_wait("orc", agent_name)
            inbox = self._drain_for(agent_name)
            task_msgs = [m for m in inbox if m.msg_type == "TASK"]
            if not task_msgs:
                raise RuntimeError(f"Failed to route TASK message to {agent_name}")
            return task_msgs[0]

        def _execute_task(agent_name: str, task_msg: Message) -> None:
            nonlocal result_msg_count
            reg = agent_registry[agent_name]
            result, elapsed_ms = self._run_agent_from_task(
                agent_name=agent_name,
                agent=reg["agent"],
                task_msg=task_msg,
                tools_schema=reg["tools"],
            )
            results_by_agent[agent_name] = result
            timings_by_agent[agent_name] = elapsed_ms
            self._record_agent_flow(
                agent_name=agent_name,
                task_msg=task_msg,
                result=result,
                elapsed_ms=elapsed_ms,
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

        # Scheduler: sequential by default, optional early VRA+GA dispatch when explicitly disjoint.
        if parallel_vra_ga and "vra" in agent_sequence and "ga" in agent_sequence:
            first_pair = [n for n in agent_sequence if n in {"vra", "ga"}]
            dispatched: dict[str, Message] = {}
            for agent_name in first_pair:
                dispatched[agent_name] = _dispatch_task(agent_name)
            for agent_name in first_pair:
                _execute_task(agent_name, dispatched[agent_name])

            prev_agent = first_pair[-1] if first_pair else None
            for agent_name in [n for n in agent_sequence if n not in set(first_pair)]:
                task_msg = _dispatch_task(agent_name, sync_from=prev_agent)
                _execute_task(agent_name, task_msg)
                prev_agent = agent_name
        else:
            prev_agent: str | None = None
            for agent_name in agent_sequence:
                task_msg = _dispatch_task(agent_name, sync_from=prev_agent)
                _execute_task(agent_name, task_msg)
                prev_agent = agent_name

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

        # Step 5b: memory budget controls.
        em_items = self.memory.items()
        token_estimate = self.memory.token_count() // 4
        ccq = 1.0
        if token_estimate > 25_600:
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
        
        # Determine success (a rough TSR proxy: safety is valid and we reached the end)
        main_result = pa_r or ga_r or vra_r
        main_output = main_result.output if main_result is not None else {}
        main_out = main_output.get("raw_output", "") if isinstance(main_output, dict) else ""
        has_answer = isinstance(main_output, dict) and (
            "ans" in main_output or len(str(main_out)) > 15
        )
        task_success = has_answer and (final_state.get("safety_valid", True) if "safety_valid" in final_state else True)
        
        metrics = {
            "ccq": ccq,
            "safety_valid": final_state.get("safety_valid"),
            "safety_missing": final_state.get("safety_missing"),
            "total_tool_calls": episode_data["total_tool_calls"],
            "task_success": task_success,
            "hrr_proxy": 1 if (vra_r is not None and ga_r is not None and vra_r.n_turns == 0 and ga_r.n_turns == 0) else 0,
            "execution_mode": execution_mode,
        }

        print("╔══════════════════════════════════════════════════════════╗")
        print(f"║  EPISODE START  task_id={task.get('scene_id', 'unknown'):<28} ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"→ ORC execution mode: {execution_mode}")
        print(f"→ ORC call order: {' -> '.join(agent_sequence)}")
        for agent_name in agent_sequence:
            if agent_name in timings_by_agent and agent_name in results_by_agent:
                agent_res = results_by_agent[agent_name]
                print(
                    f"→ {agent_name.upper()} completed in {timings_by_agent[agent_name]/1000.0:.2f}s | "
                    f"turns={agent_res.n_turns} | tools={len(agent_res.tool_calls)}"
                )
        if vra_r is not None and ga_r is not None and final_state.get("selected_by_conflict"):
            print("→ Conflict resolution: ORC resolved conflicting reports")
        if task_success:
            print("→ Safety: PASS")
        else:
            missing_str = ", ".join(final_state.get("safety_missing", []))
            print(f"→ Safety: FAIL — missing [{missing_str}]")
        print(f"→ CCQ: {ccq:.2f} | TSR: {'PASS' if task_success else 'FAIL'} | total_tool_calls: {metrics['total_tool_calls']}")
        print("══════════════════════════════════════════════════════════\n")

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
            "flow_log_path": str(flow_path),
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
                "total_episode_ms": round(total_ms, 2),
            },
            "agent_outputs": {
                "vra": results_by_agent["vra"].output if "vra" in results_by_agent else {},
                "ga": results_by_agent["ga"].output if "ga" in results_by_agent else {},
                "pa": results_by_agent["pa"].output if "pa" in results_by_agent else {},
                "orc_merged": final_state,
            },
            "agent_tool_calls": {
                "vra": results_by_agent["vra"].tool_calls if "vra" in results_by_agent else [],
                "ga": results_by_agent["ga"].tool_calls if "ga" in results_by_agent else [],
                "pa": results_by_agent["pa"].tool_calls if "pa" in results_by_agent else [],
            },
            "agent_traces": {
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
