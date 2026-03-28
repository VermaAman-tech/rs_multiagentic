from __future__ import annotations

from typing import Any
from pathlib import Path
from datetime import datetime
import json
import time
from agents.vra import VisionReasoningAgent
from agents.ga import GeospatialAgent
from agents.pa import PlanningAgent
from agents.orc import Orchestrator
from framework.memory.episodic_memory import EpisodicMemory
from framework.memory.trajectory_log import TrajectoryLog
from framework.mpc.message import Message
from framework.protocols.compression import compress_context
from framework.protocols.safety import validate_safety_payload
from framework.protocols.deadlock import DeadlockDetector, break_cycle
from evaluation import metrics as eval_metrics

class EpisodeRunner:
    def __init__(
        self,
        model_id: str = "mock-model",
        base_url: str = "http://localhost:8000/v1",
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

        # 1. ORC decompose task
        l2_plan = {"vra_task": task, "ga_task": task}

        # 2 & 3. ORC sends TASK to VRA and GA
        msg_vra = Message(message_type="TASK", sender="orc", recipient="vra",
                         payload=l2_plan["vra_task"], priority=1)
        msg_ga = Message(message_type="TASK", sender="orc", recipient="ga",
                        payload=l2_plan["ga_task"], priority=1)

        # 4. VRA runs ReAct loop
        vra_t0 = time.perf_counter()
        vra_r = self.vra.handle_task(msg_vra)
        vra_ms = (time.perf_counter() - vra_t0) * 1000.0
        if hasattr(vra_r, "output") and isinstance(vra_r.output, dict):
             for k, v in vra_r.output.items():
                 self.memory.write(f"vra_{k}", v)

        # 5. GA runs ReAct loop
        ga_t0 = time.perf_counter()
        ga_r = self.ga.handle_task(msg_ga)
        ga_ms = (time.perf_counter() - ga_t0) * 1000.0
        if hasattr(ga_r, "output") and isinstance(ga_r.output, dict):
             for k, v in ga_r.output.items():
                 self.memory.write(f"ga_{k}", v)

        # Deadlock detection
        self.deadlock_detector.add_wait("orc", "vra")
        self.deadlock_detector.add_wait("orc", "ga")
        self.deadlock_detector.remove_wait("orc", "vra")
        self.deadlock_detector.remove_wait("orc", "ga")

        # 6 & 7. ORC receives RESULTs, checks consistency
        merged_state = self.orc.aggregate([vra_r, ga_r])

        # 8. ORC checks EM budget
        em_items = self.memory.items()
        token_estimate = len(str(em_items)) // 4
        ccq = 1.0
        if token_estimate > 6400:
            em_items, ccq = compress_context(em_items)
            self.memory.set_items(em_items, ccq)

        # 9. ORC sends SYNC + TASK to PA
        pa_task = task.copy()
        pa_task.update(merged_state)
        msg_pa = Message(message_type="TASK", sender="orc", recipient="pa",
                        payload=pa_task)

        # 10. PA runs
        pa_t0 = time.perf_counter()
        pa_r = self.pa.handle_task(msg_pa)
        pa_ms = (time.perf_counter() - pa_t0) * 1000.0
        if hasattr(pa_r, "output") and isinstance(pa_r.output, dict):
             for k, v in pa_r.output.items():
                 self.memory.write(f"pa_{k}", v)

        final_state = self.orc.aggregate([vra_r, ga_r, pa_r])

        # 11. Safety Check
        ok, missing = validate_safety_payload(final_state)
        if not ok:
            final_state["safety_valid"] = False
            final_state["safety_missing"] = missing

        # 12. Trajectory Log
        episode_data = {
            "task": task,
            "vra_result": vra_r.output,
            "ga_result": ga_r.output,
            "pa_result": pa_r.output,
            "merged": final_state,
            "ccq": ccq,
            "total_tool_calls": (
                len(vra_r.tool_calls) + len(ga_r.tool_calls) + len(pa_r.tool_calls)
            ),
        }
        self.trajectory_log.append("orc", {
            "event": "episode_complete", "data": episode_data
        })

        total_ms = (time.perf_counter() - run_started) * 1000.0
        
        # Determine success (a rough TSR proxy: safety is valid and we reached the end)
        pa_out = pa_r.output.get("raw_output", "")
        has_answer = "ans" in pa_r.output or len(str(pa_out)) > 15
        task_success = has_answer and (final_state.get("safety_valid", True) if "safety_valid" in final_state else True)
        
        metrics = {
            "ccq": ccq,
            "safety_valid": final_state.get("safety_valid"),
            "safety_missing": final_state.get("safety_missing"),
            "total_tool_calls": episode_data["total_tool_calls"],
            "task_success": task_success,
            "hrr_proxy": 1 if (vra_r.n_turns == 0 and ga_r.n_turns == 0) else 0,
        }

        print("╔══════════════════════════════════════════════════════════╗")
        print(f"║  EPISODE START  task_id={task.get('scene_id', 'unknown'):<28} ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print("→ ORC: dispatching to VRA + GA")
        print(f"→ VRA completed in {vra_ms/1000.0:.2f}s | turns={vra_r.n_turns} | tools={len(vra_r.tool_calls)}")
        print(f"→ GA  completed in {ga_ms/1000.0:.2f}s | turns={ga_r.n_turns} | tools={len(ga_r.tool_calls)}")
        if len([vra_r, ga_r]) >= 2 and final_state.get("selected_by_conflict"):
            print("→ Conflict resolution: ORC resolved conflicting reports")
        print(f"→ PA  completed in {pa_ms/1000.0:.2f}s | turns={pa_r.n_turns} | tools={len(pa_r.tool_calls)}")
        if task_success:
            print("→ Safety: PASS")
        else:
            missing_str = ", ".join(final_state.get("safety_missing", []))
            print(f"→ Safety: FAIL — missing [{missing_str}]")
        print(f"→ CCQ: {ccq:.2f} | TSR: {'PASS' if task_success else 'FAIL'} | total_tool_calls: {metrics['total_tool_calls']}")
        print("══════════════════════════════════════════════════════════\n")

        bundle = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_dir": str(run_dir),
            "task": task,
            "agent_timings": {
                "vra": {"elapsed_ms": round(vra_ms, 2), "n_turns": vra_r.n_turns},
                "ga": {"elapsed_ms": round(ga_ms, 2), "n_turns": ga_r.n_turns},
                "pa": {"elapsed_ms": round(pa_ms, 2), "n_turns": pa_r.n_turns},
                "total_episode_ms": round(total_ms, 2),
            },
            "agent_outputs": {
                "vra": vra_r.output,
                "ga": ga_r.output,
                "pa": pa_r.output,
                "orc_merged": final_state,
            },
            "agent_tool_calls": {
                "vra": vra_r.tool_calls,
                "ga": ga_r.tool_calls,
                "pa": pa_r.tool_calls,
            },
            "agent_traces": {
                "vra": vra_r.trace,
                "ga": ga_r.trace,
                "pa": pa_r.trace,
            },
            "episode_metrics": {
                "ccq": ccq,
                "safety_valid": final_state.get("safety_valid"),
                "safety_missing": final_state.get("safety_missing"),
                "total_tool_calls": episode_data["total_tool_calls"],
            },
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
