from agents.base_agent import AgentResult
from framework.episode_runner import EpisodeRunner


def _ok_result(agent_name: str) -> AgentResult:
    return AgentResult(
        output={"ans": "ok", "raw_output": f"{agent_name} success"},
        confidence=0.9,
        tool_calls=[],
        n_turns=1,
        trace=[{"turn": 1, "thought": f"{agent_name} done", "tool_calls": []}],
    )


def test_orc_retries_agent_after_error(tmp_path):
    runner = EpisodeRunner()
    runner.results_root = tmp_path

    runner.orc.plan = lambda objective, available_agents, context=None: {
        "query_type": "geospatial_proximity",
        "agent_sequence": ["ga", "pa"],
        "sub_tasks": {
            "ga": "Collect boundary and POIs",
            "pa": "Compute nearest pair distance",
        },
        "rationale": "Deterministic test plan",
    }

    def _next_step(**kwargs):
        if kwargs.get("latest_error"):
            return {"action": "retry", "next_agent": kwargs.get("latest_agent"), "reason": "retry"}
        pending = kwargs.get("pending_agents", [])
        if pending:
            return {"action": "continue", "next_agent": pending[0], "reason": "continue"}
        return {"action": "terminate", "next_agent": None, "reason": "done"}

    runner.orc.evaluate_next_step = _next_step

    calls = {"ga": 0}

    def ga_handle_task(message, tools_schema=None):
        calls["ga"] += 1
        if calls["ga"] == 1:
            return AgentResult(
                output={"ans": "error", "raw_output": "context overflow"},
                confidence=0.1,
                tool_calls=[],
                n_turns=1,
                trace=[{"turn": 1, "thought": "llm call failed: 400", "tool_calls": []}],
            )
        return _ok_result("ga")

    runner.ga.handle_task = ga_handle_task
    runner.pa.handle_task = lambda message, tools_schema=None: _ok_result("pa")
    runner.vra.handle_task = lambda message, tools_schema=None: _ok_result("vra")

    episode = runner.run_single_episode(
        {
            "objective": "Find closest fire and police stations in Banff",
            "scene_id": "s1",
            "region": "banff",
        }
    )

    dispatched = episode["bundle"]["dispatched_agent_order"]
    assert calls["ga"] == 2
    assert dispatched.count("ga") >= 2
    assert "pa" in dispatched
