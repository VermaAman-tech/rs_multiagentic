import json

from agents.orc import Orchestrator


def test_orc_plan_fallback_geospatial_proximity(monkeypatch):
    orc = Orchestrator()
    monkeypatch.setattr(orc, "_call_llm", lambda messages, tools: {"content": "not-json"})

    plan = orc.plan(
        objective="Which fire station and police station are closest in Banff National Park?",
        available_agents=["vra", "ga", "pa"],
    )

    assert plan["query_type"] == "geospatial_proximity"
    assert plan["agent_sequence"] == ["ga", "pa"]
    assert "ga" in plan["sub_tasks"]
    assert "pa" in plan["sub_tasks"]


def test_orc_plan_prefers_valid_llm_plan(monkeypatch):
    orc = Orchestrator()

    llm_plan = {
        "query_type": "geospatial_proximity",
        "agent_sequence": ["ga", "invalid", "pa"],
        "sub_tasks": {
            "ga": "Get boundary and relevant POIs",
            "pa": "Compute pairwise distances and return minimum pair",
        },
        "rationale": "No imagery required.",
    }

    monkeypatch.setattr(orc, "_call_llm", lambda messages, tools: {"content": json.dumps(llm_plan)})

    plan = orc.plan(
        objective="Find nearest fire/police stations in Banff",
        available_agents=["vra", "ga", "pa"],
    )

    assert plan["agent_sequence"] == ["ga", "pa"]
    assert plan["sub_tasks"]["ga"].startswith("Get boundary")
    assert plan["sub_tasks"]["pa"].startswith("Compute pairwise")


def test_orc_plan_forces_ga_pa_on_proximity(monkeypatch):
    orc = Orchestrator()

    llm_plan = {
        "query_type": "geospatial_proximity",
        "agent_sequence": ["vra", "ga", "pa"],
        "sub_tasks": {
            "vra": "Do visual analysis",
            "ga": "Compute geospatial features",
            "pa": "Finalize nearest pair",
        },
        "rationale": "Model proposed all agents.",
    }

    monkeypatch.setattr(orc, "_call_llm", lambda messages, tools: {"content": json.dumps(llm_plan)})

    plan = orc.plan(
        objective="Find nearest stations from distance observations",
        available_agents=["vra", "ga", "pa"],
    )

    assert plan["query_type"] == "geospatial_proximity"
    assert plan["agent_sequence"] == ["ga", "pa"]


def test_orc_evaluate_next_step_skips_on_error_by_default():
    orc = Orchestrator()
    decision = orc.evaluate_next_step(
        objective="sample",
        completed_agents=["ga"],
        pending_agents=["pa"],
        latest_agent="ga",
        latest_error="context overflow",
        latest_output_preview="",
    )

    assert decision["action"] == "skip"
    assert decision["next_agent"] == "pa"


def test_orc_evaluate_next_step_sanitizes_invalid_action(monkeypatch):
    orc = Orchestrator()
    monkeypatch.setattr(
        orc,
        "_call_llm",
        lambda messages, tools: {
            "content": json.dumps(
                {
                    "action": "dance",
                    "next_agent": "ga",
                    "reason": "invalid",
                }
            )
        },
    )

    decision = orc.evaluate_next_step(
        objective="sample",
        completed_agents=["vra"],
        pending_agents=["pa"],
        latest_agent="vra",
        latest_error=None,
        latest_output_preview="ok",
    )

    assert decision["action"] == "continue"
    assert decision["next_agent"] == "pa"
