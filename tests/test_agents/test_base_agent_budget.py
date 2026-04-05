import json

from agents.base_agent import BaseAgent


class DummyAgent(BaseAgent):
    name = "dummy"


def test_tool_result_compacts_boundary_wkt():
    agent = DummyAgent()
    out = agent._tool_result_for_llm(
        {
            "boundary_wkt": "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
            "gpkg_path": "data/tmp/a.gpkg",
            "success": True,
        }
    )
    parsed = json.loads(out)
    assert "boundary_summary" in parsed
    assert parsed["boundary_summary"]["has_geometry"] is True
    assert parsed["boundary_summary"]["gpkg_path"] == "data/tmp/a.gpkg"


def test_tool_result_compacts_poi_list():
    agent = DummyAgent()
    pois = [{"name": f"p{i}", "lat": i * 1.0, "lon": i * 2.0} for i in range(12)]
    out = agent._tool_result_for_llm({"pois": pois, "success": True})
    parsed = json.loads(out)
    assert "pois_summary" in parsed
    assert parsed["pois_summary"]["count"] == 12
    assert len(parsed["pois_summary"]["sample"]) == 5


def test_prepare_messages_enforces_budget_and_history_limit():
    agent = DummyAgent(model_max_context_tokens=200)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(12):
        messages.append({"role": "assistant", "content": f"thinking {i}"})
        messages.append({"role": "tool", "content": "x" * 3000})

    prepared = agent._prepare_messages_for_llm(messages)

    assert len(prepared) <= agent._MAX_MESSAGES
    budget = int(agent.model_max_context_tokens * agent._CONTEXT_BUDGET_RATIO)
    assert agent._estimate_messages_tokens(prepared) <= budget or len(prepared) <= 4


def test_add_pois_normalization_extracts_bbox_from_wkt():
    agent = DummyAgent()
    norm = agent._normalize_tool_arguments(
        "AddPoisLayer",
        {
            "poi_category": "fire_station",
            "boundary_wkt": "POLYGON ((10 20, 10 21, 11 21, 11 20, 10 20))",
        },
    )

    assert norm["poi_category"] == "fire_station"
    assert "bbox" in norm
    assert len(norm["bbox"]) == 4


def test_task_adaptive_prompt_includes_query_type_and_subtask():
    agent = DummyAgent(thinking_mode="always")
    prompt = agent._build_task_adaptive_system_prompt(
        {
            "task_description": "Compute closest fire/police pair.",
            "query_type": "geospatial_proximity",
        }
    )

    assert "Assigned sub-task from ORC" in prompt
    assert "geospatial_proximity" in prompt
    assert "Always provide an explicit thought" in prompt
