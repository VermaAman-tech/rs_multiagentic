import json

from agents.base_agent import AgentResult
from framework.episode_runner import EpisodeRunner


def test_sanitize_task_keeps_only_primary_question_and_drops_ground_truth():
    runner = EpisodeRunner()

    task = {
        "objective": (
            "OBSERVATION: Distances (in meters) saved to line layer: "
            "Banff Fire Station , RCMP Banff Detachment, distance=836.74 m"
        ),
        "human_inputs": [
            "Which fire station and police station are closest to each other in Banff National Park, Alberta, Canada within 3000m buffered area?",
            (
                "OBSERVATION: Distances (in meters) saved to line layer: "
                "Banff Fire Station , RCMP Banff Detachment, distance=836.74 m"
            ),
        ],
        "ground_truth": "Closest pair: Banff Fire Station and RCMP Banff Detachment (836.74 m).",
    }

    sanitized = runner._sanitize_task(task)

    assert sanitized["objective"].startswith("Which fire station and police station are closest")
    assert sanitized["human_inputs"] == [sanitized["objective"]]
    assert "ground_truth" not in sanitized


def test_agents_output_logs_one_human_turn_and_keeps_terminate_answer():
    runner = EpisodeRunner()

    question = "Which fire station and police station are closest in Banff National Park?"
    task = {"objective": question, "human_inputs": [question, "OBSERVATION: leaked value"]}

    pa_result = AgentResult(
        output={"ans": "Closest pair: Banff Fire Station and RCMP Banff Detachment (836.74 m)."},
        confidence=0.9,
        tool_calls=[],
        n_turns=1,
        trace=[
            {
                "turn": 1,
                "tool_calls": [],
                "actions_in_text": [
                    {
                        "name": "Terminate",
                        "arguments": {
                            "ans": "Closest pair: Banff Fire Station and RCMP Banff Detachment (836.74 m)."
                        },
                    }
                ],
            }
        ],
    )

    out = runner._build_agents_output(
        task=task,
        planned_items=[{"agent": "ga", "sub_task": "Gather geospatial facts"}],
        dispatched_sequence=["pa"],
        results_by_agent={"pa": pa_result},
    )

    human_rows = [row for row in out if row.get("from") == "human"]
    assert len(human_rows) == 1
    assert human_rows[0]["value"] == question

    pa_rows = [row for row in out if row.get("from") == "gpt" and row.get("agent") == "pa"]
    assert pa_rows
    payload = json.loads(pa_rows[0]["value"])
    assert payload["actions"][0]["name"] == "Terminate"
    assert payload["actions"][0]["arguments"]["ans"] == "Closest pair: Banff Fire Station and RCMP Banff Detachment (836.74 m)."
