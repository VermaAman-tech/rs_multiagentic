import pytest

from agents.pa import PlanningAgent
from framework.mpc.message import Message


def test_pa_uses_observation_when_ga_context_is_placeholder():
    pa = PlanningAgent()

    objective = (
        "OBSERVATION: Distances (in meters) saved to line layer: "
        "'fire_stations_to_police_stations_distances': "
        "Banff Fire Station , RCMP Banff Detachment, distance=836.74 m, travel_time=100.4 s "
        "Fire Hall , RCMP Banff Detachment, distance=17838.97 m, travel_time=1380.4 s "
        "distances = [836.74, 17838.97] travel_times = [100.4, 1380.4] "
        "Please summarize the model outputs and answer my first question."
    )

    payload = {
        "objective": objective,
        "query_type": "geospatial_proximity",
        "scene_id": "oea-case-0-live",
        "region": "oea",
        "episodic_context": {
            "ga_result": {
                "closest_pair": {
                    "fire_station": {"name": "fire_station_candidate", "lat": 51.0756, "lon": -116.3003},
                    "police_station": {"name": "police_station_candidate", "lat": 51.0756, "lon": -116.3003},
                    "distance_meters": 0.0,
                },
                "distance_meters": 0.0,
                "distance_evidence": {"source": "ga_tool_calls"},
                "raw_output": "Closest pair: fire_station_candidate and police_station_candidate (0.00 m).",
            }
        },
    }

    msg = Message(
        message_type="TASK",
        sender="orc",
        recipient="pa",
        payload=payload,
    )
    res = pa.handle_task(msg)

    assert res.output["distance_meters"] == pytest.approx(836.74)
    assert res.output["distance_evidence"]["source"] == "observation_payload"
    assert "Banff Fire Station" in res.output["ans"]
    assert "RCMP Banff Detachment" in res.output["ans"]
