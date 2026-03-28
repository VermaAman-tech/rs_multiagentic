from agents.vra import VisionReasoningAgent
from agents.ga import GeospatialAgent
from agents.pa import PlanningAgent
from agents.orc import Orchestrator
from framework.mpc.message import Message

def test_vra_react_mock():
    vra = VisionReasoningAgent()
    msg = Message(message_type="TASK", sender="orc", recipient="vra", payload={"scene_id": "test_scene"})
    res = vra.handle_task(msg)
    assert res.output["scene_id"] == "test_scene"
    assert "damage_polygons" in res.output

def test_ga_react_mock():
    ga = GeospatialAgent()
    msg = Message(message_type="TASK", sender="orc", recipient="ga", payload={"region": "Texas"})
    res = ga.handle_task(msg)
    assert res.output["region"] == "Texas"

def test_pa_react_mock():
    pa = PlanningAgent()
    msg = Message(message_type="TASK", sender="orc", recipient="pa", payload={"objective": "rescue"})
    res = pa.handle_task(msg)
    assert res.output["objective"] == "rescue"
