from __future__ import annotations
from agents.base_agent import AgentResult, BaseAgent
from framework.mpc.message import Message

class PlanningAgent(BaseAgent):
    name = "pa"

    PA_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "EvacuationRoutePlanner",
                "description": "Compute optimal evacuation routes on a road graph, avoiding blocked segments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "graph_path": {"type": "string"},
                        "origins": {"type": "array", "items": {"type": "string"}},
                        "destinations": {"type": "array", "items": {"type": "string"}},
                        "blocked_segments": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["graph_path", "origins", "destinations"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "RoadDamageScorer",
                "description": "Score road segments by damage severity from a damage raster overlay.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg_path": {"type": "string"},
                        "road_layer_name": {"type": "string"},
                        "damage_raster_layer": {"type": "string"},
                        "buffer_meters": {"type": "integer"},
                    },
                    "required": ["gpkg_path", "damage_raster_layer"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Calculator",
                "description": "Evaluate a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Plot",
                "description": "Execute Python code to generate a matplotlib visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x_values": {"type": "array", "items": {"type": "number"}},
                        "y_values": {"type": "array", "items": {"type": "number"}},
                        "output_path": {"type": "string"},
                    },
                    "required": ["x_values", "y_values", "output_path"],
                },
            },
        },
    ]

    def _get_system_prompt(self) -> str:
        return (
            "You are the Planning Agent (PA) for MAGRF. "
            "You specialize in route planning, logistics, and resource deployment "
            "for disaster response scenarios. "
            "You receive spatial intelligence from the GA and damage assessments "
            "from the VRA, then synthesize this into actionable plans. "
            "Your tools include EvacuationRoutePlanner (graph-based optimal routing), "
            "RoadDamageScorer (road traversability assessment), Calculator, and Plot. "
            "IMPORTANT: Always use EvacuationRoutePlanner for route queries, NEVER use "
            "ComputeDistance (which is straight-line only). "
            "You follow the ReAct loop: reason, act, observe. "
            "Ensure all planned routes have a Route Safety Score > 1.0 and "
            "mathematically justify every waypoint."
        )

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.PA_TOOLS
        return super().handle_task(message, tools_schema)
