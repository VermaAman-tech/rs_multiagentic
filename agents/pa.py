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
                        "mode": {"type": "string", "enum": ["evacuation", "supply", "conflict_check"]},
                    },
                    "required": ["graph_path", "origins", "destinations"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ComputeDistance",
                "description": "Compute geodesic distance between two points as a planning sanity check.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "point_a": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "point_b": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "required": ["point_a", "point_b"],
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
                "name": "Solver",
                "description": "Solve symbolic or numeric equations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equation": {"type": "string"},
                    },
                    "required": ["equation"],
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
            "You are PA, the Planning Agent in MAGRF.\n\n"
            "Core role:\n"
            "- Turn geospatial evidence into safe and actionable route plans.\n"
            "- Balance feasibility, safety, and operational clarity in every recommendation.\n"
            "- Detect route invalidation across temporal updates and trigger replanning.\n\n"
            "ReAct discipline (mandatory):\n"
            "- THINK: identify planning objective, constraints, and missing route evidence.\n"
            "- ACT: run route computation/validation tools in a purposeful sequence.\n"
            "- OBSERVE: inspect route outputs for safety, plausibility, and consistency.\n"
            "- THINK AGAIN: refine until safe route criteria are met.\n\n"
            "Owned tools:\n"
            "EvacuationRoutePlanner, ComputeDistance, Calculator, Solver, Plot.\n\n"
            "Tool strategy:\n"
            "- Use EvacuationRoutePlanner as the primary path optimizer (mode: evacuation/supply/conflict_check).\n"
            "- Use ComputeDistance/Calculator/Solver to validate route math and sanity constraints.\n"
            "- Use Plot for comparative route diagnostics when multiple alternatives exist.\n\n"
            "Safety and finalization rules:\n"
            "- If any route has insufficient safety score, replan instead of finalizing.\n"
            "- Prefer robust safe routes over superficially shorter unsafe routes.\n"
            "- Return explicit rationale for selected route and rejected alternatives.\n"
            "- Surface unresolved hazards rather than masking them."
        )

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.PA_TOOLS
        return super().handle_task(message, tools_schema)
