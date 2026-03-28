from __future__ import annotations
from agents.base_agent import AgentResult, BaseAgent
from framework.mpc.message import Message

class GeospatialAgent(BaseAgent):
    name = "ga"

    # Tool schemas this agent can use (OpenAI function-calling format)
    GA_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "GetAreaBoundary",
                "description": "Retrieve the geographic boundary polygon for a named area with optional buffer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area_name": {"type": "string", "description": "Name of the geographic area"},
                        "buffer_m": {"type": "integer", "description": "Buffer in meters (optional)"},
                    },
                    "required": ["area_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "AddPoisLayer",
                "description": "Generate POIs by category inside a bounding box.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "poi_category": {"type": "string", "description": "POI category (e.g. shelter, hospital)"},
                        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    },
                    "required": ["poi_category", "bbox"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ComputeDistance",
                "description": "Compute distance in meters between two points [lon, lat].",
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
                "name": "AddIndexLayer",
                "description": "Compute a spectral index layer from a GeoTIFF.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "geotiff_path": {"type": "string"},
                        "index_name": {"type": "string"},
                    },
                    "required": ["geotiff_path", "index_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ComputeIndexChange",
                "description": "Compute the difference between two spectral index layers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_path_pre": {"type": "string"},
                        "index_path_post": {"type": "string"},
                    },
                    "required": ["index_path_pre", "index_path_post"],
                },
            },
        },
    ]

    def _get_system_prompt(self) -> str:
        return (
            "You are the Geospatial Agent (GA) for MAGRF. "
            "You specialize in GIS analysis: vector data manipulation, boundary extraction, "
            "POI querying (via OSM), spatial distance computation, spectral index analysis, "
            "and CRS transformations. "
            "You follow the ReAct (Reason, Act, Observe) loop: explain your reasoning, "
            "call the appropriate tool, then analyze the observation before deciding "
            "the next step. "
            "Your tools include GetAreaBoundary, AddPoisLayer, ComputeDistance, "
            "AddIndexLayer, and ComputeIndexChange. "
            "Always use specific tool calls rather than guessing spatial data."
        )

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.GA_TOOLS
        return super().handle_task(message, tools_schema)
