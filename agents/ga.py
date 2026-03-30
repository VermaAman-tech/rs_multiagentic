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
                "name": "DisplayOnMap",
                "description": "Render geospatial features on a map.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "features": {"type": "array", "items": {"type": "object"}},
                        "output_html": {"type": "string"},
                    },
                    "required": ["features", "output_html"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "GetBboxFromGeotiff",
                "description": "Extract bounding box and CRS from a GeoTIFF.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "geotiff_path": {"type": "string"},
                    },
                    "required": ["geotiff_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "DisplayOnGeotiff",
                "description": "Render GIS overlays on a GeoTIFF.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "geotiff_path": {"type": "string"},
                        "features": {"type": "array", "items": {"type": "object"}},
                        "output_path": {"type": "string"},
                    },
                    "required": ["geotiff_path", "features", "output_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "RoadDamageScorer",
                "description": "Score road segment damage from raster overlays.",
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
    ]

    def _get_system_prompt(self) -> str:
        return (
            "You are GA, the Geospatial Agent in MAGRF.\n\n"
            "Core role:\n"
            "- Convert place-based requests into verifiable geospatial operations.\n"
            "- Build GIS layers that support routing and risk-aware planning.\n"
            "- Quantify spatial relationships instead of describing them loosely.\n\n"
            "ReAct discipline (mandatory):\n"
            "- THINK: determine required geospatial entities, CRS expectations, and spatial relationships.\n"
            "- ACT: call one GIS tool at a time with clear parameter intent.\n"
            "- OBSERVE: validate whether returned geometry/statistics are sufficient and coherent.\n"
            "- THINK AGAIN: refine analysis or hand off when ready.\n"
            "Every spatial claim must be traceable to a tool output.\n\n"
            "Owned tools:\n"
            "GetAreaBoundary, AddPoisLayer, GetBboxFromGeotiff, ComputeDistance, DisplayOnMap, DisplayOnGeotiff, RoadDamageScorer.\n\n"
            "Tool strategy:\n"
            "- Use GetAreaBoundary/GetBboxFromGeotiff to anchor analysis extent before downstream steps.\n"
            "- Use AddPoisLayer for relevant infrastructure entities.\n"
            "- Use RoadDamageScorer for road-state estimation required by planning.\n"
            "- Use DisplayOnMap/DisplayOnGeotiff to produce human-auditable visual artifacts.\n\n"
            "Handoff contract to ORC:\n"
            "- Return structured GIS outputs with layer names/paths and key spatial findings.\n"
            "- Include uncertainty notes when geometry quality, coverage, or assumptions may affect planning.\n"
            "- Do not perform route optimization; PA owns final route synthesis."
        )

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.GA_TOOLS
        return super().handle_task(message, tools_schema)
