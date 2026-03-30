from __future__ import annotations
from agents.base_agent import AgentResult, BaseAgent
from framework.mpc.message import Message

class VisionReasoningAgent(BaseAgent):
    name = "vra"

    VRA_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "ObjectDetection",
                "description": "Detect common objects in an image, returning bounding boxes and labels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "text_prompt": {"type": "string", "description": "Object to detect"},
                    },
                    "required": ["image_path", "text_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "TextToBbox",
                "description": "Detect objects matching a text description, returning bounding boxes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "text_prompt": {"type": "string"},
                    },
                    "required": ["image_path", "text_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ImageDescription",
                "description": "Generate a natural-language description of an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                    },
                    "required": ["image_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ChangeDetection",
                "description": "Analyze pre/post event images to detect and describe changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path_1": {"type": "string"},
                        "image_path_2": {"type": "string"},
                    },
                    "required": ["image_path_1", "image_path_2"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "CountGivenObject",
                "description": "Count the number of specified objects in an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "text_prompt": {"type": "string"},
                    },
                    "required": ["image_path", "text_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "SegmentObjectPixels",
                "description": "Segment specified objects and return pixel-level analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bboxes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                    "required": ["image_path", "bboxes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "RegionAttributeDescription",
                "description": "Describe a specific attribute of a region in an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["image_path", "bbox"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "OCR",
                "description": "Extract text from image regions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                    },
                    "required": ["image_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "DrawBox",
                "description": "Draw boxes on an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bboxes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        "output_path": {"type": "string"},
                    },
                    "required": ["image_path", "bboxes", "output_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "AddText",
                "description": "Add text labels on an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "text": {"type": "string"},
                        "position": {"type": "array", "items": {"type": "integer"}},
                        "output_path": {"type": "string"},
                    },
                    "required": ["image_path", "text", "position", "output_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "GetBboxFromGeotiff",
                "description": "Extract geospatial bounds from a GeoTIFF before downstream visual/spectral analysis.",
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
                "name": "AddIndexLayer",
                "description": "Compute spectral indices from a GeoTIFF.",
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
                "description": "Compute index difference between pre/post raster index layers.",
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
        {
            "type": "function",
            "function": {
                "name": "ShowIndexLayer",
                "description": "Render a computed index raster layer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index_array_path": {"type": "string"},
                    },
                    "required": ["index_array_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "TemporalStackLoader",
                "description": "Load a temporal stack for an AOI and date range.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "aoi_bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                        "date_range": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                        "sensor": {"type": "string"},
                        "max_cloud_pct": {"type": "number"},
                    },
                    "required": ["aoi_bbox", "date_range"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "PrithviEmbed",
                "description": "Compute Prithvi embeddings for a raster.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "raster_path": {"type": "string"},
                    },
                    "required": ["raster_path"],
                },
            },
        },
    ]

    def _get_system_prompt(self) -> str:
        return (
            "You are VRA, the Vision Reasoning Agent in MAGRF.\n\n"
            "Core role:\n"
            "- Interpret remote-sensing imagery and temporal evidence.\n"
            "- Produce grounded visual facts for downstream GIS and planning decisions.\n"
            "- Convert visual uncertainty into explicit confidence statements.\n\n"
            "ReAct discipline (mandatory):\n"
            "- THINK: identify what is missing, what can be observed, and what can be measured.\n"
            "- ACT: call exactly the next most informative tool with correctly typed arguments.\n"
            "- OBSERVE: inspect tool outputs critically; verify they answer the intended sub-question.\n"
            "- THINK AGAIN: decide whether to continue, refine, or hand off.\n"
            "Never produce a final claim without at least one supporting observation from tools.\n\n"
            "Owned tools:\n"
            "GetBboxFromGeotiff, ObjectDetection, SegmentObjectPixels, ImageDescription, TextToBbox,\n"
            "RegionAttributeDescription, CountGivenObject, OCR, ChangeDetection, DrawBox, AddText,\n"
            "AddIndexLayer, ComputeIndexChange, ShowIndexLayer, TemporalStackLoader, PrithviEmbed.\n\n"
            "Tool strategy:\n"
            "- Use TemporalStackLoader when the question spans multiple dates or epochs.\n"
            "- Use PrithviEmbed for multispectral or temporal feature extraction when available.\n"
            "- Use AddIndexLayer/ComputeIndexChange/ShowIndexLayer when index-based evidence is needed.\n"
            "- Prefer concise tool chains that maximize evidence quality per call.\n\n"
            "Output contract to ORC:\n"
            "- Return structured findings with: what was observed, how it was measured, and confidence.\n"
            "- Separate facts from interpretations.\n"
            "- Explicitly note uncertainty sources (resolution limits, occlusion, weak detections).\n"
            "- Avoid unsupported extrapolation beyond observed data."
        )

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.VRA_TOOLS
        return super().handle_task(message, tools_schema)
