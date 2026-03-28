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
    ]

    def _get_system_prompt(self) -> str:
        return (
            "You are the Vision Reasoning Agent (VRA) for MAGRF. "
            "You specialize in analyzing satellite and aerial imagery: "
            "object detection, change detection between temporal images, "
            "counting objects, segmentation, and image description. "
            "You follow the ReAct loop: reason about what visual analysis "
            "is needed, call the appropriate vision tool, observe the result, "
            "and decide the next step. "
            "Your tools include ObjectDetection, TextToBbox, ImageDescription, "
            "ChangeDetection, CountGivenObject, SegmentObjectPixels, and "
            "RegionAttributeDescription. "
            "Always provide detailed visual evidence before making any claims "
            "about damage, flooding, or other conditions."
        )

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = self.VRA_TOOLS
        return super().handle_task(message, tools_schema)
