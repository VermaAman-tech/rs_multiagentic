def run(req):
    # Image description relies on VRA's direct multimodal capabilities.
    # The framework design lets VRA read the image using its Qwen3-VL core inside the ReAct loop.
    return {
        "description": "Image loaded into VRA context. VRA can now describe it directly.", 
        "success": True
    }
