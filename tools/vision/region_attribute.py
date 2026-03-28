def run(req):
    # Same logic as ImageDescription, VRA uses its vision encoder directly on a cropped bounding box if needed.
    return {
        "description": f"Focusing on bounding box {req.bbox}. Processing natively via VRA vision encoder.",
        "success": True
    }
