from PIL import Image
import numpy as np

from tools.vision._models import get_sam2

def run(req):
    try:
        img = Image.open(req.image_path).convert("RGB")
        img_np = np.array(img)

        if not hasattr(req, "bboxes") or not req.bboxes:
            return {
                "polygons": [],
                "success": False,
                "error": "bboxes are required for SegmentObjectPixels",
            }

        input_boxes = np.array(req.bboxes, dtype=np.float32)
        if len(input_boxes) == 0:
            return {"polygons": [], "success": True}

        # 2. Segment with SAM2
        sam2 = get_sam2()
        sam2.set_image(img_np)
        masks, _scores, _logits = sam2.predict(box=input_boxes, multimask_output=False)

        # Convert masks to simple polygon repr or return counts
        return {"polygons": masks.tolist(), "success": True}
    except Exception as e:
        return {"polygons": [], "success": False, "error": str(e)}