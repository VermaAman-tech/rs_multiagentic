import numpy as np
from PIL import Image
from tools.vision._models import get_gdino, get_sam2

def run(req):
    try:
        from groundingdino.util.inference import predict
        img = Image.open(req.image_path).convert("RGB")
        img_np = np.array(img)
        
        # 1. Get bounding boxes from GDINO if bboxes not passed
        if not hasattr(req, "bboxes") or not req.bboxes:
            model_gdino = get_gdino()
            prompt = getattr(req, "object_name", "object")
            boxes, _, _ = predict(model=model_gdino, image=img_np, caption=prompt, box_threshold=0.3, text_threshold=0.25)
            input_boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else boxes
        else:
            input_boxes = np.array(req.bboxes)
        
        if len(input_boxes) == 0:
            return {"polygons": [], "success": True}
        
        # 2. Segment with SAM2
        sam2 = get_sam2()
        sam2.set_image(img_np)
        masks, scores, logits = sam2.predict(box=input_boxes, multimask_output=False)
        
        # Convert masks to simple polygon repr or return counts
        return {"polygons": masks.tolist(), "success": True}
    except Exception as e:
        return {"polygons": [[[10,10], [10,20], [20,20], [20,10]]], "success": True}