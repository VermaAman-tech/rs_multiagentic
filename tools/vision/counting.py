from tools.vision.object_detection import run as od_run
from tools.schemas import ObjectDetectionInput

def run(req):
    try:
        od_req = ObjectDetectionInput(image_path=req.image_path, text_prompt=req.text_prompt)
        res = od_run(od_req)
        if res.get("success"):
            return {"count": len(res.get("bboxes", [])), "success": True}
        return {"count": 0, "success": False, "error": res.get("error")}
    except Exception as e:
        return {"count": 0, "success": False, "error": str(e)}