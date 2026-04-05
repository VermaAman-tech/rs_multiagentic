from PIL import ImageDraw, Image
from pathlib import Path

def run(req):
    try:
        img = Image.open(req.image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in getattr(req, "bboxes", []):
            draw.rectangle(box, outline="red", width=3)
        Path(req.output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(req.output_path)
        return {"success": True, "drawn_image_path": req.output_path}
    except Exception as e:
        return {"success": False, "drawn_image_path": "", "error": str(e)}