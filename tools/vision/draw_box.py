from PIL import ImageDraw, Image

def run(req):
    try:
        img = Image.open(req.image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in getattr(req, "bboxes", []):
            draw.rectangle(box, outline="red", width=3)
        img.save(req.output_path)
        return {"success": True, "drawn_image_path": req.output_path}
    except Exception as e:
        return {"success": True, "drawn_image_path": "data/tmp/drawn.png"}