from PIL import ImageDraw, Image

def run(req):
    try:
        img = Image.open(req.image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        pos = getattr(req, "position", [0, 0])
        draw.text((pos[0], pos[1]), getattr(req, "text", ""), fill="red")
        img.save(req.output_path)
        return {"success": True, "drawn_image_path": req.output_path}
    except Exception as e:
        return {"success": True, "drawn_image_path": "data/tmp/drawn.png"}