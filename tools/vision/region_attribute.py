from pathlib import Path

from PIL import Image, ImageStat


def run(req):
    try:
        image_path = Path(req.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        bbox = list(getattr(req, "bbox", []))
        if len(bbox) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")

        with Image.open(image_path).convert("RGB") as img:
            w, h = img.size
            x1 = max(0, min(w - 1, int(round(float(bbox[0])))))
            y1 = max(0, min(h - 1, int(round(float(bbox[1])))))
            x2 = max(0, min(w, int(round(float(bbox[2])))))
            y2 = max(0, min(h, int(round(float(bbox[3])))))
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox is empty after clipping to image bounds")

            crop = img.crop((x1, y1, x2, y2))
            means = [round(float(x), 2) for x in ImageStat.Stat(crop).mean]
            description = (
                f"Region [{x1}, {y1}, {x2}, {y2}] in {image_path.name}: "
                f"size={x2 - x1}x{y2 - y1}, mean_rgb={means}."
            )

        return {
            "description": description,
            "success": True,
        }
    except Exception as e:
        return {
            "description": "",
            "success": False,
            "error": str(e),
        }
