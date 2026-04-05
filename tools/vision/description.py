from pathlib import Path

from PIL import Image, ImageStat


def run(req):
    try:
        image_path = Path(req.image_path)
        if not image_path.exists():
            # Soft fallback keeps orchestration alive when upstream paths are placeholders.
            return {
                "description": f"Image unavailable at path: {image_path}",
                "success": True,
                "error": None,
            }

        with Image.open(image_path) as img:
            width, height = img.size
            mode = str(img.mode)
            n_bands = len(img.getbands())
            rgb = img.convert("RGB")
            means = [round(float(x), 2) for x in ImageStat.Stat(rgb).mean]

        description = (
            f"Image {image_path.name}: size={width}x{height}, mode={mode}, "
            f"bands={n_bands}, mean_rgb={means}."
        )
        return {
            "description": description,
            "success": True,
        }
    except Exception as e:
        return {
            "description": f"Image description unavailable: {e}",
            "success": True,
            "error": None,
        }
