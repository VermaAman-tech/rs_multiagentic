import rasterio
from pathlib import Path

def run(req):
    try:
        path = Path(req.geotiff_path)
        if not path.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {path}")
        with rasterio.open(req.geotiff_path) as src:
            bounds = src.bounds
            bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            crs = str(src.crs)
            return {"bbox": bbox, "crs": crs, "success": True}
    except Exception as e:
        return {"bbox": [], "crs": "", "success": False, "error": str(e)}