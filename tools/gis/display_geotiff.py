import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from pathlib import Path

def run(req):
    try:
        if not Path(req.geotiff_path).exists():
            raise FileNotFoundError(f"GeoTIFF not found: {req.geotiff_path}")
        out_path = getattr(req, "output_path", f"data/tmp/map_{hash(str(req))}.png")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(req.geotiff_path) as src:
            fig, ax = plt.subplots(figsize=(10, 10))
            show(src, ax=ax)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            return {"raster_path": out_path, "success": True}
    except Exception as e:
        return {"raster_path": "", "success": False, "error": str(e)}