import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

def run(req):
    try:
        out_path = getattr(req, "output_path", f"data/tmp/map_{hash(str(req))}.png")
        with rasterio.open(req.geotiff_path) as src:
            fig, ax = plt.subplots(figsize=(10, 10))
            show(src, ax=ax)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            return {"raster_path": out_path, "success": True}
    except Exception as e:
        return {"raster_path": "data/tmp/map.png", "success": False}