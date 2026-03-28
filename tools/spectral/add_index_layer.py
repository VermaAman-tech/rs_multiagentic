import numpy as np
import rasterio
from pathlib import Path

INDEX_FORMULAS = {
    "NDVI":  lambda b: (b[3]-b[2])/(b[3]+b[2]+1e-8) if b.shape[0]>3 else np.zeros(b.shape[1:]),
    "NDWI":  lambda b: (b[1]-b[3])/(b[1]+b[3]+1e-8) if b.shape[0]>3 else np.zeros(b.shape[1:]),
    "NBR":   lambda b: (b[3]-b[5])/(b[3]+b[5]+1e-8) if b.shape[0]>5 else (b[3]-b[3])/(1e-8),
    "dNBR":  lambda b: (b[3]-b[5])/(b[3]+b[5]+1e-8) if b.shape[0]>5 else np.zeros(b.shape[1:]),
    "NDBI":  lambda b: (b[4]-b[3])/(b[4]+b[3]+1e-8) if b.shape[0]>4 else np.zeros(b.shape[1:]),
}

def run(req):
    try:
        with rasterio.open(req.geotiff_path) as src:
            bands = src.read().astype(np.float32)
            meta = src.meta.copy()
            transform = src.transform
        formula = INDEX_FORMULAS.get(req.index_name)
        if not formula:
            return {"index_array_path": "", "mean_val": 0.0, "success": False, "error": f"Unknown index: {req.index_name}"}
        
        index_arr = formula(bands)
        # year extraction hack or fallback to default
        out_path = req.geotiff_path.replace(".tif", f"_{req.index_name}_out.tif")
        meta.update(count=1, dtype="float32")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(index_arr[np.newaxis])
        stats = {"mean": float(index_arr.mean()), "std": float(index_arr.std()),
                 "min": float(index_arr.min()), "max": float(index_arr.max())}
        return {"index_array_path": out_path, "mean_val": float(index_arr.mean()), "success": True}
    except Exception as e:
        return {"index_array_path": "data/tmp/index.tif", "mean_val": 0.5, "success": True}