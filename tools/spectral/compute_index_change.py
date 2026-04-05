import numpy as np
import rasterio
from pathlib import Path

def run(req):
    try:
        with rasterio.open(req.index_path_pre) as pre_src, rasterio.open(req.index_path_post) as post_src:
            pre_arr = pre_src.read(1).astype(np.float32)
            post_arr = post_src.read(1).astype(np.float32)
            diff = post_arr - pre_arr
            
            meta = pre_src.meta.copy()
            meta.update(driver="GTiff", dtype="float32", count=1)
            out_path = f"data/tmp/diff_{hash(req.index_path_pre)}.tif"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(diff, 1)
                
            return {"diff_path": out_path, "mean_diff": float(diff.mean()), "success": True}
    except Exception as e:
        return {"diff_path": "", "mean_diff": 0.0, "success": False, "error": str(e)}