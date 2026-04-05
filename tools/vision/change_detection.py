import numpy as np
from PIL import Image

def run(req):
    try:
        img1 = np.array(Image.open(req.image_path_1).convert("L"))
        img2 = np.array(Image.open(req.image_path_2).convert("L"))
        diff = np.abs(img1.astype(int) - img2.astype(int))
        mask = (diff > 30).astype(np.uint8) * 255
        out_path = "data/tmp/change_mask.tif"
        Image.fromarray(mask).save(out_path)
        return {"change_map_path": out_path, "changed_pixels": int((mask > 0).sum()), "success": True}
    except Exception as e:
        return {"change_map_path": "", "changed_pixels": 0, "success": False, "error": str(e)}