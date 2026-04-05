import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

def run(req):
    try:
        with rasterio.open(req.index_array_path) as src:
            fig, ax = plt.subplots()
            img = ax.imshow(src.read(1), cmap="RdYlGn")
            plt.colorbar(img, ax=ax)
            out_path = req.index_array_path.replace(".tif", "_render.png")
            plt.savefig(out_path)
            plt.close()
            return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}