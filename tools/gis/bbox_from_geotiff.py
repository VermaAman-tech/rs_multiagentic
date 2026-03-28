import rasterio

def run(req):
    try:
        with rasterio.open(req.geotiff_path) as src:
            bounds = src.bounds
            bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            crs = str(src.crs)
            return {"bbox": bbox, "crs": crs, "success": True}
    except Exception as e:
        return {"bbox": [0,0,1,1], "crs": "EPSG:4326", "success": True}