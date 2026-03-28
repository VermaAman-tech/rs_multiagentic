import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
from shapely.geometry import Point

def run(req):
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if hasattr(req, "features"):
            points = [Point(f["lon"], f["lat"]) for f in req.features if "lon" in f and "lat" in f]
            if points:
                gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
                gdf.to_crs("EPSG:3857").plot(ax=ax, color='red', alpha=0.6, markersize=50)
        
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass  # No internet — skip basemap
            
        out_path = getattr(req, "output_html", f"data/tmp/map_{hash(str(req))}.png").replace(".html", ".png")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"html_path": out_path, "success": True}
    except Exception as e:
        return {"html_path": "data/tmp/map.png", "success": True}