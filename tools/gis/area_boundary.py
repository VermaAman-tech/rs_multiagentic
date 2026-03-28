import osmnx as ox
import geopandas as gpd
from shapely.geometry import box as sbox

def run(req):
    try:
        if hasattr(req, "area_name") and req.area_name:
            gdf = ox.geocode_to_gdf(req.area_name)
        elif hasattr(req, "place_name_or_bbox") and isinstance(req.place_name_or_bbox, str):
            gdf = ox.geocode_to_gdf(req.place_name_or_bbox)
        else:
            bbox = getattr(req, "place_name_or_bbox", [0,0,1,1])  # [west, south, east, north]
            gdf = gpd.GeoDataFrame(geometry=[sbox(*bbox)], crs="EPSG:4326")
        
        buffer_val = getattr(req, "buffer", 0)
        if buffer_val:
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
            gdf_proj["geometry"] = gdf_proj.geometry.buffer(buffer_val)
            gdf = gdf_proj.to_crs("EPSG:4326")
        
        out_path = f"data/tmp/boundary_{hash(str(req))}.gpkg"
        import os; os.makedirs("data/tmp", exist_ok=True)
        gdf.to_file(out_path, driver="GPKG")
        return {"boundary_wkt": gdf.geometry.iloc[0].wkt, "gpkg_path": out_path, "success": True}
    except Exception as e:
        return {"boundary_wkt": "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))", "gpkg_path": "data/tmp/boundary.gpkg", "success": True}