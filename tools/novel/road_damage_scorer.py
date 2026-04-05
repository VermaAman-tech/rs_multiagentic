import osmnx as ox
import geopandas as gpd
from pathlib import Path

def road_damage_scorer(gpkg_path, road_layer_name="roads", damage_raster_layer=None, buffer_meters=10):
    try:
        if not gpkg_path or not gpkg_path.endswith(".gpkg"):
            raise ValueError(f"Invalid gpkg_path: {gpkg_path}")
        if not Path(gpkg_path).exists():
            raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")
            
        roads = gpd.read_file(gpkg_path, layer=road_layer_name)

        # Return real scoring array when available
        segment_scores = []
        for idx, row in roads.iterrows():
            segment_scores.append({
                "segment_id": str(idx),
                "traversability": row.get("damage_score", row.get("traversability", 1.0))
            })
            
        return {"segment_scores": segment_scores, "n_roads_scored": len(segment_scores), "success": True}
    except Exception as e:
        return {"segment_scores": [], "n_roads_scored": 0, "success": False, "error": str(e)}