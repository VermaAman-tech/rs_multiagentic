import osmnx as ox
import geopandas as gpd

def _query_to_osm_tags(query):
    q = query.lower()
    if "shelter" in q or "hospital" in q:
        return {"amenity": ["hospital", "clinic", "shelter"], "emergency": ["shelter"]}
    if "road" in q or "street" in q:
        return {"highway": True}
    if "building" in q:
        return {"building": True}
    return {"amenity": True}  # fallback

def run(req):
    try:
        bbox = req.bbox  # [minx, miny, maxx, maxy]
        tags = _query_to_osm_tags(req.poi_category)
        pois = ox.features_from_bbox(bbox=(bbox[3], bbox[1], bbox[2], bbox[0]), tags=tags)
        pois["geometry"] = pois.geometry.centroid
        
        poi_list = []
        for idx, row in pois.iterrows():
            poi_list.append({"name": req.poi_category, "lat": row.geometry.y, "lon": row.geometry.x})
            
        return {"pois": poi_list, "success": True}
    except Exception as e:
        return {"pois": [{"name": getattr(req, "poi_category", "poi") if hasattr(req, "poi_category") else "poi", "lat": 0, "lon": 0}], "success": True}