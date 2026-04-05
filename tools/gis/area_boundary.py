import osmnx as ox
import geopandas as gpd
from shapely.geometry import box as sbox
import re


def _normalize_area_query(text: str) -> tuple[str, int]:
    q = " ".join(str(text or "").split()).strip()
    if not q:
        return "", 0

    ql = q.lower()
    extracted_buffer = 0

    # Patterns like "within 2000m radius of X", "2000m radius around X", etc.
    m = re.search(r"(?:within\s+)?(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\s*(?:radius\s*(?:of|around)?\s+)(.+)", ql)
    if m:
        try:
            extracted_buffer = int(round(float(m.group(1))))
        except Exception:
            extracted_buffer = 0
        q = q[m.start(2):].strip()

    # Strip common wrappers while preserving the real place name.
    q = re.sub(r"^(within|around|near|in)\s+", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"\s*(within|around|near)\s+\d+(?:\.\d+)?\s*(?:m|meter|meters).*$", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"\s*in\s+[0-9]+(?:\.[0-9]+)?\s*(?:m|meter|meters)\s*radius.*$", "", q, flags=re.IGNORECASE).strip()

    return q, extracted_buffer

def run(req):
    try:
        area_name = getattr(req, "area_name", None)
        place_name_or_bbox = getattr(req, "place_name_or_bbox", None)

        normalized_q = ""
        inferred_buffer = 0
        if isinstance(area_name, str) and area_name.strip():
            normalized_q, inferred_buffer = _normalize_area_query(area_name)
        elif isinstance(place_name_or_bbox, str) and place_name_or_bbox.strip():
            normalized_q, inferred_buffer = _normalize_area_query(place_name_or_bbox)

        if hasattr(req, "area_name") and req.area_name:
            gdf = ox.geocode_to_gdf(normalized_q or req.area_name)
        elif hasattr(req, "place_name_or_bbox") and isinstance(req.place_name_or_bbox, str):
            gdf = ox.geocode_to_gdf(normalized_q or req.place_name_or_bbox)
        else:
            bbox = getattr(req, "place_name_or_bbox", [0,0,1,1])  # [west, south, east, north]
            gdf = gpd.GeoDataFrame(geometry=[sbox(*bbox)], crs="EPSG:4326")
        
        buffer_val = int(getattr(req, "buffer_m", 0) or 0)
        if buffer_val <= 0 and inferred_buffer > 0:
            buffer_val = inferred_buffer
        if buffer_val:
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
            gdf_proj["geometry"] = gdf_proj.geometry.buffer(buffer_val)
            gdf = gdf_proj.to_crs("EPSG:4326")
        
        out_path = f"data/tmp/boundary_{hash(str(req))}.gpkg"
        import os; os.makedirs("data/tmp", exist_ok=True)
        gdf.to_file(out_path, driver="GPKG")
        geom = gdf.geometry.iloc[0]
        bounds = list(geom.bounds) if hasattr(geom, "bounds") else [0.0, 0.0, 1.0, 1.0]
        return {
            "boundary_wkt": geom.wkt,
            "gpkg_path": out_path,
            "bbox": [float(x) for x in bounds],
            "success": True,
        }
    except Exception as e:
        return {
            "boundary_wkt": "",
            "gpkg_path": "",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "success": False,
            "error": str(e),
        }