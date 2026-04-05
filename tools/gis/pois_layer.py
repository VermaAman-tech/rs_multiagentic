import math

import osmnx as ox
from shapely.geometry import box as sbox


def _normalize_category(query: str) -> str:
    q = str(query or "").lower().replace("_", " ")
    q = " ".join(q.split())
    repl = {
        "fast foods": "fast food",
        "fire stations": "fire station",
        "police stations": "police station",
        "fuel stations": "fuel station",
        "post offices": "post office",
        "markets": "market",
        "marketplaces": "marketplace",
    }
    for a, b in repl.items():
        q = q.replace(a, b)
    return q


def _normalize_bbox(bbox: list[float]) -> list[float]:
    minx, miny, maxx, maxy = [float(v) for v in bbox]
    # Clamp to valid lon/lat ranges.
    minx = max(-180.0, min(180.0, minx))
    maxx = max(-180.0, min(180.0, maxx))
    miny = max(-90.0, min(90.0, miny))
    maxy = max(-90.0, min(90.0, maxy))
    if maxx < minx:
        minx, maxx = maxx, minx
    if maxy < miny:
        miny, maxy = maxy, miny

    # Very large extents often trigger long OSM requests; shrink to a practical window around center.
    max_span = 1.2
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half_w = min((maxx - minx) / 2.0, max_span / 2.0)
    half_h = min((maxy - miny) / 2.0, max_span / 2.0)
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

def _query_to_osm_tags(query):
    q = _normalize_category(query)
    if any(k in q for k in ("shelter", "hospital", "clinic")):
        return {"amenity": ["hospital", "clinic", "shelter"], "emergency": ["shelter"]}
    if "fire" in q:
        return {"amenity": ["fire_station"]}
    if "police" in q:
        return {"amenity": ["police"]}
    if "school" in q:
        return {"amenity": ["school", "college", "university"]}
    if "restaurant" in q:
        return {"amenity": ["restaurant", "fast_food", "cafe"]}
    if "park" in q or "garden" in q or "nature reserve" in q or "green space" in q:
        return {"leisure": ["park", "garden", "nature_reserve", "common"]}
    if "bank" in q:
        return {"amenity": ["bank", "atm"]}
    if "hotel" in q:
        return {"tourism": ["hotel", "motel", "hostel", "guest_house"]}
    if "supermarket" in q:
        return {"shop": ["supermarket", "convenience", "department_store"]}
    if "marketplace" in q or "market" in q:
        return {"shop": ["marketplace", "supermarket", "convenience"]}
    if "atm" in q:
        return {"amenity": ["atm", "bank"]}
    if "courthouse" in q or "court" in q:
        return {"amenity": ["courthouse"]}
    if "post office" in q or "post" in q:
        return {"amenity": ["post_office"]}
    if "hostel" in q:
        return {"tourism": ["hostel", "guest_house", "hotel"]}
    if "fuel station" in q:
        return {"amenity": ["fuel"]}
    if "fast food" in q:
        return {"amenity": ["fast_food", "restaurant", "cafe"]}
    if "parking" in q:
        return {"amenity": ["parking", "parking_entrance", "parking_space"]}
    if "bus" in q:
        return {"highway": ["bus_stop"], "public_transport": ["platform", "station"]}
    if "train" in q:
        return {"railway": ["station", "halt", "tram_stop"]}
    if "airport" in q:
        return {"aeroway": ["aerodrome", "terminal", "gate"]}
    if "pharmacy" in q:
        return {"amenity": ["pharmacy"]}
    if "gas" in q or "fuel" in q:
        return {"amenity": ["fuel"]}
    if "mall" in q or "shopping centre" in q or "shopping center" in q or "shopping" in q:
        return {"shop": ["mall", "department_store", "supermarket"], "amenity": ["marketplace"]}
    if "road" in q or "street" in q:
        return {"highway": True}
    if "building" in q:
        return {"building": True}
    return {"amenity": ["restaurant", "cafe", "bank", "school", "hospital", "pharmacy"]}  # bounded fallback

def run(req):
    try:
        bbox = _normalize_bbox(req.bbox)  # [minx, miny, maxx, maxy]
        tags = _query_to_osm_tags(req.poi_category)
        north, south, east, west = bbox[3], bbox[1], bbox[2], bbox[0]
        ox.settings.use_cache = True
        ox.settings.timeout = max(int(getattr(ox.settings, "timeout", 60)), 120)
        ox.settings.requests_timeout = max(int(getattr(ox.settings, "requests_timeout", 60)), 120)
        try:
            try:
                pois = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
            except TypeError:
                # OSMnx API changed across versions; keep compatibility for both signatures.
                pois = ox.features_from_bbox(north=north, south=south, east=east, west=west, tags=tags)
        except Exception as bbox_exc:
            # Fallback: polygon query can be more robust on some OSMnx/GEOS combinations.
            try:
                pois = ox.features_from_polygon(sbox(west, south, east, north), tags=tags)
            except Exception as poly_exc:
                # Final retry: slightly shrink bbox to reduce query complexity.
                cx = (west + east) / 2.0
                cy = (south + north) / 2.0
                half_w = max(1e-6, (east - west) * 0.4)
                half_h = max(1e-6, (north - south) * 0.4)
                west2, east2 = cx - half_w, cx + half_w
                south2, north2 = cy - half_h, cy + half_h
                try:
                    try:
                        pois = ox.features_from_bbox(bbox=(north2, south2, east2, west2), tags=tags)
                    except TypeError:
                        pois = ox.features_from_bbox(north=north2, south=south2, east=east2, west=west2, tags=tags)
                except Exception as shrink_exc:
                    return {
                        "pois": [],
                        "success": False,
                        "error": (
                            f"bbox query failed: {bbox_exc}; "
                            f"polygon fallback failed: {poly_exc}; "
                            f"shrink-bbox retry failed: {shrink_exc}"
                        ),
                    }

        if pois.empty:
            return {"pois": [], "success": True}

        if pois.crs is None:
            pois = pois.set_crs("EPSG:4326", allow_override=True)
        elif str(pois.crs).upper() != "EPSG:4326":
            pois = pois.to_crs("EPSG:4326")

        pois = pois[pois.geometry.notnull()]
        if pois.empty:
            return {"pois": [], "success": True}
        
        poi_list = []
        for idx, row in pois.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Use representative_point to avoid centroid warnings on invalid/multipart geometries.
            pt = geom if geom.geom_type == "Point" else geom.representative_point()
            if pt is None or pt.is_empty:
                continue

            x, y = float(pt.x), float(pt.y)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            name = row.get("name", req.poi_category)
            poi_list.append({"name": str(name), "lat": y, "lon": x})
            
        return {"pois": poi_list, "success": True}
    except Exception as e:
        return {"pois": [], "success": False, "error": str(e)}