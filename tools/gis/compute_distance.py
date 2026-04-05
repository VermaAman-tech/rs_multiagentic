import math


def _normalize_point(pt):
    """
    Accept either [lat, lon] or [lon, lat].
    Heuristic: values outside latitude range imply the first value is longitude.
    """
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        raise ValueError(f"Invalid point format: {pt}")

    a = float(pt[0])
    b = float(pt[1])

    # Likely [lon, lat] ordering (common in GIS APIs).
    if abs(a) > 90 and abs(b) <= 90:
        lat = b
        lon = a
    else:
        lat = a
        lon = b

    if abs(lat) > 90 or abs(lon) > 180:
        raise ValueError(f"Invalid coordinate range after normalization: lat={lat}, lon={lon}")

    return lat, lon

def run(req):
    try:
        try:
            lat1, lon1 = _normalize_point(req.point_a)
            lat2, lon2 = _normalize_point(req.point_b)
            from geopy.distance import geodesic

            dist = geodesic((lat1, lon1), (lat2, lon2)).meters
            return {"distance_meters": float(dist), "success": True, "distance_mode": "geodesic_wgs84"}
        except Exception:
            # Fallback for projected/pixel coordinate pairs that are out of lat/lon ranges.
            a = req.point_a if isinstance(req.point_a, (list, tuple)) else [0.0, 0.0]
            b = req.point_b if isinstance(req.point_b, (list, tuple)) else [0.0, 0.0]
            if len(a) < 2 or len(b) < 2:
                raise ValueError(f"Invalid point format: point_a={req.point_a}, point_b={req.point_b}")
            x1, y1 = float(a[0]), float(a[1])
            x2, y2 = float(b[0]), float(b[1])
            dist = math.hypot(x2 - x1, y2 - y1)
            return {
                "distance_meters": float(dist),
                "success": True,
                "distance_mode": "planar_fallback",
                "warning": "Input coordinates were not valid WGS84 lat/lon; used planar Euclidean fallback.",
            }
    except Exception as e:
        return {"distance_meters": 0.0, "success": False, "error": str(e)}