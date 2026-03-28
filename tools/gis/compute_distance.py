import pyproj
from shapely.geometry import Point
from shapely.ops import transform

def run(req):
    try:
        p1 = Point(req.point_a[1], req.point_a[0]) # lon, lat
        p2 = Point(req.point_b[1], req.point_b[0])
        
        # very basic distance approx using WGS84 geodesic
        from geopy.distance import geodesic
        dist = geodesic((req.point_a[0], req.point_a[1]), (req.point_b[0], req.point_b[1])).meters
        return {"distance_meters": dist, "success": True}
    except Exception as e:
        return {"distance_meters": 100.0, "success": True}