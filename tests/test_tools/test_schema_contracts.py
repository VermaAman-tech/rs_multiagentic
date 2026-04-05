from tools.gis.pois_layer import _query_to_osm_tags
from tools.schemas import (
    AddPoisLayerOutput,
    ComputeDistanceOutput,
    GetAreaBoundaryInput,
    GetAreaBoundaryOutput,
)


def _field_names(model_cls):
    fields = getattr(model_cls, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    fields = getattr(model_cls, "__fields__", {})
    return set(fields.keys())


def test_get_area_boundary_schema_has_buffer_and_outputs():
    in_fields = _field_names(GetAreaBoundaryInput)
    out_fields = _field_names(GetAreaBoundaryOutput)

    assert "buffer_m" in in_fields
    assert {"boundary_wkt", "gpkg_path", "bbox", "success"}.issubset(out_fields)


def test_poi_and_distance_outputs_include_success_flag():
    poi_fields = _field_names(AddPoisLayerOutput)
    dist_fields = _field_names(ComputeDistanceOutput)

    assert "success" in poi_fields
    assert "success" in dist_fields


def test_poi_category_mapping_includes_emergency_and_utility():
    assert _query_to_osm_tags("fire station near city") == {"amenity": ["fire_station"]}
    assert _query_to_osm_tags("nearest police") == {"amenity": ["police"]}
    assert _query_to_osm_tags("gas station") == {"amenity": ["fuel"]}
