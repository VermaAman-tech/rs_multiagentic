from __future__ import annotations

from typing import Any


REQUIRED_KEYS = ["damage_polygons", "flood_extent", "impassable_roads", "conflict_events"]


def _contains_key_recursive(value: Any, target_key: str) -> bool:
    if isinstance(value, dict):
        if target_key in value and value.get(target_key) not in (None, "", [], {}):
            return True
        return any(_contains_key_recursive(v, target_key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key_recursive(v, target_key) for v in value)
    return False


def _requires_proximity_evidence(objective: str, query_type: str) -> bool:
    if query_type == "geospatial_proximity":
        return True

    if any(k in objective for k in ("closest", "nearest", "proximity")):
        return True

    if "distance" not in objective:
        return False

    # Exclude common non-geospatial phrases that should not require nearest-pair evidence.
    exclusions = (
        "ground sampling distance",
        "gsd",
        "m/pixel",
        "per pixel",
        "pixel distance",
        "nearest whole number",
        "nearest integer",
        "round to the nearest",
        "rounded to the nearest",
    )
    if any(k in objective for k in exclusions):
        return False

    return True


def validate_safety_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    if payload.get("enforce_safety_keys"):
        missing = [k for k in REQUIRED_KEYS if k not in payload]
        return (len(missing) == 0, missing)

    objective = str(payload.get("objective", "")).lower()
    query_type = str(payload.get("query_type", "")).lower()

    required_groups: list[tuple[str, list[str]]] = []

    if query_type in {"disaster_assessment", "change_detection"} or any(
        k in objective for k in ("flood", "damage", "disaster", "change")
    ):
        required_groups.append(("disaster_evidence", ["damage_polygons", "flood_extent", "change_map_path", "conflict_events"]))

    if _requires_proximity_evidence(objective, query_type):
        required_groups.append(("distance_evidence", ["distance_meters", "closest_pair", "segment_scores"]))

    if any(k in objective for k in ("route", "evacuation", "supply", "safe path")):
        required_groups.append(("route_evidence", ["routes", "impassable_roads", "segment_scores"]))

    missing: list[str] = []
    for group_name, alternatives in required_groups:
        if not any(_contains_key_recursive(payload, k) for k in alternatives):
            missing.append(group_name)

    return (len(missing) == 0, missing)
