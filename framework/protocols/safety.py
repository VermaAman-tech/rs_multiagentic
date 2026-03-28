from __future__ import annotations

from typing import Any


REQUIRED_KEYS = ["damage_polygons", "flood_extent", "impassable_roads", "conflict_events"]


def validate_safety_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    if payload.get("enforce_safety_keys"):
        missing = [k for k in REQUIRED_KEYS if k not in payload]
        return (len(missing) == 0, missing)
    return (True, [])
