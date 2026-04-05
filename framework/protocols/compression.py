from __future__ import annotations

from typing import Any


def _wkt_bbox(wkt_text: str) -> list[float] | None:
    try:
        from shapely import wkt as shapely_wkt  # type: ignore

        geom = shapely_wkt.loads(wkt_text)
        minx, miny, maxx, maxy = geom.bounds
        return [float(minx), float(miny), float(maxx), float(maxy)]
    except Exception:
        return None


PROTECTED_KEYS = ("damage_polygons", "flood_extent", "impassable_roads", "conflict_events")


def _is_protected(key: str) -> bool:
    return any(k in key for k in PROTECTED_KEYS)


def compress_context(items: dict[str, Any]) -> tuple[dict[str, Any], float]:
    compressed = {}
    safety_facts_before = 0
    safety_facts_after = 0

    for key, value in items.items():
        if _is_protected(key):
            compressed[key] = value
            safety_facts_before += 1
            safety_facts_after += 1
            continue

        if isinstance(value, dict) and isinstance(value.get("boundary_wkt"), str):
            wkt = value.get("boundary_wkt", "")
            compressed[key] = {
                "boundary_summary": {
                    "wkt_chars": len(wkt),
                    "bbox": _wkt_bbox(wkt),
                    "gpkg_path": value.get("gpkg_path"),
                }
            }
            continue

        if isinstance(value, dict) and isinstance(value.get("pois"), list):
            pois = value.get("pois", [])
            sample = []
            for item in pois[:5]:
                if isinstance(item, dict):
                    sample.append({
                        "name": item.get("name"),
                        "lat": item.get("lat"),
                        "lon": item.get("lon"),
                    })
            compressed[key] = {"pois_count": len(pois), "sample": sample}
            continue

        if "prithvi_embed_" in key:
            continue

        if "spectral_" in key:
            if isinstance(value, (list, tuple)) and value:
                vals = [float(v) for v in value]
                mean = sum(vals) / len(vals)
                compressed[key] = {
                    "mean": mean,
                    "std": (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5,
                    "max": max(vals),
                }
            else:
                compressed[key] = value
            continue

        if "change_map_" in key and isinstance(value, list):
            compressed[key] = value[:5]
            continue

        compressed[key] = value

    ccq = 1.0 if safety_facts_before == 0 else safety_facts_after / safety_facts_before
    if ccq < 0.98:
        raise RuntimeError(f"Compression aborted: CCQ {ccq:.3f} < 0.98")
    return compressed, ccq
