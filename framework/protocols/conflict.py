from __future__ import annotations

from typing import Any


def resolve_conflict(candidate_a: dict[str, Any], candidate_b: dict[str, Any], confidence_gap: float = 0.2) -> dict[str, Any]:
    ca = float(candidate_a.get("confidence", 0.0))
    cb = float(candidate_b.get("confidence", 0.0))
    if abs(ca - cb) >= confidence_gap:
        return candidate_a if ca > cb else candidate_b

    sa = len(str(candidate_a.get("output", "")))
    sb = len(str(candidate_b.get("output", "")))
    return candidate_a if sa >= sb else candidate_b
