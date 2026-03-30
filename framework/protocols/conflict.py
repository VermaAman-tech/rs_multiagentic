from __future__ import annotations

from typing import Any


def resolve_conflict(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    confidence_gap: float = 0.2,
    high_confidence: float = 0.85,
) -> dict[str, Any]:
    ca = float(candidate_a.get("confidence", 0.0))
    cb = float(candidate_b.get("confidence", 0.0))

    winner: dict[str, Any]
    strategy: str

    # Tier 1: confidence comparator.
    if abs(ca - cb) >= confidence_gap:
        winner = candidate_a if ca > cb else candidate_b
        strategy = "tier1_confidence_gap"
    elif max(ca, cb) >= high_confidence:
        winner = candidate_a if ca >= cb else candidate_b
        strategy = "tier1_high_confidence"
    else:
        # Tier 2 fallback: choose the richer structured output as proxy judge.
        sa = len(str(candidate_a.get("output", "")))
        sb = len(str(candidate_b.get("output", "")))
        winner = candidate_a if sa >= sb else candidate_b
        strategy = "tier2_plausibility"

    out = dict(winner)
    out.setdefault("conflict_resolution", {})
    out["conflict_resolution"].update(
        {
            "strategy": strategy,
            "confidence_a": ca,
            "confidence_b": cb,
            "confidence_gap": abs(ca - cb),
        }
    )
    return out
