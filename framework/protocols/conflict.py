from __future__ import annotations

from typing import Any

# Evidence-rich keys that signal a candidate actually produced usable output.
# Presence of ANY of these in the output strongly suggests real work was done.
_EVIDENCE_KEYS = (
    "distance_meters",
    "closest_pair",
    "segment_scores",
    "damage_polygons",
    "flood_extent",
    "impassable_roads",
    "change_map_path",
    "routes",
    "pois",
    "boundary",
    "bbox",
    "index_statistics",
    "distance_evidence",
    "distance_observations",
)

# Phrases that indicate a "failure/insufficient" style answer.
_INSUFFICIENT_PHRASES = (
    "insufficient",
    "unable to",
    "could not",
    "cannot determine",
    "no data",
    "not available",
    "no accessible",
    "no trustworthy",
    "mock result",
)


def _score_evidence(candidate: dict[str, Any]) -> float:
    """Score a candidate by the richness of its output, independent of confidence.

    Higher score = more concrete, usable evidence.
    """
    output = candidate.get("output", {})
    if not isinstance(output, dict):
        return 0.0

    score = 0.0

    # Strongly reward presence of evidence keys with non-empty values.
    for key in _EVIDENCE_KEYS:
        val = output.get(key)
        if val is None or val == "" or val == [] or val == {}:
            continue
        score += 2.0

    # Penalize "insufficient"/"unable to" answers.
    ans = str(output.get("ans", "") or "").lower()
    thought = str(output.get("thought", "") or "").lower()
    combined = ans + " " + thought
    for phrase in _INSUFFICIENT_PHRASES:
        if phrase in combined:
            score -= 3.0
            break

    # Reward non-trivial answer length (concrete answers tend to be longer).
    if ans and not any(p in ans for p in _INSUFFICIENT_PHRASES):
        score += min(len(ans) / 100.0, 2.0)

    # Reward actual tool usage (more calls => more grounded).
    tool_calls = candidate.get("tool_calls", [])
    if isinstance(tool_calls, list):
        score += min(len(tool_calls) * 0.3, 1.5)

    return score


def resolve_conflict(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    confidence_gap: float = 0.2,
    high_confidence: float = 0.85,
) -> dict[str, Any]:
    ca = float(candidate_a.get("confidence", 0.0))
    cb = float(candidate_b.get("confidence", 0.0))

    # Score evidence richness independent of reported confidence.
    sa_ev = _score_evidence(candidate_a)
    sb_ev = _score_evidence(candidate_b)

    winner: dict[str, Any]
    strategy: str

    # Tier 0: if one candidate has evidence and the other does not, evidence wins.
    # This prevents a high-confidence "insufficient" answer from beating a
    # lower-confidence answer that has real distance_meters/closest_pair/etc.
    if sa_ev > 0 and sb_ev <= 0:
        winner = candidate_a
        strategy = "tier0_evidence_presence"
    elif sb_ev > 0 and sa_ev <= 0:
        winner = candidate_b
        strategy = "tier0_evidence_presence"
    # Tier 0b: large evidence gap beats confidence gap.
    elif abs(sa_ev - sb_ev) >= 3.0:
        winner = candidate_a if sa_ev > sb_ev else candidate_b
        strategy = "tier0_evidence_gap"
    # Tier 1: confidence comparator (only when evidence is comparable).
    elif abs(ca - cb) >= confidence_gap:
        winner = candidate_a if ca > cb else candidate_b
        strategy = "tier1_confidence_gap"
    elif max(ca, cb) >= high_confidence:
        winner = candidate_a if ca >= cb else candidate_b
        strategy = "tier1_high_confidence"
    else:
        # Tier 2 fallback: combine evidence score and confidence.
        combined_a = sa_ev + ca
        combined_b = sb_ev + cb
        winner = candidate_a if combined_a >= combined_b else candidate_b
        strategy = "tier2_plausibility"

    out = dict(winner)
    out.setdefault("conflict_resolution", {})
    out["conflict_resolution"].update(
        {
            "strategy": strategy,
            "confidence_a": ca,
            "confidence_b": cb,
            "confidence_gap": abs(ca - cb),
            "evidence_score_a": sa_ev,
            "evidence_score_b": sb_ev,
        }
    )
    return out
