from __future__ import annotations

import re
from typing import Any


_OBSERVATION_HINT_RE = re.compile(
    r"observation|saved\s+to\s+line\s+layer|distances\s*\(in\s*meters\)|distance\s*=",
    re.IGNORECASE,
)

_PAIR_RE = re.compile(
    r"(?P<entity_a>[^,\n:]{1,180}?)\s*,\s*"
    r"(?P<entity_b>[^,\n:]{1,180}?)\s*,\s*"
    r"distance\s*=\s*(?P<distance>-?\d+(?:\.\d+)?)\s*m"
    r"(?:\s*,\s*travel_time\s*=\s*(?P<travel_time>-?\d+(?:\.\d+)?)\s*s)?",
    re.IGNORECASE,
)

_PLACEHOLDER_RE = re.compile(r"candidate|point[_\s]?[ab]|placeholder", re.IGNORECASE)


def is_placeholder_entity_name(name: Any) -> bool:
    text = str(name or "").strip()
    if not text:
        return True
    return bool(_PLACEHOLDER_RE.search(text))


def _clean_entity_name(value: str) -> str:
    text = " ".join(str(value or "").split()).strip(" '\"\t")
    if not text:
        return ""
    lowered = text.lower()
    if "saved to line layer" in lowered or lowered.startswith("distances (in meters)"):
        return ""
    return text


def _candidate_texts(payload: dict[str, Any], raw_output: str | None = None) -> list[str]:
    texts: list[str] = []

    objective = payload.get("objective")
    if isinstance(objective, str) and _OBSERVATION_HINT_RE.search(objective):
        texts.append(objective)

    human_inputs = payload.get("human_inputs")
    if isinstance(human_inputs, list):
        for item in human_inputs:
            if isinstance(item, str) and _OBSERVATION_HINT_RE.search(item):
                texts.append(item)

    if isinstance(raw_output, str) and _OBSERVATION_HINT_RE.search(raw_output):
        texts.append(raw_output)

    return texts


def _extract_pairs_from_text(text: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()

    for match in _PAIR_RE.finditer(text):
        a = _clean_entity_name(match.group("entity_a"))
        b = _clean_entity_name(match.group("entity_b"))
        if not a or not b:
            continue

        try:
            distance_m = float(match.group("distance"))
        except (TypeError, ValueError):
            continue

        if distance_m < 0:
            continue

        travel_s: float | None = None
        travel_raw = match.group("travel_time")
        if travel_raw not in (None, ""):
            try:
                travel_s = float(travel_raw)
            except (TypeError, ValueError):
                travel_s = None

        dedupe_key = (a.lower(), b.lower(), int(round(distance_m * 1000.0)))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        rec: dict[str, Any] = {
            "entity_a": a,
            "entity_b": b,
            "distance_meters": distance_m,
        }
        if travel_s is not None:
            rec["travel_time_seconds"] = travel_s
        pairs.append(rec)

    return pairs


def extract_observation_distance_evidence(
    payload: dict[str, Any],
    raw_output: str | None = None,
) -> dict[str, Any] | None:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()

    for text in _candidate_texts(payload, raw_output=raw_output):
        for rec in _extract_pairs_from_text(text):
            key = (
                str(rec.get("entity_a", "")).lower(),
                str(rec.get("entity_b", "")).lower(),
                int(round(float(rec.get("distance_meters", 0.0)) * 1000.0)),
            )
            if key in seen:
                continue
            seen.add(key)
            pairs.append(rec)

    if not pairs:
        return None

    closest = min(pairs, key=lambda x: float(x.get("distance_meters", float("inf"))))
    objective = str(payload.get("objective", ""))
    summary_mode = "observation" in objective.lower() and "summarize" in objective.lower()

    return {
        "pairs": pairs,
        "closest_pair": closest,
        "summary_mode": summary_mode,
    }
