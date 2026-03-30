from __future__ import annotations


def google_search(query: str, k: int = 10) -> dict:
    """Return a lightweight structured web-search placeholder in offline-safe mode."""
    q = query.strip()
    if not q:
        return {"results": [], "n_results": 0, "success": False}
    return {
        "results": [f"Search result {i + 1} for: {q}" for i in range(max(1, k))],
        "n_results": max(1, k),
        "success": True,
    }
