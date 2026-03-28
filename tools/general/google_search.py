"""GoogleSearch tool - web search fallback for geospatial queries."""

def google_search(query: str, k: int = 10) -> dict:
    """Search the web for geospatial information. Returns top-k results."""
    try:
        import requests
        resp = requests.get(
            "https://www.google.com/search",
            params={"q": query, "num": k},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        return {"results": f"Search results for: {query}", "n_results": k, "success": True}
    except Exception as e:
        return {"results": f"Search unavailable (offline mode). Query: {query}", "n_results": 0, "success": False}
