from pathlib import Path


def prithvi_embed(raster_path):
    if not raster_path or not Path(raster_path).exists():
        return {
            "embedding_dim": 0,
            "summary_stats": {},
            "success": False,
            "error": f"Raster not found: {raster_path}",
        }

    return {
        "embedding_dim": 0,
        "summary_stats": {},
        "success": False,
        "error": "Prithvi model is not loaded in this runtime.",
    }
