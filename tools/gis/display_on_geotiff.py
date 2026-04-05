"""DisplayOnGeotiff tool - renders vector overlays on top of a GeoTIFF."""

from pathlib import Path


def display_on_geotiff(
    geotiff_path: str,
    gpkg_path: str = None,
    layer_name: str = None,
    features: list | None = None,
    output_path: str | None = None,
) -> dict:
    """Overlay vector data on a GeoTIFF and return the rendered image path."""
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not geotiff_path or not Path(geotiff_path).exists():
            raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

        with rasterio.open(geotiff_path) as src:
            data = src.read([1, 2, 3]) if src.count >= 3 else src.read(1)
            bounds = src.bounds

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if len(data.shape) == 3:
            ax.imshow(data.transpose(1, 2, 0), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        else:
            ax.imshow(data, cmap="gray", extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

        if gpkg_path and layer_name:
            import geopandas as gpd
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
            gdf.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2)

        if isinstance(features, list) and features:
            xs = []
            ys = []
            for feat in features:
                if not isinstance(feat, dict):
                    continue
                if isinstance(feat.get("lon"), (int, float)) and isinstance(feat.get("lat"), (int, float)):
                    xs.append(float(feat["lon"]))
                    ys.append(float(feat["lat"]))
            if xs and ys:
                ax.scatter(xs, ys, c="red", s=20)

        out_path = output_path or str(Path(geotiff_path).with_suffix(".overlay.png"))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"raster_path": out_path, "success": True}
    except Exception as e:
        return {"raster_path": "", "success": False, "error": str(e)}


def run(req):
    return display_on_geotiff(
        geotiff_path=req.geotiff_path,
        features=getattr(req, "features", None),
        output_path=getattr(req, "output_path", None),
    )
