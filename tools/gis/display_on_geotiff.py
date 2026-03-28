"""DisplayOnGeotiff tool - renders vector overlays on top of a GeoTIFF."""

def display_on_geotiff(geotiff_path: str, gpkg_path: str = None, layer_name: str = None) -> dict:
    """Overlay vector data on a GeoTIFF and return the rendered image path."""
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path

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

        out_path = str(Path(geotiff_path).with_suffix(".overlay.png"))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"image_path": out_path, "success": True}
    except Exception as e:
        return {"image_path": "", "success": False, "error": str(e)}
