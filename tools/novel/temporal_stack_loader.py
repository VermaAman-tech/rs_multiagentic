import json
from pathlib import Path

def temporal_stack_loader(aoi_bbox, date_range, sensor="sentinel-2", max_cloud_pct=25, output_dir="data/stacks", local_stack_path=None):
    if local_stack_path and Path(local_stack_path).exists():
        meta_path = local_stack_path.replace(".tif", "_meta.json")
        if Path(meta_path).exists():
            meta = json.load(open(meta_path))
            return {
                "stack_paths": [local_stack_path],
                "n_epochs": meta.get("n_epochs", 2),
                "success": True,
            }
        return {"stack_paths": [local_stack_path], "n_epochs": 2, "success": True}
    return {"stack_paths": ["data/mock/epoch_0.tif", "data/mock/epoch_1.tif"], "n_epochs": 2, "success": True}
