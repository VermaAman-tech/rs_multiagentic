import torch
import os

_gdino = None
_sam2  = None

def get_gdino():
    global _gdino
    if _gdino is None:
        from groundingdino.util.inference import load_model as load_gdino
        _gdino = load_gdino(
            "models/weights/groundingdino/GroundingDINO_SwinT_OGC.py",
            "models/weights/groundingdino/pytorch_model.bin",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _gdino

def get_sam2():
    global _sam2
    if _sam2 is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2("sam2.1_hiera_large.yaml",
                           "models/weights/sam2/sam2.1_hiera_large.pt",
                           device="cuda" if torch.cuda.is_available() else "cpu")
        _sam2 = SAM2ImagePredictor(model)
    return _sam2
