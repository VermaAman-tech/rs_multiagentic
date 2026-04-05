import torch
import os
from pathlib import Path

_gdino = None
_sam2  = None


def _resolve_gdino_paths() -> tuple[str, str]:
    weights_dir = Path("models/weights/groundingdino")
    cfg_path = weights_dir / "GroundingDINO_SwinT_OGC.py"
    ckpt_path = weights_dir / "pytorch_model.bin"

    # The config ships with the groundingdino package; use it if local config is absent.
    if not cfg_path.exists():
        try:
            import groundingdino  # type: ignore

            pkg_cfg = Path(groundingdino.__file__).resolve().parent / "config" / "GroundingDINO_SwinT_OGC.py"
            if pkg_cfg.exists():
                cfg_path = pkg_cfg
        except Exception:
            pass

    if not ckpt_path.exists():
        alt_ckpt = weights_dir / "groundingdino_swint_ogc.pth"
        if alt_ckpt.exists():
            ckpt_path = alt_ckpt

    missing = []
    if not cfg_path.exists():
        missing.append(str(cfg_path))
    if not ckpt_path.exists():
        missing.append(str(ckpt_path))
    if missing:
        raise FileNotFoundError(
            "Missing GroundingDINO assets: " + ", ".join(missing) + ". "
            "Run setup_eval.sh or place checkpoints under models/weights/groundingdino/."
        )

    return str(cfg_path.resolve()), str(ckpt_path.resolve())

def get_gdino():
    global _gdino
    if _gdino is None:
        cfg_path, ckpt_path = _resolve_gdino_paths()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig

        args = SLConfig.fromfile(cfg_path)
        args.device = device
        model = build_model(args)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if isinstance(checkpoint.get("model"), dict):
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint.get("state_dict"), dict):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(clean_state_dict(state_dict), strict=False)
        model.eval()
        _gdino = model
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
