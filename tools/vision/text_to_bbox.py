import torch
from torchvision.ops import box_convert, nms

from tools.vision._models import get_gdino


def run(req):
    try:
        from groundingdino.util.inference import load_image, predict

        image_source, image_tensor = load_image(req.image_path)
        model = get_gdino()

        text_prompt = getattr(req, "text_prompt", None)
        if not text_prompt:
            text_prompt = getattr(req, "text", "object")
        prompt_l = str(text_prompt).lower()

        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=getattr(req, "confidence_threshold", 0.3),
            text_threshold=0.25,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        if len(boxes) > 0:
            h, w, _ = image_source.shape
            scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
            xyxy = box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu()
            scores = logits.detach().cpu() if hasattr(logits, "detach") else torch.tensor(logits, dtype=torch.float32)

            xyxy[:, 0] = xyxy[:, 0].clamp(0, w)
            xyxy[:, 2] = xyxy[:, 2].clamp(0, w)
            xyxy[:, 1] = xyxy[:, 1].clamp(0, h)
            xyxy[:, 3] = xyxy[:, 3].clamp(0, h)

            bw = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0)
            bh = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
            area_ratio = (bw * bh) / max(float(w * h), 1.0)
            aspect = torch.maximum(bw / torch.clamp(bh, min=1.0), bh / torch.clamp(bw, min=1.0))
            valid = (bw >= 6.0) & (bh >= 6.0) & (area_ratio >= 3e-4) & (area_ratio <= 0.08) & (aspect <= 4.0)

            idx = torch.where(valid)[0]
            if idx.numel() == 0:
                valid = (bw >= 2.0) & (bh >= 2.0)
                idx = torch.where(valid)[0]

            if idx.numel() == 0:
                return {"bboxes": [], "success": True}

            nms_iou = float(getattr(req, "nms_iou_threshold", 0.5) or 0.5)
            keep = nms(xyxy[idx], scores[idx], iou_threshold=nms_iou)
            ranked_idx = idx[keep]

            boxes_xyxy = xyxy.tolist()
            scores_list = scores.tolist()

            top1 = bool(getattr(req, "top1", True))
            if top1:
                best_idx = max(ranked_idx.tolist(), key=lambda i: float(scores_list[i]))
                return {"bboxes": [boxes_xyxy[best_idx]], "success": True}

            default_max = 8
            max_boxes = int(getattr(req, "max_boxes", default_max) or default_max)
            ranked = ranked_idx[:max_boxes].tolist()
            return {
                "bboxes": [boxes_xyxy[i] for i in ranked],
                "scores": [scores_list[i] for i in ranked],
                "phrases": [phrases[i] for i in ranked],
                "success": True,
            }

        return {"bboxes": [], "success": True}
    except Exception as e:
        return {"bboxes": [], "success": False, "error": str(e)}