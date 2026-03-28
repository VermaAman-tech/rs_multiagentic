"""
OEA Full Inference Evaluator — Vision-Language Model (e.g. Qwen2.5-VL)
Computes: Inst, Tool, ArgN, ArgV per tool-call step.

This script supports both text-only and vision-language models:
  - For VL models: images are included inline in the prompt
  - For text-only models: image-requiring samples are skipped

Uses conversation-prefix prompting: for each tool-call step in the GT
conversation, we feed all prior turns as context and ask the model to
predict the NEXT tool call.
"""

# ── Triton patch must come before any vllm import ────────────────────────────
import types, sys

def _patch_triton():
    try:
        import triton.backends.nvidia.driver as _drv
        class _MockCudaUtils:
            _inst = None
            def __new__(cls):
                if cls._inst is None:
                    cls._inst = super().__new__(cls)
                return cls._inst
            def get_device_properties(self, *a, **kw):
                return {"max_shared_mem": 0}
            def load_binary(self, *a, **kw):
                raise NotImplementedError
            def cuOccupancyMaxActiveClusters(self, *a, **kw): return 0
            def set_printf_fifo_size(self, *a, **kw): pass
            def fill_1d_tma_descriptor(self, *a, **kw): pass
            def fill_2d_tma_descriptor(self, *a, **kw): pass
        _drv.CudaUtils = _MockCudaUtils
        _drv.CudaUtils._inst = _MockCudaUtils()
        print("[triton_patch] Triton CudaUtils patched.")
    except Exception as e:
        print(f"[triton_patch] patch skipped: {e}")

_patch_triton()
# ─────────────────────────────────────────────────────────────────────────────

import json
import argparse
import os
import re
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a geospatial reasoning agent. You have access to these tools:\n"
    "GetAreaBoundary, AddPoisLayer, ComputeDistance, DisplayOnMap, "
    "GetBboxFromGeotiff, AddIndexLayer, ComputeIndexChange, ShowIndexLayer, "
    "ObjectDetection, TextToBbox, CountGivenObject, ImageDescription, "
    "RegionAttributeDescription, OCR, ChangeDetection, DrawBox, AddText, "
    "SegmentObjectPixels, GoogleSearch, Plot, DisplayOnGeotiff, "
    "Calculator, Solver, Terminate\n\n"
    "Given the conversation history, determine the NEXT tool call.\n"
    'Respond ONLY with a JSON object: {"name": "<ToolName>", "arguments": {<key>: <value>, ...}}\n'
    "No explanation or extra text."
)


def extract_tool_call(text: str) -> dict | None:
    """Extract JSON tool call from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try direct JSON
    try:
        obj = json.loads(text)
        if "actions" in obj and isinstance(obj["actions"], list) and obj["actions"]:
            return obj["actions"][0]
        if obj.get("name"):
            return obj
    except Exception:
        pass

    # Try extracting JSON block
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if "actions" in obj and isinstance(obj["actions"], list) and obj["actions"]:
                return obj["actions"][0]
            if obj.get("name"):
                return obj
        except Exception:
            continue

    return None


def build_eval_instances(oea_data: list, img_dir: str = "data/openearth_agent",
                         limit: int = 0, use_images: bool = False) -> list[dict]:
    """
    Build evaluation instances from OEA multi-turn conversations.

    For each tool-call step, create an instance with conversation prefix as context.
    If use_images=True, include image references for VL models.
    If use_images=False, skip image-requiring samples.
    """
    instances = []

    for sample in oea_data:
        has_images = bool(sample.get("images"))

        # For text-only mode, skip samples that need images
        if not use_images and has_images:
            continue

        conv = sample.get("conversation", [])
        if not conv:
            continue

        # Resolve image paths for VL mode
        image_paths = []
        if use_images and has_images:
            for img_name in sample.get("images", [])[:1]:
                full_path = os.path.join(img_dir, img_name)
                if os.path.exists(full_path):
                    image_paths.append(os.path.abspath(full_path))

        # Walk conversation and create one eval instance per tool-call step
        prefix_messages = []
        for turn_idx, turn in enumerate(conv):
            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "gpt":
                try:
                    parsed = json.loads(value)
                    actions = parsed.get("actions", [])
                    if actions and actions[0].get("name"):
                        gt_action = actions[0]
                        gt_name = gt_action.get("name", "").strip()
                        gt_args = gt_action.get("arguments", {})

                        # Build chat messages from prefix
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

                        for pt in prefix_messages:
                            pt_role = pt.get("from", "")
                            pt_value = pt.get("value", "")

                            if pt_role in ("human", "user"):
                                clean = pt_value.replace("<AGENT_PROMPT>", "").strip()
                                if clean.startswith("Question:"):
                                    clean = clean[len("Question:"):].strip()

                                if use_images and image_paths and len(messages) == 1:
                                    # First user message: include image for VL
                                    user_content = []
                                    for ip in image_paths:
                                        user_content.append({"type": "image", "image": f"file://{ip}"})
                                    user_content.append({"type": "text", "text": clean})
                                    messages.append({"role": "user", "content": user_content})
                                else:
                                    messages.append({"role": "user", "content": clean})
                            elif pt_role == "gpt":
                                messages.append({"role": "assistant", "content": pt_value})

                        instances.append({
                            "sample_idx": sample.get("idx", -1),
                            "turn_idx": turn_idx,
                            "messages": messages,
                            "gt_name": gt_name,
                            "gt_args": gt_args,
                        })
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

            prefix_messages.append(turn)

        if limit > 0 and len(instances) >= limit:
            instances = instances[:limit]
            break

    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--oea-data", default="data/openearth_agent/test.json")
    parser.add_argument("--img-dir", default="data/openearth_agent")
    parser.add_argument("--limit", type=int, default=0, help="0 = all samples")
    parser.add_argument("--use-images", action="store_true", default=False,
                        help="Include images for VL models")
    parser.add_argument("--output", default="results/oea_inference_results.json")
    args = parser.parse_args()

    # Auto-detect if model is VL
    is_vl = "VL" in args.model.upper() or "vl" in args.model.lower()
    use_images = args.use_images or is_vl

    with open(args.oea_data) as f:
        oea_data = json.load(f)

    instances = build_eval_instances(
        oea_data, img_dir=args.img_dir, limit=args.limit, use_images=use_images
    )
    print(f"Built {len(instances)} evaluation instances (VL={use_images})")

    if not instances:
        print("ERROR: No evaluation instances built.")
        return

    print(f"Initializing vLLM engine: {args.model}")
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    if use_images:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 1}

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    all_messages = [inst["messages"] for inst in instances]

    print("Running batched inference...")
    outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)

    # ── Compute metrics ──────────────────────────────────────────────────────
    inst_correct = 0
    tool_correct = 0
    argn_correct = 0
    argn_total = 0
    argv_correct = 0
    argv_total = 0
    per_sample_results = []

    for out_idx, out in enumerate(outputs):
        generated_text = out.outputs[0].text
        pred = extract_tool_call(generated_text)
        inst = instances[out_idx]

        gt_name = inst["gt_name"]
        gt_args = inst["gt_args"]

        pred_name = (pred.get("name") or "").strip() if pred else ""
        pred_args = (pred.get("arguments") or pred.get("parameters") or {}) if pred else {}

        tool_match = gt_name.lower().strip() == pred_name.lower().strip()
        if tool_match:
            tool_correct += 1

        is_inst_correct = tool_match
        for k, v in gt_args.items():
            argn_total += 1
            argv_total += 1
            if k in pred_args:
                argn_correct += 1
                if str(pred_args[k]).lower().strip() == str(v).lower().strip():
                    argv_correct += 1
                else:
                    is_inst_correct = False
            else:
                is_inst_correct = False

        if is_inst_correct:
            inst_correct += 1

        per_sample_results.append({
            "sample_idx": inst["sample_idx"],
            "turn_idx": inst["turn_idx"],
            "gt_tool": gt_name,
            "pred_tool": pred_name,
            "tool_correct": tool_match,
            "inst_correct": is_inst_correct,
            "raw_output": generated_text[:300],
        })

    total = len(instances)
    metrics = {
        "model": args.model,
        "dataset": "OpenEarthAgent",
        "total_instances": total,
        "Inst": round((inst_correct / total) * 100, 2) if total > 0 else 0.0,
        "Tool": round((tool_correct / total) * 100, 2) if total > 0 else 0.0,
        "ArgN": round((argn_correct / argn_total) * 100, 2) if argn_total > 0 else 0.0,
        "ArgV": round((argv_correct / argv_total) * 100, 2) if argv_total > 0 else 0.0,
        "per_sample": per_sample_results,
    }

    print(f"\n--- OEA Full Inference Metrics ({args.model}) ---")
    summary = {k: v for k, v in metrics.items() if k != "per_sample"}
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
