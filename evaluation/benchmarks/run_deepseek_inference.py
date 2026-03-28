"""
E2 OpenEarthAgent Evaluation — vLLM offline inference
Metrics: Inst (full match), Tool (name match), ArgN (argument names), ArgV (argument values).

Key design:
  - Each OEA sample is a multi-turn conversation (human→gpt→human→gpt→...).
  - We create ONE evaluation instance per tool-call step: feed the conversation
    prefix (everything *before* that GPT turn) as context, and ask the model
    to predict the NEXT tool call.
  - The system prompt includes the full OEA tool catalogue so the model knows
    which tools are available.
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


# ── Full OEA tool catalogue ─────────────────────────────────────────────────
OEA_TOOLS = [
    {"name": "GetAreaBoundary",
     "description": "Retrieve the geographic boundary polygon for a named area. Optionally apply a buffer in meters.",
     "parameters": {"area": "str — Name of the area", "buffer_m": "int — optional buffer in meters"}},
    {"name": "AddPoisLayer",
     "description": "Query and add a layer of Points of Interest (POIs) to a GeoPackage.",
     "parameters": {"gpkg": "str — GeoPackage ID", "query": "dict — OSM tag query e.g. {\"amenity\": \"fire_station\"}", "layer_name": "str — name for the layer"}},
    {"name": "ComputeDistance",
     "description": "Compute pairwise distances between features in two layers of a GeoPackage.",
     "parameters": {"gpkg": "str — GeoPackage ID", "src_layer": "str — source layer", "tar_layer": "str — target layer", "top": "int — return top-K nearest"}},
    {"name": "DisplayOnMap",
     "description": "Display GeoPackage layers on an interactive map.",
     "parameters": {"gpkg": "str — GeoPackage ID"}},
    {"name": "GetBboxFromGeotiff",
     "description": "Extract bounding box and CRS from a GeoTIFF file.",
     "parameters": {"geotiff_path": "str — path to GeoTIFF"}},
    {"name": "AddIndexLayer",
     "description": "Compute a spectral index (NDVI, NDBI, NBR, etc.) raster layer from satellite imagery.",
     "parameters": {"gpkg": "str — GeoPackage ID", "index_type": "str — index name e.g. NDVI", "layer_name": "str — output layer name", "year": "int — year of imagery"}},
    {"name": "ComputeIndexChange",
     "description": "Compute the difference between two spectral index layers to detect change.",
     "parameters": {"gpkg": "str", "index_type": "str", "layer1_name": "str", "layer2_name": "str", "diff_layer_name": "str"}},
    {"name": "ShowIndexLayer",
     "description": "Visualize a spectral index raster layer on a map.",
     "parameters": {"gpkg": "str", "layer_name": "str"}},
    {"name": "ObjectDetection",
     "description": "Detect common objects in an image, returning bounding boxes and labels.",
     "parameters": {"image": "str — image path"}},
    {"name": "TextToBbox",
     "description": "Detect objects matching a text description, returning bounding boxes.",
     "parameters": {"image": "str — image ID", "text": "str — object description", "top1": "bool — return only top result"}},
    {"name": "CountGivenObject",
     "description": "Count the number of specified objects in an image.",
     "parameters": {"image": "str — image ID", "text": "str — object description", "bbox": "str — optional region"}},
    {"name": "ImageDescription",
     "description": "Generate a natural-language description of an image.",
     "parameters": {"image": "str — image ID"}},
    {"name": "RegionAttributeDescription",
     "description": "Describe a specific attribute of a region in an image.",
     "parameters": {"image": "str", "bbox": "str — bounding box", "attribute": "str — what to describe"}},
    {"name": "OCR",
     "description": "Recognize text in an image.",
     "parameters": {"image": "str — image ID"}},
    {"name": "ChangeDetection",
     "description": "Analyze pre/post event images to detect and describe changes.",
     "parameters": {"pre_image": "str", "post_image": "str", "text": "str — task description"}},
    {"name": "DrawBox",
     "description": "Draw a bounding box annotation on an image.",
     "parameters": {"image": "str", "bbox": "str — (x1,y1,x2,y2)", "annotation": "str — label text"}},
    {"name": "AddText",
     "description": "Add text annotation to an image.",
     "parameters": {"image": "str", "text": "str", "position": "str"}},
    {"name": "SegmentObjectPixels",
     "description": "Segment specified objects in an image and return pixel counts.",
     "parameters": {"image": "str", "text": "str — object description"}},
    {"name": "GoogleSearch",
     "description": "Search the web for information.",
     "parameters": {"query": "str — search query"}},
    {"name": "Plot",
     "description": "Execute Python code to generate a matplotlib plot.",
     "parameters": {"command": "str — Python code"}},
    {"name": "DisplayOnGeotiff",
     "description": "Overlay vector features on a GeoTIFF raster.",
     "parameters": {"geotiff_path": "str", "gpkg_path": "str", "layer_name": "str"}},
    {"name": "Calculator",
     "description": "Evaluate a mathematical expression. Supports math module functions.",
     "parameters": {"expression": "str — Python math expression"}},
    {"name": "Solver",
     "description": "Solve mathematical equations using sympy.",
     "parameters": {"command": "str — Python code with sympy"}},
    {"name": "Terminate",
     "description": "End the task and provide a final answer.",
     "parameters": {"ans": "str — final answer text"}},
]

TOOL_NAMES = [t["name"] for t in OEA_TOOLS]

def _format_tool_descriptions() -> str:
    """Format tool catalogue for the system prompt."""
    lines = []
    for t in OEA_TOOLS:
        params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items())
        lines.append(f"  - {t['name']}: {t['description']}  Parameters: {params}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a geospatial reasoning agent with access to the following tools:\n\n"
    + _format_tool_descriptions() + "\n\n"
    "Given the conversation history (prior user requests, your previous tool calls, "
    "and observation results), determine the NEXT tool call to make.\n\n"
    "YOU MUST respond with ONLY a raw JSON object, nothing else. No markdown, no explanation, "
    "no code fences, no preamble. Just the JSON.\n\n"
    "FORMAT (copy exactly):\n"
    '{"name": "<ToolName>", "arguments": {"param1": "value1", "param2": "value2"}}\n\n'
    "EXAMPLE:\n"
    '{"name": "GetAreaBoundary", "arguments": {"area": "Banff National Park, Alberta, Canada", "buffer_m": 3000}}\n\n'
    "Rules:\n"
    "- Tool name MUST be from the list above exactly (case-sensitive).\n"
    "- Do NOT wrap in ```json```. Do NOT add 'Thought:' prefix.\n"
    "- If the task is complete, use: {\"name\": \"Terminate\", \"arguments\": {\"ans\": \"<final answer>\"}}\n"
)


def extract_tool_call(text: str) -> dict | None:
    """
    Parse the model's output into a tool-call dict.
    Handles:
      - Plain JSON: {"name": ..., "arguments": {...}}
      - Nested "actions" list: {"actions": [{"name": ..., "arguments": {...}}]}
      - <tool_call>...</tool_call> tags
      - ```json blocks
      - Tool call with leading 'Thought:' text
    """
    # Strip <think> tags, markdown think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json|python)?\n?", "", text).replace("```", "").strip()

    # Try <tool_call> tags first
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try direct parse first (model output is clean JSON)
    try:
        obj = json.loads(text)
        if "actions" in obj and isinstance(obj["actions"], list) and obj["actions"]:
            inner = obj["actions"][0]
            if inner.get("name"):
                return inner
        if obj.get("name"):
            return obj
    except Exception:
        pass

    # Try to extract the outermost JSON object (handles Thought: prefix etc.)
    # First pass: try greedy match for well-nested JSON
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate)
                    if "actions" in obj and isinstance(obj["actions"], list) and obj["actions"]:
                        inner = obj["actions"][0]
                        if inner.get("name"):
                            return inner
                    if obj.get("name"):
                        return obj
                except Exception:
                    pass
                start = -1

    return None


def build_eval_instances(oea_data: list, limit: int = 0, text_only: bool = True) -> list[dict]:
    """
    Build evaluation instances from OEA multi-turn conversations.

    For each sample, we find every GPT turn that contains a non-empty tool call.
    For each such turn, we create an instance where:
      - The prompt is the conversation prefix (system + all turns before this GPT turn)
      - The GT is the tool call from this GPT turn

    Args:
        oea_data: List of OEA test samples
        limit: Max number of instances (0 = all)
        text_only: If True, skip samples that require images
    """
    instances = []

    for sample in oea_data:
        # Optionally skip image-requiring samples for text-only models
        if text_only and sample.get("images"):
            continue

        conv = sample.get("conversation", [])
        if not conv:
            continue

        # Walk through conversation turns and create one eval instance per tool-call step
        prefix_turns = []
        for turn_idx, turn in enumerate(conv):
            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "gpt":
                # Try to extract tool call from this GPT turn
                try:
                    parsed = json.loads(value)
                    actions = parsed.get("actions", [])
                    if actions and actions[0].get("name"):
                        gt_action = actions[0]
                        gt_name = gt_action.get("name", "").strip()
                        gt_args = gt_action.get("arguments", {})

                        # Build the prompt from the conversation prefix
                        prompt_parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
                        for pt in prefix_turns:
                            pt_role = pt.get("from", "")
                            pt_value = pt.get("value", "")
                            if pt_role in ("human", "user"):
                                # Strip <AGENT_PROMPT> header from the first human turn
                                clean = pt_value.replace("<AGENT_PROMPT>", "").strip()
                                # Remove duplicate "Question:" prefix if present
                                if clean.startswith("Question:"):
                                    clean = clean[len("Question:"):].strip()
                                prompt_parts.append(f"<|im_start|>user\n{clean}<|im_end|>\n")
                            elif pt_role == "gpt":
                                prompt_parts.append(f"<|im_start|>assistant\n{pt_value}<|im_end|>\n")

                        prompt_parts.append("<|im_start|>assistant\n")
                        prompt = "".join(prompt_parts)

                        instances.append({
                            "sample_idx": sample.get("idx", -1),
                            "turn_idx": turn_idx,
                            "prompt": prompt,
                            "gt_name": gt_name,
                            "gt_args": gt_args,
                        })
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

            # Add this turn to the prefix for subsequent eval instances
            prefix_turns.append(turn)

        if limit > 0 and len(instances) >= limit:
            instances = instances[:limit]
            break

    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--oea-data", default="data/openearth_agent/test.json")
    parser.add_argument("--limit", type=int, default=0, help="0 = all instances")
    parser.add_argument("--text-only", action="store_true", default=True,
                        help="Skip samples requiring images (for text-only models)")
    parser.add_argument("--output", default="results/e2_oea_real.json")
    args = parser.parse_args()

    with open(args.oea_data) as f:
        oea_data = json.load(f)

    instances = build_eval_instances(oea_data, limit=args.limit, text_only=args.text_only)
    print(f"Built {len(instances)} evaluation instances from {len(oea_data)} OEA samples")

    if not instances:
        print("ERROR: No evaluation instances built. Check dataset format.")
        return

    prompts = [inst["prompt"] for inst in instances]

    print(f"Initializing vLLM with {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=256, stop=["<|im_end|>"]
    )

    print("Running batched inference...")
    outputs = llm.generate(prompts, sampling_params)

    # ── Compute metrics ──────────────────────────────────────────────────────
    inst_correct = 0
    tool_correct = 0
    argn_correct = 0
    argn_total = 0
    argv_correct = 0
    argv_total = 0
    parse_failures = 0
    per_sample = []

    for i, out in enumerate(outputs):
        generated_text = out.outputs[0].text
        pred = extract_tool_call(generated_text)
        inst = instances[i]

        gt_name = inst["gt_name"]
        gt_args = inst["gt_args"]

        if pred is None:
            parse_failures += 1
            per_sample.append({
                "sample_idx": inst["sample_idx"],
                "turn_idx": inst["turn_idx"],
                "parsed": False,
                "gt_name": gt_name,
                "gt_args": gt_args,
                "pred_name": None,
                "pred_args": {},
                "name_match": False,
                "inst_correct": False,
                "argn_correct": 0,
                "argv_correct": 0,
                "arg_scores": {},
                "full_raw_output": generated_text,
                "full_prompt_tail": inst["prompt"][-1200:],
            })
            continue

        pred_name = (pred.get("name") or "").strip()
        pred_args = pred.get("arguments") or pred.get("parameters") or {}

        name_match = pred_name.lower() == gt_name.lower()
        if name_match:
            tool_correct += 1

        is_inst = name_match
        arg_scores = {}
        sample_argn = 0
        sample_argv = 0
        for k, v in gt_args.items():
            argn_total += 1
            argv_total += 1
            if k in pred_args:
                argn_correct += 1
                sample_argn += 1
                val_match = str(pred_args[k]).strip().lower() == str(v).strip().lower()
                if val_match:
                    argv_correct += 1
                    sample_argv += 1
                else:
                    is_inst = False
                arg_scores[k] = {"gt": str(v), "pred": str(pred_args[k]),
                                  "argn": True, "argv": val_match}
            else:
                is_inst = False
                arg_scores[k] = {"gt": str(v), "pred": "MISSING",
                                  "argn": False, "argv": False}

        if is_inst:
            inst_correct += 1

        per_sample.append({
            "sample_idx": inst["sample_idx"],
            "turn_idx": inst["turn_idx"],
            "parsed": True,
            "gt_name": gt_name,
            "gt_args": gt_args,
            "pred_name": pred_name,
            "pred_args": pred_args,
            "name_match": name_match,
            "inst_correct": is_inst,
            "argn_correct": sample_argn,
            "argv_correct": sample_argv,
            "arg_scores": arg_scores,
            "full_raw_output": generated_text,
            "full_prompt_tail": inst["prompt"][-1200:],
        })

    total = len(instances)
    parsed_total = total - parse_failures

    metrics = {
        "model": args.model,
        "n_instances": total,
        "n_parsed": parsed_total,
        "parse_failures": parse_failures,
        "Inst": round(inst_correct / total * 100, 2) if total > 0 else 0.0,
        "Tool": round(tool_correct / total * 100, 2) if total > 0 else 0.0,
        "ArgN": round(argn_correct / argn_total * 100, 2) if argn_total > 0 else 0.0,
        "ArgV": round(argv_correct / argv_total * 100, 2) if argv_total > 0 else 0.0,
        "per_sample": per_sample,
    }

    print("\n--- OEA Full Inference Metrics ---")
    summary = {k: v for k, v in metrics.items() if k != "per_sample"}
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
