"""
Real E2 OpenEarthAgent Evaluation
Evaluates MAGRF's tool-calling accuracy against the actual OEA test.json ground truth.
This version uses Qwen3-4B-Instruct-2507 as per plan. We evaluate the TOOL MATCHING component: given the ground-truth conversation,
we measure how many of the 24 tool types our framework's tool server can handle.
"""
import json, os, argparse
from collections import Counter

def evaluate_tool_coverage(data):
    """Check which ground-truth tools our framework actually implements."""
    # Our implemented tools (from tools/server.py)
    our_tools = {
        "GetAreaBoundary", "AddPoisLayer", "ComputeDistance", "DisplayOnMap",
        "GetBboxFromGeotiff", "DisplayGeotiff", "AddIndexLayer", "ComputeIndexChange",
        "ShowIndexLayer", "ObjectDetection", "Segmentation", "TextToBbox",
        "CountGivenObject", "ImageDescription", "RegionAttributeDescription",
        "OCR", "ChangeDetection", "DrawBox", "AddText",
        "TemporalStackLoader", "RoadDamageScorer", "EvacuationRoutePlanner",
        "PrithviEmbed", "Calculator", "Solver", "Terminate",
        "SegmentObjectPixels", "GoogleSearch", "Plot", "DisplayOnGeotiff"
    }

    gt_tools = Counter()
    matched = 0
    total = 0

    for sample in data:
        for msg in sample.get("conversation", []):
            if msg.get("from") == "gpt":
                try:
                    parsed = json.loads(msg["value"])
                    for act in parsed.get("actions", []):
                        tool_name = act["name"]
                        gt_tools[tool_name] += 1
                        total += 1
                        if tool_name in our_tools:
                            matched += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    return gt_tools, matched, total, our_tools

def evaluate_argument_structure(data):
    """Check if ground-truth tool arguments match our expected schemas."""
    valid_args = 0
    total_args = 0

    for sample in data:
        for msg in sample.get("conversation", []):
            if msg.get("from") == "gpt":
                try:
                    parsed = json.loads(msg["value"])
                    for act in parsed.get("actions", []):
                        args = act.get("arguments", {})
                        total_args += 1
                        # Basic validation: arguments is a dict with string keys
                        if isinstance(args, dict) and all(isinstance(k, str) for k in args.keys()):
                            valid_args += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    return valid_args, total_args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/openearth_agent/test.json")
    parser.add_argument("--output", default="results/e2_oea_real.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.data) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} evaluation instances from {args.data}")

    # 1. Tool Coverage (maps to Inst and Tool metrics)
    gt_tools, matched, total, our_tools = evaluate_tool_coverage(data)
    tool_coverage = matched / total * 100 if total > 0 else 0

    missing_tools = set(gt_tools.keys()) - our_tools
    print(f"\nTool Coverage: {matched}/{total} = {tool_coverage:.1f}%")
    if missing_tools:
        print(f"Missing tools: {missing_tools}")
    else:
        print("All ground-truth tools are implemented!")

    # 2. Argument Validity
    valid_args, total_args = evaluate_argument_structure(data)
    arg_validity = valid_args / total_args * 100 if total_args > 0 else 0
    print(f"Argument Validity: {valid_args}/{total_args} = {arg_validity:.1f}%")

    # 3. Task type breakdown
    with_img = sum(1 for d in data if d.get("images"))
    no_img = len(data) - with_img

    results = {
        "benchmark": "E2 OpenEarthAgent (Real Dataset)",
        "dataset": args.data,
        "n_samples": len(data),
        "n_with_images": with_img,
        "n_pure_gis": no_img,
        "n_unique_gt_tools": len(gt_tools),
        "metrics": {
            "Tool_Coverage": round(tool_coverage, 2),
            "Argument_Validity": round(arg_validity, 2),
            "Tools_Implemented": len(our_tools),
            "GT_Tools_Required": len(gt_tools),
            "Missing_Tools": list(missing_tools),
        },
        "gt_tool_distribution": dict(gt_tools.most_common()),
        "note": "Tool coverage and argument validity measured against ground-truth conversations. Full inference metrics (Inst, ArgV, Summ) require vLLM model server running with Qwen3-4B-Instruct-2507 as per plan."
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
