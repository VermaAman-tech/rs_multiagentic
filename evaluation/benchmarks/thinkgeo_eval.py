"""
Real E5 ThinkGeo Evaluation
Evaluates MAGRF's tool coverage and task structure against the actual ThinkGeoBench.json.
"""
import json, os, argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/thinkgeo/ThinkGeoBench.json")
    parser.add_argument("--output", default="results/e5_thinkgeo_real.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.data) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} ThinkGeo evaluation tasks")

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

    # Analyze task structure
    task_types = Counter()
    all_gt_tools = Counter()
    tasks_with_images = 0
    tasks_solvable = 0
    tool_match_count = 0
    tool_total_count = 0

    for task_id, task in data.items():
        # Count tools available per task
        task_tools = [t["name"] for t in task.get("tools", [])]
        for t in task_tools:
            all_gt_tools[t] += 1
            tool_total_count += 1
            if t in our_tools:
                tool_match_count += 1

        # Check if has image files
        if task.get("files"):
            tasks_with_images += 1

        # Check if has ground truth answer
        if task.get("gt_answer") is not None:
            tasks_solvable += 1

        # Analyze dialog structure for task classification
        dialogs = task.get("dialogs", [])
        if any("route" in str(d).lower() or "path" in str(d).lower() or "distance" in str(d).lower() for d in dialogs):
            task_types["routing_or_distance"] += 1
        elif any("detect" in str(d).lower() or "count" in str(d).lower() or "segment" in str(d).lower() for d in dialogs):
            task_types["vision"] += 1
        elif any("index" in str(d).lower() or "ndvi" in str(d).lower() or "ndbi" in str(d).lower() for d in dialogs):
            task_types["spectral"] += 1
        else:
            task_types["gis_general"] += 1

    missing_tools = set(all_gt_tools.keys()) - our_tools
    tool_coverage = tool_match_count / tool_total_count * 100 if tool_total_count > 0 else 0

    results = {
        "benchmark": "E5 ThinkGeo (Real Dataset)",
        "dataset": args.data,
        "n_tasks": len(data),
        "n_with_images": tasks_with_images,
        "n_solvable": tasks_solvable,
        "n_unique_gt_tools": len(all_gt_tools),
        "task_type_distribution": dict(task_types),
        "metrics": {
            "Tool_Coverage": round(tool_coverage, 2),
            "Tools_Implemented": len(our_tools),
            "GT_Tools_Required": len(all_gt_tools),
            "Missing_Tools": list(missing_tools),
        },
        "gt_tool_distribution": dict(all_gt_tools.most_common()),
        "note": "Tool coverage measured against ground-truth task definitions. Full inference metrics (TSR, HRR) require vLLM model server running with Qwen3-4B-Instruct-2507 as per plan."
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Tool Coverage: {tool_coverage:.1f}%")
    if missing_tools:
        print(f"Missing tools: {missing_tools}")

if __name__ == "__main__":
    main()
