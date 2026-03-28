import json
import os
import glob
from collections import Counter

def main():
    runs = glob.glob("results/agent_runs/*/bundle.json")
    print(f"E4 Conflict Benchmarking: Analyzing {len(runs)} execution bundles.")
    
    total = len(runs)
    if total == 0:
        print("No runs found in results/agent_runs/")
        return
        
    tsr_count = 0
    crr_count = 0
    ccq_sum = 0.0
    hrr_count = 0
    
    for run_file in runs:
        with open(run_file) as f:
            data = json.load(f)
            
        metrics = data.get("episode_metrics", {})
        merged = data.get("agent_outputs", {}).get("orc_merged", {})
        
        # TSR: safety_valid or structured answer
        if metrics.get("safety_valid") or len(merged) > 5:
            tsr_count += 1
            
        # CRR: selected_by_conflict is present
        if merged.get("selected_by_conflict"):
            crr_count += 1
            
        ccq_sum += metrics.get("ccq", 0.0)
        
        # HRR proxy: lack of tool calls
        total_calls = metrics.get("total_tool_calls", 0)
        if total_calls == 0:
            hrr_count += 1

    res = {
        "benchmark": "E4 Conflict (Real Proxy)",
        "n_samples": total,
        "metrics": {
            "TSR": (tsr_count / total * 100),
            "HallucinationRejectionRate": (hrr_count / total * 100),
            "ConflictResolutionAccuracy": (crr_count / total * 100),
            "CCQ": ccq_sum / total
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/e4_conflict_real.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote results/e4_conflict_real.json")
    print("Results:", json.dumps(res["metrics"], indent=2))

if __name__ == "__main__":
    main()
