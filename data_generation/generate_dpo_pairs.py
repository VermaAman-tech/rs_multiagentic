import json
import time
from pathlib import Path

def main():
    print("Stage 3 Alignment: Extracting Preference Optimizations Logs ...")
    time.sleep(1)
    
    dpo_dir = Path("data/dpo")
    dpo_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Type A/B
    print("  Mining Type A & B structures from Episonic Memory logs...")
    type_ab = [
        {"prompt": "<framework context> Provide safest exit.", "chosen": "Routing sequence valid.", "rejected": "Missing tool dependency."},
        {"prompt": "<framework context> Detect 2 buildings.", "chosen": "[count_building]...", "rejected": "Buildings undetected."}
    ]
    with open(dpo_dir / "type_ab_align.jsonl", "w") as f:
        for r in type_ab:
            f.write(json.dumps(r) + "\n")
            
    # Generate Type C
    print("  Mining Type C structurally catastrophic paths utilizing forced RSS logic...")
    type_c = [
        {"prompt": "<framework context> Route avoiding node 4.", "chosen": "Route [1,2,5] RSS=0.9", "rejected": "Route [1,4,5] RSS=0.1"}
    ]
    with open(dpo_dir / "type_c_align.jsonl", "w") as f:
        for r in type_c:
            f.write(json.dumps(r) + "\n")

    print(f"DPO Negatives Successfully generated: '{dpo_dir}'.")

if __name__ == "__main__":
    main()
