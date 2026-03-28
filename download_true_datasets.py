import os
import json

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

def main():
    print("Initiating True Dataset Downloads via HuggingFace Hub...")
    
    # 1. Download OpenEarthAgent
    print("Downloading MBZUAI/OpenEarthAgent...")
    try:
        # Some datasets use 'test', 'validation', or 'train'
        oea = load_dataset("MBZUAI/OpenEarthAgent", split="test") 
    except Exception as e:
        print(f"Failed to find test split, trying validation. Error: {e}")
        oea = load_dataset("MBZUAI/OpenEarthAgent", split="validation")
        
    os.makedirs("data/openearth_agent", exist_ok=True)
    with open("data/openearth_agent/eval.jsonl", "w") as f:
        for item in oea:
            f.write(json.dumps(item, default=str) + "\n")
    print(f"✅ OpenEarthAgent downloaded: {len(oea)} real spatial samples saved.")

    # 2. Download ThinkGeo
    print("\nDownloading MBZUAI/ThinkGeo...")
    try:
        tg = load_dataset("MBZUAI/ThinkGeo", split="test") 
    except Exception as e:
        print(f"Failed to find test split, trying train. Error: {e}")
        try:
            tg = load_dataset("MBZUAI/ThinkGeo", split="train")
        except Exception:
            tg = load_dataset("MBZUAI/ThinkGeo")['train'] # Fallback
            
    os.makedirs("data/thinkgeo", exist_ok=True)
    with open("data/thinkgeo/eval.jsonl", "w") as f:
        for item in tg:
            f.write(json.dumps(item, default=str) + "\n")
    print(f"✅ ThinkGeo downloaded: {len(tg)} real spatial samples saved.")
    
    print("\n==================================")
    print("True evaluation sequences successfully embedded to disk.")

if __name__ == "__main__":
    main()
