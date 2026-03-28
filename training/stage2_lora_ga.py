import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    p = Path(args.data)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    out = {"stage": "stage2_lora_ga", "data": str(p), "status": "ready_to_train"}
    rp = Path("results/stage2_lora_ga_ready.json")
    rp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
