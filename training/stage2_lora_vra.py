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

    out = {
        "stage": "stage2_lora_vra",
        "data": str(p),
        "status": "ready_to_train",
        "config": {"lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 3},
    }
    rp = Path("results/stage2_lora_vra_ready.json")
    rp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
