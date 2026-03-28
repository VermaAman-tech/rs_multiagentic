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

    out = {"stage": "stage3_dpo_orc", "data": str(p), "status": "ready_to_train"}
    rp = Path("results/stage3_dpo_orc_ready.json")
    rp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
