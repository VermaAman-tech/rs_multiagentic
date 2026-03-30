#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(pattern))


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 ** 3)


def main() -> None:
    checks = {
        "xbd": {
            "path": Path("data/xbd"),
            "required": [
                Path("data/xbd/train/images"),
                Path("data/xbd/train/labels"),
                Path("data/xbd/test/images"),
                Path("data/xbd/test/labels"),
            ],
        },
        "floodnet": {"path": Path("data/floodnet"), "required": [Path("data/floodnet")]},
        "sen1floods11": {"path": Path("data/sen1floods11"), "required": [Path("data/sen1floods11")]},
        "levircd": {"path": Path("data/levircd"), "required": [Path("data/levircd")]},
        "openearth": {"path": Path("data/openearth_agent"), "required": [Path("data/openearth_agent")]},
        "thinkgeo": {"path": Path("data/thinkgeo"), "required": [Path("data/thinkgeo")]},
    }

    report: dict[str, dict] = {}
    all_ok = True

    for name, spec in checks.items():
        path = spec["path"]
        required = spec["required"]
        missing = [str(p) for p in required if not p.exists()]
        n_files = count_files(path)
        size_gb = round(dir_size_gb(path), 2)

        ok = len(missing) == 0 and n_files > 0
        all_ok = all_ok and ok

        report[name] = {
            "path": str(path),
            "exists": path.exists(),
            "ok": ok,
            "missing": missing,
            "n_files": n_files,
            "size_gb": size_gb,
        }

    out = Path("data/verify_downloads_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not all_ok:
        raise SystemExit("Some dataset checks failed. See data/verify_downloads_report.json")


if __name__ == "__main__":
    main()
