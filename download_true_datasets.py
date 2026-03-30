#!/usr/bin/env python3
from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.check_call(["bash", "scripts/download_datasets.sh", "all"])


if __name__ == "__main__":
    main()
