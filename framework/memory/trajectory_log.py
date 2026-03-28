from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TrajectoryLog:
    def __init__(self, path: str = "logs/trajectory.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, actor: str, entry: dict[str, Any]) -> None:
        if actor != "orc":
            raise PermissionError("Trajectory log is ORC-only write")
        row = {"ts": datetime.utcnow().isoformat() + "Z", **entry}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
