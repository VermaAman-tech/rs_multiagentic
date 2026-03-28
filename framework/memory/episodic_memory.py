from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MemoryRecord:
    version: int
    value: Any


class EpisodicMemory:
    def __init__(self) -> None:
        self._store: Dict[str, MemoryRecord] = {}
        self.ccq: float = 1.0

    def write(self, key: str, value: Any) -> int:
        prev = self._store.get(key)
        version = 1 if prev is None else prev.version + 1
        self._store[key] = MemoryRecord(version=version, value=copy.deepcopy(value))
        return version

    def read(self, key: str, default: Any | None = None) -> Any:
        rec = self._store.get(key)
        return default if rec is None else copy.deepcopy(rec.value)

    def token_count(self) -> int:
        return sum(len(str(v.value)) for v in self._store.values())

    def items(self) -> dict[str, Any]:
        return {k: copy.deepcopy(v.value) for k, v in self._store.items()}

    def set_items(self, new_items: dict[str, Any], ccq: float) -> None:
        self._store.clear()
        for k, v in new_items.items():
            self.write(k, v)
        self.ccq = ccq
