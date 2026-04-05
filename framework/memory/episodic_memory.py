from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict


@dataclass
class MemoryRecord:
    version: int
    value: Any
    created_at: datetime
    ttl_seconds: int | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.ttl_seconds is None or self.ttl_seconds <= 0:
            return False
        now = now or datetime.utcnow()
        return now > (self.created_at + timedelta(seconds=self.ttl_seconds))


class EpisodicMemory:
    def __init__(self) -> None:
        self._store: Dict[str, MemoryRecord] = {}
        self._history: Dict[str, list[MemoryRecord]] = {}
        self.ccq: float = 1.0

    def write(self, key: str, value: Any, ttl_seconds: int | None = None) -> int:
        prev = self._store.get(key)
        version = 1 if prev is None else prev.version + 1
        rec = MemoryRecord(
            version=version,
            value=copy.deepcopy(value),
            created_at=datetime.utcnow(),
            ttl_seconds=ttl_seconds,
        )
        self._store[key] = rec
        self._history.setdefault(key, []).append(rec)
        return version

    def read(self, key: str, default: Any | None = None) -> Any:
        self.gc()
        rec = self._store.get(key)
        return default if rec is None else copy.deepcopy(rec.value)

    def token_count(self) -> int:
        self.gc()
        total = 0
        for v in self._store.values():
            try:
                import tiktoken  # type: ignore

                enc = tiktoken.get_encoding("cl100k_base")
                total += len(enc.encode(str(v.value)))
            except Exception:
                total += max(1, len(str(v.value)) // 4)
        return total

    def items(self) -> dict[str, Any]:
        self.gc()
        return {k: copy.deepcopy(v.value) for k, v in self._store.items()}

    def set_items(self, new_items: dict[str, Any], ccq: float) -> None:
        self._store.clear()
        self._history.clear()
        for k, v in new_items.items():
            self.write(k, v)
        self.ccq = ccq

    def gc(self) -> None:
        now = datetime.utcnow()
        stale = [k for k, rec in self._store.items() if rec.is_expired(now)]
        for k in stale:
            self._store.pop(k, None)

    def history(self, key: str) -> list[dict[str, Any]]:
        return [
            {
                "version": rec.version,
                "value": copy.deepcopy(rec.value),
                "created_at": rec.created_at.isoformat() + "Z",
                "ttl_seconds": rec.ttl_seconds,
            }
            for rec in self._history.get(key, [])
        ]
