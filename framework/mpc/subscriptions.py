from __future__ import annotations

from typing import Dict, Set


DEFAULT_SUBSCRIPTIONS: Dict[str, Set[str]] = {
    "orc": {"TASK", "RESULT", "SYNC", "CONFLICT", "DEADLOCK", "SYSTEM", "TERMINATE"},
    "vra": {"TASK", "SYNC", "TERMINATE"},
    "ga": {"TASK", "SYNC", "TERMINATE"},
    "pa": {"TASK", "SYNC", "TERMINATE"},
}
