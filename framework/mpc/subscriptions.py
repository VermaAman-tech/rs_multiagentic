from __future__ import annotations

from typing import Dict, Set


DEFAULT_SUBSCRIPTIONS: Dict[str, Set[str]] = {
    "orc": {"TASK", "RESULT", "QUERY", "ALERT", "SYNC", "ERROR", "ACK"},
    "vra": {"TASK", "ALERT", "SYNC"},
    "ga": {"TASK", "ALERT", "SYNC"},
    "pa": {"TASK", "ALERT", "SYNC"},
}


def is_subscribed(agent_id: str, msg_type: str) -> bool:
    return msg_type.upper() in DEFAULT_SUBSCRIPTIONS.get(agent_id.lower(), set())
