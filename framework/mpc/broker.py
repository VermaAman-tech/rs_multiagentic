from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from framework.mpc.message import Message


@dataclass(order=True)
class _QueuedMessage:
    sort_key: tuple = field(init=False)
    msg: Message = field(compare=False)

    def __post_init__(self) -> None:
        self.sort_key = (self.msg.priority, self.msg.created_at.timestamp())


class MessageBroker:
    def __init__(self) -> None:
        self._queue: List[_QueuedMessage] = []

    def publish(self, msg: Message) -> None:
        if msg.sender != "orc" and msg.recipient != "orc":
            msg.recipient = "orc"
            msg.payload = {"forward_to": msg.payload.get("forward_to"), "wrapped": msg.payload}
        heapq.heappush(self._queue, _QueuedMessage(msg=msg))

    def drain_for(self, recipient: str) -> list[Message]:
        now = datetime.utcnow()
        kept: list[_QueuedMessage] = []
        out: list[Message] = []
        while self._queue:
            q = heapq.heappop(self._queue)
            if q.msg.is_expired(now):
                continue
            if q.msg.recipient == recipient:
                out.append(q.msg)
            else:
                kept.append(q)
        for q in kept:
            heapq.heappush(self._queue, q)
        return out
