from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List

from framework.mpc.message import Message


@dataclass(order=True)
class _QueuedMessage:
    sort_key: tuple = field(init=False)
    msg: Message = field(compare=False)

    def __post_init__(self) -> None:
        # Higher priority value should be delivered first.
        self.sort_key = (-self.msg.priority, self.msg.timestamp_ms)


class MessageBroker:
    def __init__(self) -> None:
        self._queue: List[_QueuedMessage] = []

    def publish(self, msg: Message) -> None:
        # MPC rule: no direct agent-to-agent traffic; all non-ORC messages route to ORC first.
        if msg.sender != "orc" and msg.recipient != "orc":
            original_recipient = msg.recipient
            msg.recipient = "orc"
            msg.payload = {
                "forward_to": original_recipient,
                "wrapped": msg.payload,
            }
        heapq.heappush(self._queue, _QueuedMessage(msg=msg))

    def drain_for(self, recipient: str) -> list[Message]:
        recipient = recipient.lower()
        kept: list[_QueuedMessage] = []
        out: list[Message] = []
        while self._queue:
            q = heapq.heappop(self._queue)
            if q.msg.is_expired():
                continue
            if q.msg.recipient == recipient:
                out.append(q.msg)
            else:
                kept.append(q)
        for q in kept:
            heapq.heappush(self._queue, q)
        return out
