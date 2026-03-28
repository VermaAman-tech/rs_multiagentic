from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4


@dataclass
class Message:
	message_type: str
	sender: str
	recipient: str
	payload: dict[str, Any]
	priority: int = 5
	ttl_seconds: int = 60
	message_id: str = field(default_factory=lambda: str(uuid4()))
	created_at: datetime = field(default_factory=datetime.utcnow)

	def is_expired(self, now: datetime | None = None) -> bool:
		now = now or datetime.utcnow()
		return now > self.created_at + timedelta(seconds=self.ttl_seconds)
