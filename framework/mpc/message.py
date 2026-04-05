from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


def _estimate_token_count(payload: dict[str, Any] | None) -> int:
	if not payload:
		return 0
	try:
		text = json.dumps(payload, ensure_ascii=False)
	except TypeError:
		text = str(payload)
	try:
		import tiktoken  # type: ignore

		enc = tiktoken.get_encoding("cl100k_base")
		return max(1, len(enc.encode(text)))
	except Exception:
		# Lightweight fallback token estimate.
		return max(1, len(text) // 4)


def _json_safe(value: Any) -> Any:
	try:
		json.dumps(value)
		return value
	except TypeError:
		pass

	if isinstance(value, dict):
		return {str(k): _json_safe(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(v) for v in value]
	return str(value)


@dataclass
class Message:
	# Backward-compatible fields used throughout the current codebase.
	message_type: str
	sender: str
	recipient: str
	payload: dict[str, Any]
	priority: int = 1
	ttl_seconds: int = 60
	message_id: str = field(default_factory=lambda: str(uuid4()))
	created_at: datetime = field(default_factory=datetime.utcnow)

	# MPC schema extensions.
	thread_id: str | None = None
	parent_msg_id: str | None = None
	confidence: float = 1.0
	epoch_ref: int | None = None
	token_count: int = 0
	timestamp_ms: int = 0
	ttl_ms: int = 0

	def __post_init__(self) -> None:
		self.message_type = str(self.message_type).upper()
		self.sender = str(self.sender).lower()
		self.recipient = str(self.recipient).lower()

		if self.thread_id is None:
			self.thread_id = str(uuid4())

		if self.timestamp_ms <= 0:
			# Use wall clock ms directly to avoid timezone ambiguity for naive datetimes.
			self.timestamp_ms = int(time.time() * 1000)
			self.created_at = datetime.utcfromtimestamp(self.timestamp_ms / 1000.0)
		else:
			self.created_at = datetime.utcfromtimestamp(self.timestamp_ms / 1000.0)

		if self.ttl_ms <= 0:
			self.ttl_ms = int(self.ttl_seconds * 1000)
		else:
			self.ttl_seconds = max(1, int(round(self.ttl_ms / 1000.0)))

		if self.token_count <= 0:
			self.token_count = _estimate_token_count(self.payload)

		# Keep priority in the intended range: 0 (heartbeat) to 3 (critical).
		self.priority = max(0, min(3, int(self.priority)))

	# Spec-friendly aliases.
	@property
	def msg_id(self) -> str:
		return self.message_id

	@msg_id.setter
	def msg_id(self, value: str) -> None:
		self.message_id = value

	@property
	def msg_type(self) -> str:
		return self.message_type

	@msg_type.setter
	def msg_type(self, value: str) -> None:
		self.message_type = value.upper()

	@property
	def receiver(self) -> str:
		return self.recipient

	@receiver.setter
	def receiver(self, value: str) -> None:
		self.recipient = value.lower()

	def is_expired(self, now_ms: int | None = None) -> bool:
		now_ms = int(time.time() * 1000) if now_ms is None else now_ms
		return now_ms > (self.timestamp_ms + self.ttl_ms)

	def to_dict(self) -> dict[str, Any]:
		return {
			"msg_id": self.msg_id,
			"sender": self.sender,
			"receiver": self.receiver,
			"thread_id": self.thread_id,
			"parent_msg_id": self.parent_msg_id,
			"msg_type": self.msg_type,
			"payload": _json_safe(self.payload),
			"confidence": self.confidence,
			"epoch_ref": self.epoch_ref,
			"priority": self.priority,
			"token_count": self.token_count,
			"timestamp_ms": self.timestamp_ms,
			"ttl_ms": self.ttl_ms,
		}
