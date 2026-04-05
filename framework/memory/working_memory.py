from __future__ import annotations


class WorkingMemory:
    def __init__(self, token_limit: int = 4096) -> None:
        self.token_limit = token_limit
        self._buffer: list[str] = []
        self._token_estimate = 0

    def append(self, text: str) -> bool:
        t = len(text)
        self._buffer.append(text)
        self._token_estimate += t
        if self._token_estimate > self.token_limit:
            self._buffer = []
            self._token_estimate = 0
            return True
        return False

    def flush(self) -> str:
        joined = "\n".join(self._buffer)
        self._buffer = []
        self._token_estimate = 0
        return joined

    def view(self) -> str:
        return "\n".join(self._buffer)
