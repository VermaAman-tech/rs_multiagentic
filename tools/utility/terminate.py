from __future__ import annotations

from typing import Any


def terminate(final_answer: Any = None, ans: Any = None) -> dict:
    """Return a normalized terminate payload."""
    answer = final_answer if final_answer is not None else ans
    return {"status": "terminated", "final_answer": answer}
