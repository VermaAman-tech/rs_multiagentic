from __future__ import annotations

import math


def calculator(expression: str) -> dict:
    """Safely evaluate a math expression and return a numeric result."""
    allowed = {
        "__builtins__": {},
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "int": int,
        "float": float,
    }
    try:
        result = float(eval(compile(expression, "<calc>", "eval"), allowed))
        return {"result": result}
    except Exception as exc:
        return {"result": 0.0, "error": str(exc)}
