from __future__ import annotations

import math


def calculator(expression: str) -> dict:
    """Safely evaluate a math expression and return a numeric result."""
    allowed = {
        "__builtins__": {},
        "math": math,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
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
        return {"result": result, "success": True}
    except Exception as exc:
        return {"result": 0.0, "success": False, "error": str(exc)}
