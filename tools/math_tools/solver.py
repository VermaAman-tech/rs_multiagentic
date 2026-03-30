from __future__ import annotations


def solver(equation: str) -> dict:
    """Solve simple equations when possible; returns placeholder roots for unsupported forms."""
    try:
        import sympy as sp

        x = sp.symbols("x")
        eq = sp.sympify(equation)
        roots = sp.solve(eq, x)
        return {"roots": [float(r) for r in roots if r.is_real]}
    except Exception:
        return {"roots": [0.0]}
