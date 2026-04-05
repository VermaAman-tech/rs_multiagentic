from __future__ import annotations


def solver(equation: str) -> dict:
    """Solve simple equations when possible and return explicit errors when not solvable."""
    try:
        import sympy as sp

        x = sp.symbols("x")
        eq = sp.sympify(equation)
        roots = sp.solve(eq, x)
        return {"roots": [float(r) for r in roots if r.is_real], "success": True}
    except Exception as exc:
        return {"roots": [], "success": False, "error": str(exc)}
