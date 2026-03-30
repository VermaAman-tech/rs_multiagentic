from __future__ import annotations

from pathlib import Path


def plot(x_values: list[float], y_values: list[float], output_path: str) -> dict:
    """Render an XY plot and save it to disk."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_values, y_values, marker="o")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return {"success": True, "output_path": str(out)}
    except Exception as exc:
        return {"success": False, "error": str(exc), "output_path": output_path}
