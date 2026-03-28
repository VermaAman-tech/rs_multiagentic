"""Plot tool - executes matplotlib code and returns figure as image."""
import io, base64

def plot(command: str) -> dict:
    """Execute Python plotting code containing a solution() function that returns a matplotlib figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        local_ns = {}
        exec(command, {"__builtins__": __builtins__, "plt": plt, "matplotlib": matplotlib}, local_ns)

        if "solution" in local_ns:
            fig = local_ns["solution"]()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            return {"image_base64": img_b64, "success": True}
        else:
            return {"image_base64": "", "success": False, "error": "No solution() function found"}
    except Exception as e:
        return {"image_base64": "", "success": False, "error": str(e)}
