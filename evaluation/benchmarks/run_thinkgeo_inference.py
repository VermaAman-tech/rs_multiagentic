"""
ThinkGeo Full Inference Evaluator — Qwen3-4B-Instruct-2507 (text-only)
Computes: TSR (Task Success Rate) and HRR (Hallucination Rejection Rate).

Fix over previous version:
    - Uses Qwen3-4B-Instruct-2507 (text-only, stable) as required by plan.
  - Per-task tool descriptions still included in system prompt.
  - Full verbose per-sample trace: full prompt, full raw output, gt/pred, scores.
"""

# ── Triton patch must come before any vllm import ────────────────────────────
import sys

def _patch_triton():
    try:
        import triton.backends.nvidia.driver as _drv
        class _MockCudaUtils:
            _inst = None
            def __new__(cls):
                if cls._inst is None:
                    cls._inst = super().__new__(cls)
                return cls._inst
            def get_device_properties(self, *a, **kw): return {"max_shared_mem": 0}
            def load_binary(self, *a, **kw): raise NotImplementedError
            def cuOccupancyMaxActiveClusters(self, *a, **kw): return 0
            def set_printf_fifo_size(self, *a, **kw): pass
            def fill_1d_tma_descriptor(self, *a, **kw): pass
            def fill_2d_tma_descriptor(self, *a, **kw): pass
        _drv.CudaUtils = _MockCudaUtils
        _drv.CudaUtils._inst = _MockCudaUtils()
        print("[triton_patch] Triton CudaUtils patched.")
    except Exception as e:
        print(f"[triton_patch] patch skipped: {e}")

_patch_triton()
# ─────────────────────────────────────────────────────────────────────────────

import json
import argparse
import os
import re
from vllm import LLM, SamplingParams


def format_tool_descriptions(tools: list[dict]) -> str:
    """Format per-task tool descriptions for the system prompt."""
    lines = []
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        inputs = tool.get("inputs", [])
        param_parts = []
        for inp in inputs:
            p_name = inp.get("name", "?")
            p_type = inp.get("type", "any")
            p_desc = inp.get("description", "")
            opt = " (optional)" if inp.get("optional", False) else ""
            detail = f" — {p_desc}" if p_desc else ""
            param_parts.append(f"    - {p_name} ({p_type}{opt}){detail}")
        params_str = "\n".join(param_parts) if param_parts else "    (no parameters)"
        lines.append(f"  {name}: {desc}\n  Parameters:\n{params_str}")
    return "\n".join(lines)


def make_system_prompt(tools: list[dict]) -> str:
    tool_desc = format_tool_descriptions(tools)
    return (
        "You are an expert geospatial AI assistant that solves tasks step-by-step "
        "using the available tools.\n\n"
        "Available tools:\n" + tool_desc + "\n\n"
        "When you need to use a tool, output ONLY a raw JSON object (no markdown, "
        "no preamble, no code fences):\n"
        '{"name": "<ToolName>", "arguments": {"param1": "value1"}}\n\n'
        "When you have the final answer, state it directly in plain text.\n"
        "If the task cannot be solved with the given tools or image, say clearly: "
        "'I cannot solve this task with the available tools.'"
    )


def extract_tool_call(text: str) -> dict | None:
    """Extract a tool call JSON from text output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json|python)?\n?", "", text).replace("```", "").strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if obj.get("name"):
            return obj
    except Exception:
        pass

    # Brace-matching scan
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(text[start:i+1])
                    if obj.get("name"):
                        return obj
                except Exception:
                    pass
                start = -1
    return None


def check_gt_match(generated_text: str, gt_answer) -> bool:
    """Check if generated text matches gt answer across legacy schemas."""
    if not gt_answer:
        return False

    gen = generated_text.lower()

    # Legacy schema: gt_answer is a list of acceptable answer snippets.
    if isinstance(gt_answer, list):
        terms = [str(x).strip().lower() for x in gt_answer if str(x).strip()]
        return any(term in gen for term in terms) if terms else False

    # Current schema: {"whitelist": [[...], ...], "blacklist": [...]}
    if isinstance(gt_answer, dict):
        for term in (gt_answer.get("blacklist") or []):
            if isinstance(term, str) and term.lower() in gen:
                return False
        for group in (gt_answer.get("whitelist") or []):
            if not isinstance(group, list):
                group = [group]
            if not any(str(item).lower() in gen for item in group):
                return False
        return True

    return False


def build_eval_instances(items: list) -> list[dict]:
    """
    Build one text-only evaluation instance per ThinkGeo task.
    Feeds the full conversation history as context (text only, no images).
    """
    instances = []

    for task_id, sample in items:
        dialogs = sample.get("dialogs", [])
        tools = sample.get("tools", [])
        gt_answer = sample.get("gt_answer", None)

        if not dialogs:
            continue

        solvable = gt_answer is not None
        system_prompt = make_system_prompt(tools)

        # Build conversation messages from dialog turns
        messages = [{"role": "system", "content": system_prompt}]

        for turn in dialogs:
            role = turn.get("role", "")
            content = turn.get("content", "")
            thought = turn.get("thought", "")
            tool_calls = turn.get("tool_calls", [])

            if role == "user":
                messages.append({"role": "user", "content": content})

            elif role == "assistant":
                if tool_calls:
                    fn = tool_calls[0].get("function", {})
                    tc_text = ""
                    if thought:
                        tc_text += f"Thought: {thought}\n"
                    tc_text += json.dumps({
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", {})
                    })
                    messages.append({"role": "assistant", "content": tc_text})
                # Final answer turns are what we predict — don't add to prefix

            elif role == "tool":
                tool_name = turn.get("name", "")
                raw = turn.get("content", {})
                obs = raw.get("content", str(raw)) if isinstance(raw, dict) else str(raw)
                messages.append({
                    "role": "user",
                    "content": f"[Tool Result — {tool_name}]: {obs}"
                })

        instances.append({
            "task_id": task_id,
            "messages": messages,
            "gt_answer": gt_answer,
            "solvable": solvable,
            "tool_names": [t.get("name", "") for t in tools],
        })

    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tg-data", default="data/thinkgeo/ThinkGeoBench.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="results/e5_thinkgeo_real.json")
    args = parser.parse_args()

    with open(args.tg_data) as f:
        tg_data = json.load(f)

    items = list(tg_data.items())
    if args.limit > 0:
        items = items[:args.limit]
    print(f"Loaded {len(items)} ThinkGeo tasks")

    instances = build_eval_instances(items)
    print(f"Built {len(instances)} evaluation instances")
    solvable_n = sum(1 for inst in instances if inst["solvable"])
    print(f"Solvable: {solvable_n}, Unsolvable: {len(instances) - solvable_n}")

    print(f"Initializing vLLM with {args.model} (text-only, no VL)...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    all_messages = [inst["messages"] for inst in instances]
    print("Running batched inference...")
    outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)

    REFUSAL_TERMS = [
        "cannot solve", "cannot be determined", "i cannot", "unable to",
        "not possible", "insufficient", "no information", "i don't know",
        "cannot answer", "not enough", "sorry", "can't",
    ]

    tsr_correct = 0
    hrr_correct = 0
    total_solvable = 0
    total_unsolvable = 0
    parse_attempts = 0
    parse_successes = 0
    per_sample = []

    for i, out in enumerate(outputs):
        generated_text = out.outputs[0].text
        inst = instances[i]
        gt = inst["gt_answer"]
        solvable = inst["solvable"]

        if solvable:
            total_solvable += 1
            matched = check_gt_match(generated_text, gt)
            if matched:
                tsr_correct += 1

            # Also check if a tool call was attempted
            parse_attempts += 1
            tc = extract_tool_call(generated_text)
            if tc:
                parse_successes += 1
        else:
            total_unsolvable += 1
            matched = any(t in generated_text.lower() for t in REFUSAL_TERMS)
            if matched:
                hrr_correct += 1

        # Build full verbose prompt text from messages
        full_prompt_text = ""
        for msg in inst["messages"]:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            full_prompt_text += f"[{role.upper()}]: {content}\n"

        per_sample.append({
            "task_id": inst["task_id"],
            "solvable": solvable,
            "available_tools": inst["tool_names"],
            "gt_answer": gt,
            "tsr_correct": matched if solvable else None,
            "hrr_refused": matched if not solvable else None,
            "extracted_tool_call": extract_tool_call(generated_text) if solvable else None,
            "full_raw_output": generated_text,
            "full_prompt": full_prompt_text[-2000:],
        })

    metrics = {
        "model": args.model,
        "dataset": "ThinkGeoBench",
        "total_samples": len(outputs),
        "solvable_count": total_solvable,
        "unsolvable_count": total_unsolvable,
        "TSR": round(tsr_correct / total_solvable * 100 if total_solvable > 0 else 0, 2),
        "HRR": round(hrr_correct / total_unsolvable * 100 if total_unsolvable > 0 else 0, 2),
        "tsr_correct": tsr_correct,
        "hrr_correct": hrr_correct,
        "tool_parse_rate": round(parse_successes / parse_attempts * 100 if parse_attempts > 0 else 0, 2),
        "per_sample": per_sample,
    }

    print("\n--- ThinkGeo Inference Metrics ---")
    summary = {k: v for k, v in metrics.items() if k != "per_sample"}
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
