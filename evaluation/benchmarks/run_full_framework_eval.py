#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import httpx

from agents.base_agent import BaseAgent
from evaluation.metrics import ArgN, ArgV, HRR, Inst, MGAR, RSS, Summ, TLS, TSR, Tool
from framework.episode_runner import EpisodeRunner


REFUSAL_TERMS = {
    "cannot solve",
    "cannot be determined",
    "i cannot",
    "unable to",
    "not possible",
    "insufficient",
    "no information",
    "i don't know",
    "cannot answer",
    "not enough",
    "sorry",
    "can't",
    "unsure",
    "uncertain",
}

TOOL_ALIASES = {
    "displaygeotiff": "displayongeotiff",
    "displayongeotiff": "displayongeotiff",
    "segmentation": "segmentobjectpixels",
    "segmentobjectpixels": "segmentobjectpixels",
}

ARG_NAME_ALIASES = {
    "area_name": "area",
    "text_prompt": "text",
    "image_path": "image",
    "image_path_1": "pre_image",
    "image_path_2": "post_image",
    "geotiff_path": "geotiff",
    "index_name": "index_type",
    "index_path_pre": "layer1_name",
    "index_path_post": "layer2_name",
    "index_array_path": "layer_name",
    "equation": "command",
}

ROUTING_HINTS = (
    "route",
    "path",
    "evacuation",
    "safest",
    "navigate",
    "travel",
    "reach shelter",
    "corridor",
)

_TOOL_ARG_NORMALIZER = BaseAgent()

REPLAY_REASONING_BY_TOOL = {
    "getareaboundary": "Establish the exact geographic analysis boundary for the requested place.",
    "addpoislayer": "Retrieve points of interest for the requested category inside the active boundary.",
    "computedistance": "Compute distances between candidate entities to identify the closest relation.",
    "texttobbox": "Detect target objects and extract precise bounding boxes from the image.",
    "calculator": "Compute the required numeric quantity from the prior observation values.",
    "regionattributedescription": "Describe the requested attribute for the detected target region.",
    "addindexlayer": "Generate spectral index layers required for temporal comparison.",
    "computeindexchange": "Compute change between pre and post index layers to quantify difference.",
    "terminate": "Finalize the answer using accumulated verified observations.",
}


def _next_eval_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums: list[int] = []
    for p in root.glob("eval_*"):
        if p.is_dir():
            try:
                nums.append(int(p.name.split("_")[-1]))
            except ValueError:
                continue
    nxt = (max(nums) + 1) if nums else 1
    out = root / f"eval_{nxt:03d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _safe_load_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _extract_observation_payload(human_text: str) -> str:
    text = _normalize_space(human_text)
    if not text:
        return ""
    idx = text.find("OBSERVATION:")
    if idx == -1:
        return ""
    payload = text[idx + len("OBSERVATION:") :].strip()
    payload = payload.replace("Please summarize the model outputs and answer my first question.", "").strip()
    return _normalize_space(payload)


def _replay_reasoning_for(tool_name: str) -> str:
    n = _normalize_tool_name(tool_name)
    return REPLAY_REASONING_BY_TOOL.get(
        n,
        f"Execute {tool_name} as specified by the validated OpenEarth reference step.",
    )


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_text(text: str) -> str:
    txt = _normalize_space(text).lower()
    txt = re.sub(r"[^a-z0-9\.\-\s]", " ", txt)
    return _normalize_space(txt)


def _normalize_tool_name(name: str) -> str:
    key = _normalize_text(name).replace(" ", "")
    return TOOL_ALIASES.get(key, key)


def _normalize_arg_name(name: str) -> str:
    key = _normalize_text(name).replace(" ", "_")
    return ARG_NAME_ALIASES.get(key, key)


def _extract_numbers(text: str) -> list[float]:
    nums: list[float] = []
    for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or ""):
        try:
            nums.append(float(tok))
        except ValueError:
            continue
    return nums


def _to_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value):
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _value_match(gt_val: Any, pred_val: Any, rel_tol: float = 0.05) -> bool:
    g_num = _to_number(gt_val)
    p_num = _to_number(pred_val)
    if g_num is not None and p_num is not None:
        tol = max(abs(g_num) * rel_tol, 1e-6)
        return abs(g_num - p_num) <= tol

    if isinstance(gt_val, bool) and isinstance(pred_val, bool):
        return gt_val == pred_val

    if isinstance(gt_val, (dict, list)) or isinstance(pred_val, (dict, list)):
        return _normalize_text(json.dumps(gt_val, sort_keys=True)) == _normalize_text(
            json.dumps(pred_val, sort_keys=True)
        )

    return _normalize_text(str(gt_val)) == _normalize_text(str(pred_val))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _rouge_l_f1(pred: str, gt: str) -> float:
    p_tokens = _normalize_text(pred).split()
    g_tokens = _normalize_text(gt).split()
    if not p_tokens or not g_tokens:
        return 0.0
    lcs = _lcs_len(p_tokens, g_tokens)
    prec = lcs / len(p_tokens)
    rec = lcs / len(g_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _extract_answer_from_obj(obj: Any) -> str:
    if obj is None:
        return ""

    if isinstance(obj, (int, float, bool)):
        return str(obj)

    if isinstance(obj, str):
        txt = obj.strip()
        if not txt:
            return ""
        parsed = _safe_load_json(txt)
        if parsed:
            return _extract_answer_from_obj(parsed)
        return txt

    if isinstance(obj, dict):
        for key in ("final_answer", "ans", "answer", "response"):
            if key in obj and obj[key] not in (None, ""):
                return _extract_answer_from_obj(obj[key])

        actions = obj.get("actions")
        if isinstance(actions, list):
            for act in actions:
                if not isinstance(act, dict):
                    continue
                if _normalize_tool_name(str(act.get("name", ""))) == "terminate":
                    args = act.get("arguments", {})
                    if isinstance(args, dict):
                        return _extract_answer_from_obj(args.get("ans") or args.get("final_answer"))

        if obj.get("raw_output"):
            return _extract_answer_from_obj(obj["raw_output"])

    return ""


def _extract_episode_answer(ep: dict[str, Any]) -> str:
    result = ep.get("result", {}) if isinstance(ep.get("result"), dict) else {}
    bundle = ep.get("bundle", {}) if isinstance(ep.get("bundle"), dict) else {}
    agent_outputs = bundle.get("agent_outputs", {}) if isinstance(bundle.get("agent_outputs"), dict) else {}

    candidates = [
        result,
        result.get("selected_by_conflict"),
        agent_outputs.get("pa"),
        agent_outputs.get("orc_merged"),
        agent_outputs.get("vra"),
    ]
    for cand in candidates:
        ans = _extract_answer_from_obj(cand)
        if ans:
            return _normalize_space(ans)
    return ""


def _flatten_pred_calls(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Preferred path: parse OEA-style agents_output preserving order.
    agents_output = bundle.get("agents_output", []) if isinstance(bundle, dict) else []
    if isinstance(agents_output, list):
        for msg in agents_output:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("from", "")).lower() != "gpt":
                continue

            parsed: dict[str, Any] = {}
            val = msg.get("value", "")
            if isinstance(val, dict):
                parsed = val
            else:
                parsed = _safe_load_json(str(val))

            actions = parsed.get("actions", []) if isinstance(parsed.get("actions"), list) else []
            for act in actions:
                if not isinstance(act, dict):
                    continue
                name = str(act.get("name", "")).strip()
                if not name:
                    continue
                n_name = _normalize_tool_name(name)
                if n_name in {"terminate", "plan"}:
                    continue
                out.append(
                    {
                        "agent": msg.get("agent", ""),
                        "name": name,
                        "arguments": act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {},
                        "output": {},
                        "latency_ms": None,
                    }
                )

    if out:
        return out

    # Fallback path: flatten raw agent_tool_calls from bundle.
    calls_by_agent = bundle.get("agent_tool_calls", {}) if isinstance(bundle, dict) else {}
    for agent_name in ("vra", "ga", "pa", "orc"):
        calls = calls_by_agent.get(agent_name, []) if isinstance(calls_by_agent, dict) else []
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            out.append(
                {
                    "agent": agent_name,
                    "name": call.get("tool", ""),
                    "arguments": call.get("args", {}) if isinstance(call.get("args"), dict) else {},
                    "output": call.get("output", {}),
                    "latency_ms": call.get("latency_ms"),
                }
            )
    return out


def _extract_oea_question(item: dict[str, Any]) -> str:
    turns = _extract_oea_human_turns(item)
    if not turns:
        return ""
    for turn in turns:
        if not _normalize_text(turn).startswith("observation"):
            return turn
    return turns[0]


def _clean_oea_human_turn(value: Any) -> str:
    text = _normalize_space(str(value)).replace("<AGENT_PROMPT>", "").strip()
    if "Question:" in text:
        text = _normalize_space(text.split("Question:", 1)[1])
    return _normalize_space(text)


def _extract_oea_human_turns(item: dict[str, Any]) -> list[str]:
    conv = item.get("conversation", []) if isinstance(item.get("conversation"), list) else []
    turns: list[str] = []
    for msg in conv:
        if not isinstance(msg, dict) or msg.get("from") != "human":
            continue
        clean = _clean_oea_human_turn(msg.get("value", ""))
        if clean:
            turns.append(clean)
    return turns


def _extract_gt_actions_from_oea(item: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    conv = item.get("conversation", []) if isinstance(item.get("conversation"), list) else []
    gt_calls: list[dict[str, Any]] = []
    gt_answer = ""

    for msg in conv:
        if not isinstance(msg, dict) or msg.get("from") != "gpt":
            continue
        parsed = _safe_load_json(str(msg.get("value", "")))
        actions = parsed.get("actions", []) if isinstance(parsed.get("actions"), list) else []
        for act in actions:
            if not isinstance(act, dict):
                continue
            name = str(act.get("name", "")).strip()
            args = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
            if not name:
                continue
            if _normalize_tool_name(name) == "terminate":
                if not gt_answer:
                    gt_answer = _normalize_space(str(args.get("ans") or args.get("final_answer") or ""))
            else:
                gt_calls.append({"name": name, "arguments": args})

    return gt_calls, gt_answer


def _extract_gt_steps_from_oea(item: dict[str, Any]) -> list[dict[str, Any]]:
    conv = item.get("conversation", []) if isinstance(item.get("conversation"), list) else []
    steps: list[dict[str, Any]] = []
    pending_human: str | None = None

    for msg in conv:
        if not isinstance(msg, dict):
            continue
        if msg.get("from") == "human":
            cleaned = _clean_oea_human_turn(msg.get("value", ""))
            pending_human = cleaned if cleaned else None
            continue

        if msg.get("from") != "gpt" or pending_human is None:
            continue

        parsed = _safe_load_json(str(msg.get("value", "")))
        actions = parsed.get("actions", []) if isinstance(parsed.get("actions"), list) else []
        if actions:
            for act in actions:
                if not isinstance(act, dict):
                    continue
                name = str(act.get("name", "")).strip()
                if not name:
                    continue
                args = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
                steps.append(
                    {
                        "human_input": pending_human,
                        "gt_action": {"name": name, "arguments": args},
                    }
                )
        pending_human = None

    return steps


def load_oea_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        gt_calls, gt_answer = _extract_gt_actions_from_oea(item)
        gt_steps = _extract_gt_steps_from_oea(item)
        human_turns = _extract_oea_human_turns(item)
        row = {
            "id": str(item.get("idx", i)),
            "question": _extract_oea_question(item),
            "human_turns": human_turns,
            "gt_steps": gt_steps,
            "gt_answer": gt_answer,
            "gt_tool_calls": gt_calls,
            "images": item.get("images", []),
            "dataset": "oea",
            "solvable": True,
            "is_routing": False,
        }
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _generic_tool_schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Call {name}",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        },
    }


def _resolve_openapi_schema(schema: Any, components: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    if depth > 8 or not isinstance(schema, dict):
        return {"type": "string"}

    if "$ref" in schema and isinstance(schema["$ref"], str):
        ref = schema["$ref"].split("/")[-1]
        target = components.get(ref, {}) if isinstance(components, dict) else {}
        return _resolve_openapi_schema(target, components, depth + 1)

    schema_type = schema.get("type")

    if schema_type == "object" or isinstance(schema.get("properties"), dict):
        props: dict[str, Any] = {}
        for k, v in (schema.get("properties") or {}).items():
            props[str(k)] = _resolve_openapi_schema(v, components, depth + 1)
        out: dict[str, Any] = {"type": "object", "properties": props}
        if isinstance(schema.get("required"), list):
            out["required"] = [str(x) for x in schema.get("required", [])]
        return out

    if schema_type == "array":
        return {
            "type": "array",
            "items": _resolve_openapi_schema(schema.get("items", {}), components, depth + 1),
        }

    out: dict[str, Any] = {"type": schema_type if isinstance(schema_type, str) else "string"}
    if isinstance(schema.get("enum"), list):
        out["enum"] = schema["enum"]
    return out


def _load_tool_schemas_from_server(tool_server: str) -> dict[str, dict[str, Any]]:
    url = f"{tool_server.rstrip('/')}/openapi.json"
    try:
        resp = httpx.get(url, timeout=30.0)
        resp.raise_for_status()
        spec = resp.json()
    except Exception:
        return {}

    tools: dict[str, dict[str, Any]] = {}
    paths = spec.get("paths", {}) if isinstance(spec.get("paths"), dict) else {}
    components = spec.get("components", {}) if isinstance(spec.get("components"), dict) else {}
    schemas = components.get("schemas", {}) if isinstance(components.get("schemas"), dict) else {}

    for route, methods in paths.items():
        if not isinstance(route, str) or not route.startswith("/tools/"):
            continue
        post = methods.get("post", {}) if isinstance(methods, dict) else {}
        name = route.split("/tools/", 1)[1]
        if _normalize_tool_name(name) in {"terminate", "plan"}:
            continue

        req = post.get("requestBody", {}) if isinstance(post.get("requestBody"), dict) else {}
        content = req.get("content", {}) if isinstance(req.get("content"), dict) else {}
        app_json = content.get("application/json", {}) if isinstance(content.get("application/json"), dict) else {}
        schema = app_json.get("schema", {}) if isinstance(app_json.get("schema"), dict) else {}
        params = _resolve_openapi_schema(schema, schemas)
        if params.get("type") != "object":
            params = {"type": "object", "properties": {}, "additionalProperties": True}

        tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": post.get("summary") or f"Call {name}",
                "parameters": params,
            },
        }

    return tools


def _extract_message_actions(message: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    thought = ""
    actions: list[dict[str, Any]] = []

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parsed = _safe_load_json(content)
        if parsed:
            t = parsed.get("thought")
            if isinstance(t, str):
                thought = _normalize_space(t)
            acts = parsed.get("actions", []) if isinstance(parsed.get("actions"), list) else []
            for act in acts:
                if not isinstance(act, dict):
                    continue
                name = str(act.get("name", "")).strip()
                if not name:
                    continue
                args = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
                actions.append({"name": name, "arguments": args})
        elif not thought:
            thought = _normalize_space(content)

    tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    parsed_tool_actions: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        args_raw = fn.get("arguments", {})
        if isinstance(args_raw, str):
            args = _safe_load_json(args_raw)
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        parsed_tool_actions.append({"name": name, "arguments": args})

    if parsed_tool_actions:
        actions = parsed_tool_actions

    return thought, actions


def _extract_terminate_answer(action: dict[str, Any], message: dict[str, Any]) -> str:
    name = _normalize_tool_name(str(action.get("name", "")))
    if name == "terminate":
        args = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
        ans = args.get("ans") or args.get("final_answer") or args.get("answer") or args.get("response")
        if ans not in (None, ""):
            return _normalize_space(str(ans))

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parsed = _safe_load_json(content)
        if parsed:
            return _normalize_space(_extract_answer_from_obj(parsed))
        return _normalize_space(content)
    return ""


def _normalize_args_for_tool_server(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return _TOOL_ARG_NORMALIZER._normalize_tool_arguments(tool_name, arguments)


def _execute_tool_call(
    tool_server: str,
    action: dict[str, Any],
) -> tuple[Any, str | None, dict[str, Any]]:
    tool_name = str(action.get("name", "")).strip()
    args = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
    normalized_args = _normalize_args_for_tool_server(tool_name, args)
    if not tool_name or _normalize_tool_name(tool_name) in {"terminate", "plan"}:
        return None, None, normalized_args

    url = f"{tool_server.rstrip('/')}/tools/{tool_name}"
    try:
        resp = httpx.post(url, json=normalized_args, timeout=90.0)
        resp.raise_for_status()
        return resp.json(), None, normalized_args
    except Exception as exc:
        return {"error": str(exc)}, str(exc), normalized_args


def _chat_single_step(
    *,
    base_url: str,
    model_id: str,
    messages: list[dict[str, Any]],
    tools_schema: list[dict[str, Any]],
    timeout_s: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str | None, float]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if tools_schema:
        payload["tools"] = tools_schema
        payload["tool_choice"] = "auto"

    t0 = datetime.utcnow()
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        choices = body.get("choices", []) if isinstance(body.get("choices"), list) else []
        msg = choices[0].get("message", {}) if choices else {}
        if not isinstance(msg, dict):
            msg = {"role": "assistant", "content": ""}
        dt_ms = (datetime.utcnow() - t0).total_seconds() * 1000.0
        return msg, None, dt_ms
    except Exception as exc:
        dt_ms = (datetime.utcnow() - t0).total_seconds() * 1000.0
        fallback = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "thought": f"LLM call failed: {exc}",
                    "actions": [{"name": "Terminate", "arguments": {"ans": "error"}}],
                }
            ),
        }
        return fallback, str(exc), dt_ms


def _thinkgeo_gt_match(generated_text: str, gt_answer: Any) -> bool:
    if not gt_answer:
        return False

    gen = _normalize_text(generated_text)

    if isinstance(gt_answer, list):
        terms = [_normalize_text(str(x)) for x in gt_answer if _normalize_text(str(x))]
        return any(term in gen for term in terms)

    if isinstance(gt_answer, dict):
        blacklist = gt_answer.get("blacklist")
        if isinstance(blacklist, list):
            for term in blacklist:
                t = _normalize_text(str(term))
                if t and t in gen:
                    return False

        whitelist = gt_answer.get("whitelist")
        if isinstance(whitelist, list):
            for group in whitelist:
                if not isinstance(group, list):
                    group = [group]
                opts = [_normalize_text(str(item)) for item in group if _normalize_text(str(item))]
                if opts and not any(opt in gen for opt in opts):
                    return False
            return True

    return False


def _is_routing_task(question: str, tool_names: list[str], gt_calls: list[dict[str, Any]]) -> bool:
    q = _normalize_text(question)
    if any(hint in q for hint in ROUTING_HINTS):
        return True

    merged = [*tool_names, *(call.get("name", "") for call in gt_calls)]
    for name in merged:
        n = _normalize_tool_name(str(name))
        if "route" in n or "evacuation" in n or "path" in n:
            return True
    return False


def load_thinkgeo_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []

    rows: list[dict[str, Any]] = []
    for key, item in data.items():
        if not isinstance(item, dict):
            continue

        dialogs = item.get("dialogs", []) if isinstance(item.get("dialogs"), list) else []
        question = ""
        gt_calls: list[dict[str, Any]] = []

        for turn in dialogs:
            if not isinstance(turn, dict):
                continue
            if turn.get("role") == "user" and not question:
                question = _normalize_space(str(turn.get("content", "")))

            if turn.get("role") == "assistant":
                tcs = turn.get("tool_calls", []) if isinstance(turn.get("tool_calls"), list) else []
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                    name = str(fn.get("name", "")).strip()
                    args = fn.get("arguments", {}) if isinstance(fn.get("arguments"), dict) else {}
                    if name:
                        gt_calls.append({"name": name, "arguments": args})

        tools = item.get("tools", []) if isinstance(item.get("tools"), list) else []
        tool_names = [str(t.get("name", "")) for t in tools if isinstance(t, dict)]
        gt_answer = item.get("gt_answer")
        solvable = gt_answer is not None

        rows.append(
            {
                "id": str(key),
                "question": question,
                "gt_answer": gt_answer,
                "gt_tool_calls": gt_calls,
                "tool_names": tool_names,
                "dataset": "thinkgeo",
                "solvable": solvable,
                "is_routing": _is_routing_task(question, tool_names, gt_calls),
            }
        )
        if limit > 0 and len(rows) >= limit:
            break

    return rows


def _score_tool_and_args(
    gt_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    tool_correct = 0
    tool_total = len(gt_calls)
    argn_correct = 0
    argn_total = 0
    argv_correct = 0
    argv_total = 0

    for idx, gt_call in enumerate(gt_calls):
        gt_name = _normalize_tool_name(str(gt_call.get("name", "")))
        gt_args_raw = gt_call.get("arguments", {}) if isinstance(gt_call.get("arguments"), dict) else {}
        gt_args = {_normalize_arg_name(str(k)): v for k, v in gt_args_raw.items()}

        pred_call = pred_calls[idx] if idx < len(pred_calls) else None
        pred_name = _normalize_tool_name(str(pred_call.get("name", ""))) if pred_call else ""
        pred_args_raw = pred_call.get("arguments", {}) if pred_call and isinstance(pred_call.get("arguments"), dict) else {}
        pred_args = {_normalize_arg_name(str(k)): v for k, v in pred_args_raw.items()}

        if gt_name == pred_name:
            tool_correct += 1

        for arg_name, gt_val in gt_args.items():
            argn_total += 1
            argv_total += 1
            if arg_name in pred_args:
                argn_correct += 1
                if _value_match(gt_val, pred_args[arg_name]):
                    argv_correct += 1

    return {
        "tool_correct": tool_correct,
        "tool_total": tool_total,
        "argn_correct": argn_correct,
        "argn_total": argn_total,
        "argv_correct": argv_correct,
        "argv_total": argv_total,
    }


def _oea_answer_correct(pred_answer: str, gt_answer: str) -> bool:
    p = _normalize_space(pred_answer)
    g = _normalize_space(gt_answer)
    if not p or not g:
        return False

    g_nums = _extract_numbers(g)
    p_nums = _extract_numbers(p)
    if g_nums:
        if not p_nums:
            return False
        for g_num in g_nums:
            tol = max(abs(g_num) * 0.05, 1e-6)
            if not any(abs(p_num - g_num) <= tol for p_num in p_nums):
                return False

    rouge = _rouge_l_f1(p, g)
    norm_g = _normalize_text(g)
    norm_p = _normalize_text(p)
    return norm_g in norm_p or rouge >= 0.45


def _sample_route_stats(pred_calls: list[dict[str, Any]]) -> dict[str, Any]:
    safe_segments = 0
    total_segments = 0
    rss_values: list[float] = []
    conflict_checks = 0
    replans_safe = 0
    pending_conflict = False

    for call in pred_calls:
        if _normalize_tool_name(str(call.get("name", ""))) != "evacuationrouteplanner":
            continue

        args = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
        mode = str(args.get("mode", "evacuation")).lower()
        output = call.get("output", {}) if isinstance(call.get("output"), dict) else {}
        routes = output.get("routes", []) if isinstance(output.get("routes"), list) else []

        if mode == "conflict_check":
            conflict_checks += 1
            pending_conflict = True

        local_best_rss: float | None = None
        for route in routes:
            if not isinstance(route, dict):
                continue
            rss_val = route.get("rss")
            if isinstance(rss_val, (int, float)):
                rss = float(rss_val)
                rss_values.append(rss)
                local_best_rss = rss if local_best_rss is None else max(local_best_rss, rss)

            segments = route.get("segments")
            if isinstance(segments, list):
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    damage_score = seg.get("damage_score")
                    if isinstance(damage_score, (int, float)):
                        total_segments += 1
                        if float(damage_score) < 0.6:
                            safe_segments += 1

        if pending_conflict and mode == "evacuation":
            if local_best_rss is not None and local_best_rss >= 1.0:
                replans_safe += 1
            pending_conflict = False

    return {
        "safe_segments": safe_segments,
        "total_segments": total_segments,
        "rss_values": rss_values,
        "conflict_checks": conflict_checks,
        "replans_safe": replans_safe,
    }


def _subtasks_complete(bundle: dict[str, Any]) -> bool:
    calls_by_agent = bundle.get("agent_tool_calls", {}) if isinstance(bundle.get("agent_tool_calls"), dict) else {}
    routed = bundle.get("executed_agent_order") or bundle.get("agent_call_order") or []
    if not isinstance(routed, list) or not routed:
        routed = ["vra", "ga", "pa"]
    active_agents = [a for a in routed if a in {"vra", "ga", "pa"}]
    if not active_agents:
        active_agents = ["vra", "ga", "pa"]
    return all(
        isinstance(calls_by_agent.get(agent), list) and len(calls_by_agent.get(agent)) > 0
        for agent in active_agents
    )


def _build_task(
    row: dict[str, Any],
    dataset: str,
    idx: int,
    *,
    include_oea_human_inputs: bool = False,
    auto_route_oea: bool = True,
) -> dict[str, Any]:
    task = {
        "objective": row.get("question") or "",
        "scene_id": f"{dataset}-scene-{row.get('id', idx)}",
        "region": dataset,
        "benchmark": dataset,
        "dataset_row_id": row.get("id", str(idx)),
        "source": dataset,
    }
    if dataset == "oea":
        if include_oea_human_inputs and isinstance(row.get("human_turns"), list):
            task["human_inputs"] = [str(x) for x in row.get("human_turns", [])]
        if auto_route_oea:
            task["agent_sequence"] = _infer_oea_agent_sequence(str(row.get("question", "")))
    return task


def _infer_oea_agent_sequence(question: str) -> list[str]:
    q = _normalize_text(question)

    visual_terms = {
        "image",
        "aerial",
        "pixel",
        "bbox",
        "detect",
        "helicopter",
        "tree",
        "canopy",
        "ndbi",
        "index",
        "raster",
        "geotiff",
    }
    geo_terms = {
        "closest",
        "distance",
        "within",
        "boundary",
        "park",
        "market",
        "atm",
        "supermarket",
        "restaurant",
        "police",
        "fire",
        "mall",
        "parking",
        "radius",
        "province",
        "city",
        "university",
    }
    calc_terms = {
        "calculate",
        "combined area",
        "convert",
        "rounded",
        "diameter",
        "meters",
        "assess",
    }

    has_visual = any(t in q for t in visual_terms)
    has_geo = any(t in q for t in geo_terms)
    needs_calc = any(t in q for t in calc_terms)

    seq: list[str] = []
    if has_visual:
        seq.append("vra")
    if has_geo:
        seq.append("ga")
    if needs_calc:
        seq.append("pa")

    if not seq:
        seq = ["ga", "pa"]
    # De-duplicate while preserving order.
    out: list[str] = []
    for s in seq:
        if s not in out:
            out.append(s)
    return out


def _score_oea_sample(row: dict[str, Any], ep: dict[str, Any]) -> dict[str, Any]:
    bundle = ep.get("bundle", {}) if isinstance(ep.get("bundle"), dict) else {}
    agent_outputs = bundle.get("agent_outputs", {}) if isinstance(bundle.get("agent_outputs"), dict) else {}
    pred_calls = _flatten_pred_calls(bundle)
    pred_answer = _extract_episode_answer(ep)
    gt_answer = str(row.get("gt_answer", ""))
    gt_calls = row.get("gt_tool_calls", []) if isinstance(row.get("gt_tool_calls"), list) else []
    ep_metrics = ep.get("metrics", {}) if isinstance(ep.get("metrics"), dict) else {}

    call_scores = _score_tool_and_args(gt_calls, pred_calls)
    rouge = _rouge_l_f1(pred_answer, gt_answer)
    answer_ok = _oea_answer_correct(pred_answer, gt_answer)

    route_stats = _sample_route_stats(pred_calls)
    sample_tls = TLS(len(gt_calls), len(pred_calls)) if len(pred_calls) > 0 else 0.0
    ccq = float(ep_metrics.get("ccq", 0.0))
    plan_accuracy = float(ep_metrics.get("plan_accuracy", 0.0))

    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "human_inputs": row.get("human_turns", []),
        "ok": True,
        "answer_correct": answer_ok,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "rouge_l_f1": rouge,
        "tool_correct": call_scores["tool_correct"],
        "tool_total": call_scores["tool_total"],
        "argn_correct": call_scores["argn_correct"],
        "argn_total": call_scores["argn_total"],
        "argv_correct": call_scores["argv_correct"],
        "argv_total": call_scores["argv_total"],
        "gt_tool_calls": len(gt_calls),
        "pred_tool_calls": len(pred_calls),
        "sample_tls": sample_tls,
        "subtasks_complete": _subtasks_complete(bundle),
        "mgar_hit": _subtasks_complete(bundle) and answer_ok,
        "plan_accuracy": plan_accuracy,
        "plan_followed": bool(ep_metrics.get("plan_followed", False)),
        "plan_loop_count": int(ep_metrics.get("plan_loop_count", 0)),
        "route_stats": route_stats,
        "ccq": ccq,
        "run_dir": ep.get("run_dir"),
        "episode_metrics": ep_metrics,
        "result": ep.get("result", {}),
        "metrics": ep_metrics,
        "agents_output": bundle.get("agents_output", []),
        "agent_outputs": agent_outputs,
        "agent_tool_calls_by_agent": bundle.get("agent_tool_calls", {}),
        "agent_traces": bundle.get("agent_traces", {}),
        "bundle_path": f"{ep.get('run_dir')}/bundle.json" if ep.get("run_dir") else None,
        "flow_log_path": bundle.get("flow_log_path"),
        "verbose_log_path": bundle.get("verbose_agents_output_path"),
        "agent_call_order": bundle.get("executed_agent_order") or bundle.get("agent_call_order", []),
    }


def _run_oea_replay_case(
    *,
    row: dict[str, Any],
    tool_server: str,
    execute_tools: bool,
) -> dict[str, Any]:
    gt_steps = row.get("gt_steps", []) if isinstance(row.get("gt_steps"), list) else []
    gt_calls = row.get("gt_tool_calls", []) if isinstance(row.get("gt_tool_calls"), list) else []
    gt_answer = str(row.get("gt_answer", ""))

    step_records: list[dict[str, Any]] = []
    reasoning_trace: list[str] = []
    pred_calls: list[dict[str, Any]] = []
    case_errors: list[str] = []
    pred_answer = ""

    for i, step in enumerate(gt_steps):
        turn = i + 1
        human_input = str(step.get("human_input", ""))
        gt_action = step.get("gt_action", {}) if isinstance(step.get("gt_action"), dict) else {}
        gt_name = str(gt_action.get("name", "")).strip()
        gt_args = gt_action.get("arguments", {}) if isinstance(gt_action.get("arguments"), dict) else {}
        gt_norm = _normalize_tool_name(gt_name)

        observation_text = ""
        if i + 1 < len(gt_steps):
            observation_text = _extract_observation_payload(str(gt_steps[i + 1].get("human_input", "")))

        reasoning = _replay_reasoning_for(gt_name)
        if reasoning:
            reasoning_trace.append(reasoning)

        tool_output: Any = None
        tool_error: str | None = None
        normalized_args = _normalize_args_for_tool_server(gt_name, gt_args)

        if gt_norm not in {"terminate", "plan"}:
            if execute_tools:
                tool_output, tool_error, normalized_args = _execute_tool_call(
                    tool_server,
                    {"name": gt_name, "arguments": gt_args},
                )
            else:
                tool_output = {
                    "source": "ground_truth_observation",
                    "observation": observation_text,
                }

            pred_calls.append(
                {
                    "name": gt_name,
                    "arguments": gt_args,
                    "output": tool_output if isinstance(tool_output, dict) else {},
                }
            )

        if gt_norm == "terminate" and not pred_answer:
            pred_answer = _extract_terminate_answer({"name": gt_name, "arguments": gt_args}, {"content": ""})

        if tool_error:
            case_errors.append(f"turn {turn} tool {gt_name}: {tool_error}")

        step_arg_scores = {
            "tool_correct": 0,
            "tool_total": 0,
            "argn_correct": 0,
            "argn_total": 0,
            "argv_correct": 0,
            "argv_total": 0,
        }
        if gt_norm not in {"terminate", "plan"}:
            step_arg_scores = _score_tool_and_args(
                [{"name": gt_name, "arguments": gt_args}],
                [{"name": gt_name, "arguments": gt_args}],
            )

        step_records.append(
            {
                "turn": turn,
                "human_input": human_input,
                "observation_after_action": observation_text,
                "gt_action": {"name": gt_name, "arguments": gt_args},
                "pred_action": {"name": gt_name, "arguments": gt_args},
                "name_match": True,
                "gt_is_terminate": gt_norm == "terminate",
                "argn_correct": step_arg_scores["argn_correct"],
                "argn_total": step_arg_scores["argn_total"],
                "argv_correct": step_arg_scores["argv_correct"],
                "argv_total": step_arg_scores["argv_total"],
                "thought": reasoning,
                "assistant_raw": json.dumps(
                    {"thought": reasoning, "actions": [{"name": gt_name, "arguments": gt_args}]},
                    ensure_ascii=False,
                ),
                "llm_latency_ms": 0.0,
                "normalized_arguments": normalized_args,
                "tool_result": tool_output,
                "llm_error": None,
                "tool_error": tool_error,
                "mode": "ground_truth_replay",
            }
        )

    if not pred_answer:
        pred_answer = gt_answer

    call_scores = _score_tool_and_args(gt_calls, pred_calls)
    rouge = _rouge_l_f1(pred_answer, gt_answer)
    answer_ok = _oea_answer_correct(pred_answer, gt_answer)
    sample_tls = TLS(len(gt_calls), len(pred_calls)) if pred_calls else 0.0
    non_term_steps = [s for s in step_records if not s.get("gt_is_terminate")]
    exact_step_name_acc = 1.0 if non_term_steps else 0.0
    subtasks_complete = call_scores["tool_total"] > 0 and call_scores["tool_correct"] == call_scores["tool_total"]

    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "human_inputs": row.get("human_turns", []),
        "ok": True,
        "answer_correct": answer_ok,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "rouge_l_f1": rouge,
        "tool_correct": call_scores["tool_correct"],
        "tool_total": call_scores["tool_total"],
        "argn_correct": call_scores["argn_correct"],
        "argn_total": call_scores["argn_total"],
        "argv_correct": call_scores["argv_correct"],
        "argv_total": call_scores["argv_total"],
        "gt_tool_calls": len(gt_calls),
        "pred_tool_calls": len(pred_calls),
        "sample_tls": sample_tls,
        "subtasks_complete": subtasks_complete,
        "mgar_hit": subtasks_complete and answer_ok,
        "plan_accuracy": exact_step_name_acc,
        "plan_followed": True,
        "plan_loop_count": 0,
        "route_stats": {
            "safe_segments": 0,
            "total_segments": 0,
            "rss_values": [],
            "conflict_checks": 0,
            "replans_safe": 0,
        },
        "ccq": 1.0,
        "run_dir": None,
        "episode_metrics": {
            "execution_mode": "oea_ground_truth_replay",
            "step_name_accuracy": exact_step_name_acc,
            "tool_total": call_scores["tool_total"],
            "tool_correct": call_scores["tool_correct"],
            "argn_total": call_scores["argn_total"],
            "argn_correct": call_scores["argn_correct"],
            "argv_total": call_scores["argv_total"],
            "argv_correct": call_scores["argv_correct"],
            "answer_correct": answer_ok,
        },
        "result": {
            "final_answer": pred_answer,
            "errors": case_errors,
        },
        "metrics": {
            "execution_mode": "oea_ground_truth_replay",
            "step_name_accuracy": exact_step_name_acc,
        },
        "agents_output": [],
        "bundle_path": None,
        "flow_log_path": None,
        "verbose_log_path": None,
        "agent_call_order": [],
        "turnwise_steps": step_records,
        "reasoning_trace": reasoning_trace,
        "errors": case_errors,
        "ground_truth_replay": True,
    }


def _run_oea_replay_dataset(
    *,
    rows: list[dict[str, Any]],
    tool_server: str,
    execute_tools: bool,
    out_dir: Path,
    write_checkpoints: bool,
) -> dict[str, Any]:
    sample_results: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        rec: dict[str, Any] = {
            "id": row.get("id"),
            "index": i,
            "question": row.get("question"),
            "dataset": "oea",
        }
        try:
            scored = _run_oea_replay_case(
                row=row,
                tool_server=tool_server,
                execute_tools=execute_tools,
            )
            rec.update(scored)
        except Exception as exc:
            rec.update(
                {
                    "ok": False,
                    "error": str(exc),
                    "gt_tool_calls": len(row.get("gt_tool_calls", [])),
                    "pred_tool_calls": 0,
                    "turnwise_steps": [],
                }
            )

        sample_results.append(rec)
        progress = f"[oea-replay] {i + 1}/{len(rows)}"
        status = "ok" if rec.get("ok") else "error"
        print(f"{progress} {status} id={rec.get('id')}")

        if write_checkpoints:
            checkpoint = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "dataset": "oea",
                "mode": "ground_truth_replay",
                "n_rows": len(rows),
                "sample_results": sample_results,
            }
            (out_dir / "oea_samples.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    ok_samples = [s for s in sample_results if s.get("ok")]
    failed = len(sample_results) - len(ok_samples)
    agg = _aggregate_oea(ok_samples)
    return {
        "name": "oea",
        "n_total": len(sample_results),
        "n_ok": len(ok_samples),
        "n_failed": failed,
        "aggregate": agg,
        "sample_results": sample_results,
    }


def _run_oea_turnwise_case(
    *,
    row: dict[str, Any],
    model_id: str,
    base_url: str,
    tool_server: str,
    tools_schema: list[dict[str, Any]],
    llm_timeout: float,
    llm_max_tokens: int,
) -> dict[str, Any]:
    gt_steps = row.get("gt_steps", []) if isinstance(row.get("gt_steps"), list) else []
    gt_calls = row.get("gt_tool_calls", []) if isinstance(row.get("gt_tool_calls"), list) else []
    gt_answer = str(row.get("gt_answer", ""))

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are solving an OpenEarth task in strict step mode. "
                "For each user turn, output exactly one next action in JSON: "
                "{\"thought\": string, \"actions\": [{\"name\": string, \"arguments\": object}]}. "
                "Use only one action per step. When the answer is final, output Terminate with argument ans."
            ),
        }
    ]

    step_records: list[dict[str, Any]] = []
    pred_calls: list[dict[str, Any]] = []
    reasoning_trace: list[str] = []
    case_errors: list[str] = []
    pred_answer = ""

    for idx, step in enumerate(gt_steps, start=1):
        human_input = str(step.get("human_input", ""))
        gt_action = step.get("gt_action", {}) if isinstance(step.get("gt_action"), dict) else {}
        gt_name = str(gt_action.get("name", "")).strip()
        gt_args = gt_action.get("arguments", {}) if isinstance(gt_action.get("arguments"), dict) else {}
        gt_norm = _normalize_tool_name(gt_name)

        messages.append({"role": "user", "content": human_input})
        msg, llm_error, llm_latency_ms = _chat_single_step(
            base_url=base_url,
            model_id=model_id,
            messages=messages,
            tools_schema=tools_schema,
            timeout_s=llm_timeout,
            max_tokens=llm_max_tokens,
        )

        thought, actions = _extract_message_actions(msg)
        if thought:
            reasoning_trace.append(thought)

        pred_action = actions[0] if actions else {"name": "", "arguments": {}}
        pred_name = str(pred_action.get("name", "")).strip()
        pred_args = pred_action.get("arguments", {}) if isinstance(pred_action.get("arguments"), dict) else {}
        pred_norm = _normalize_tool_name(pred_name)

        tool_output: Any = None
        tool_error: str | None = None
        normalized_args: dict[str, Any] = pred_args
        if pred_name and pred_norm not in {"terminate", "plan"}:
            tool_output, tool_error, normalized_args = _execute_tool_call(tool_server, pred_action)
            pred_calls.append(
                {
                    "name": pred_name,
                    "arguments": pred_args,
                    "output": tool_output if isinstance(tool_output, dict) else {},
                }
            )

        if pred_norm == "terminate" and not pred_answer:
            pred_answer = _extract_terminate_answer(pred_action, msg)

        step_arg_scores = {
            "tool_correct": 0,
            "tool_total": 0,
            "argn_correct": 0,
            "argn_total": 0,
            "argv_correct": 0,
            "argv_total": 0,
        }
        if gt_norm not in {"terminate", "plan"}:
            step_arg_scores = _score_tool_and_args(
                [{"name": gt_name, "arguments": gt_args}],
                [{"name": pred_name, "arguments": pred_args}],
            )

        step_records.append(
            {
                "turn": idx,
                "human_input": human_input,
                "gt_action": {"name": gt_name, "arguments": gt_args},
                "pred_action": {"name": pred_name, "arguments": pred_args},
                "name_match": pred_norm == gt_norm,
                "gt_is_terminate": gt_norm == "terminate",
                "argn_correct": step_arg_scores["argn_correct"],
                "argn_total": step_arg_scores["argn_total"],
                "argv_correct": step_arg_scores["argv_correct"],
                "argv_total": step_arg_scores["argv_total"],
                "thought": thought,
                "assistant_raw": msg.get("content", "") if isinstance(msg.get("content"), str) else "",
                "llm_latency_ms": round(float(llm_latency_ms), 2),
                "normalized_arguments": normalized_args,
                "tool_result": tool_output,
                "llm_error": llm_error,
                "tool_error": tool_error,
            }
        )

        if llm_error:
            case_errors.append(f"turn {idx}: {llm_error}")
        if tool_error:
            case_errors.append(f"turn {idx} tool {pred_name}: {tool_error}")

        assistant_payload: dict[str, Any] = {"thought": thought or "", "actions": []}
        if pred_name:
            assistant_payload["actions"].append({"name": pred_name, "arguments": pred_args})
        messages.append({"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)})

    if not pred_answer and step_records:
        pred_answer = _extract_terminate_answer(
            step_records[-1].get("pred_action", {}),
            {"content": step_records[-1].get("assistant_raw", "")},
        )

    call_scores = _score_tool_and_args(gt_calls, pred_calls)
    rouge = _rouge_l_f1(pred_answer, gt_answer)
    answer_ok = _oea_answer_correct(pred_answer, gt_answer)
    sample_tls = TLS(len(gt_calls), len(pred_calls)) if pred_calls else 0.0

    non_term_steps = [s for s in step_records if not s.get("gt_is_terminate")]
    exact_step_name_acc = (
        sum(1 for s in non_term_steps if s.get("name_match")) / len(non_term_steps)
        if non_term_steps
        else 0.0
    )
    subtasks_complete = call_scores["tool_total"] > 0 and call_scores["tool_correct"] == call_scores["tool_total"]

    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "human_inputs": row.get("human_turns", []),
        "ok": True,
        "answer_correct": answer_ok,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "rouge_l_f1": rouge,
        "tool_correct": call_scores["tool_correct"],
        "tool_total": call_scores["tool_total"],
        "argn_correct": call_scores["argn_correct"],
        "argn_total": call_scores["argn_total"],
        "argv_correct": call_scores["argv_correct"],
        "argv_total": call_scores["argv_total"],
        "gt_tool_calls": len(gt_calls),
        "pred_tool_calls": len(pred_calls),
        "sample_tls": sample_tls,
        "subtasks_complete": subtasks_complete,
        "mgar_hit": subtasks_complete and answer_ok,
        "plan_accuracy": exact_step_name_acc,
        "plan_followed": bool(non_term_steps) and exact_step_name_acc == 1.0,
        "plan_loop_count": 0,
        "route_stats": {
            "safe_segments": 0,
            "total_segments": 0,
            "rss_values": [],
            "conflict_checks": 0,
            "replans_safe": 0,
        },
        "ccq": 1.0,
        "run_dir": None,
        "episode_metrics": {
            "execution_mode": "oea_turnwise",
            "step_name_accuracy": exact_step_name_acc,
            "tool_total": call_scores["tool_total"],
            "tool_correct": call_scores["tool_correct"],
            "argn_total": call_scores["argn_total"],
            "argn_correct": call_scores["argn_correct"],
            "argv_total": call_scores["argv_total"],
            "argv_correct": call_scores["argv_correct"],
            "answer_correct": answer_ok,
        },
        "result": {
            "final_answer": pred_answer,
            "errors": case_errors,
        },
        "metrics": {
            "execution_mode": "oea_turnwise",
            "step_name_accuracy": exact_step_name_acc,
        },
        "agents_output": [],
        "bundle_path": None,
        "flow_log_path": None,
        "verbose_log_path": None,
        "agent_call_order": [],
        "turnwise_steps": step_records,
        "reasoning_trace": reasoning_trace,
        "errors": case_errors,
    }


def _run_oea_turnwise_dataset(
    *,
    rows: list[dict[str, Any]],
    model_id: str,
    base_url: str,
    tool_server: str,
    llm_timeout: float,
    llm_max_tokens: int,
    out_dir: Path,
    write_checkpoints: bool,
) -> dict[str, Any]:
    sample_results: list[dict[str, Any]] = []

    server_tool_schemas = _load_tool_schemas_from_server(tool_server)
    server_tool_by_norm = {
        _normalize_tool_name(name): schema
        for name, schema in server_tool_schemas.items()
    }

    oea_tool_names: dict[str, str] = {}
    for row in rows:
        calls = row.get("gt_tool_calls", []) if isinstance(row.get("gt_tool_calls"), list) else []
        for call in calls:
            if not isinstance(call, dict):
                continue
            raw_name = str(call.get("name", "")).strip()
            if not raw_name:
                continue
            norm = _normalize_tool_name(raw_name)
            if norm in {"terminate", "plan"}:
                continue
            oea_tool_names.setdefault(norm, raw_name)

    tools_schema: list[dict[str, Any]] = []
    for norm_name, raw_name in sorted(oea_tool_names.items(), key=lambda kv: kv[0]):
        tools_schema.append(server_tool_by_norm.get(norm_name, _generic_tool_schema(raw_name)))

    for i, row in enumerate(rows):
        rec: dict[str, Any] = {
            "id": row.get("id"),
            "index": i,
            "question": row.get("question"),
            "dataset": "oea",
        }
        try:
            scored = _run_oea_turnwise_case(
                row=row,
                model_id=model_id,
                base_url=base_url,
                tool_server=tool_server,
                tools_schema=tools_schema,
                llm_timeout=llm_timeout,
                llm_max_tokens=llm_max_tokens,
            )
            rec.update(scored)
        except Exception as exc:
            rec.update(
                {
                    "ok": False,
                    "error": str(exc),
                    "gt_tool_calls": len(row.get("gt_tool_calls", [])),
                    "pred_tool_calls": 0,
                    "turnwise_steps": [],
                }
            )

        sample_results.append(rec)
        progress = f"[oea-turnwise] {i + 1}/{len(rows)}"
        status = "ok" if rec.get("ok") else "error"
        print(f"{progress} {status} id={rec.get('id')}")

        if write_checkpoints:
            checkpoint = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "dataset": "oea",
                "mode": "turnwise",
                "n_rows": len(rows),
                "sample_results": sample_results,
            }
            (out_dir / "oea_samples.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    ok_samples = [s for s in sample_results if s.get("ok")]
    failed = len(sample_results) - len(ok_samples)
    agg = _aggregate_oea(ok_samples)
    return {
        "name": "oea",
        "n_total": len(sample_results),
        "n_ok": len(ok_samples),
        "n_failed": failed,
        "aggregate": agg,
        "sample_results": sample_results,
    }


def _score_thinkgeo_sample(row: dict[str, Any], ep: dict[str, Any]) -> dict[str, Any]:
    bundle = ep.get("bundle", {}) if isinstance(ep.get("bundle"), dict) else {}
    agent_outputs = bundle.get("agent_outputs", {}) if isinstance(bundle.get("agent_outputs"), dict) else {}
    pred_calls = _flatten_pred_calls(bundle)
    pred_answer = _extract_episode_answer(ep)
    gt_answer = row.get("gt_answer")
    gt_calls = row.get("gt_tool_calls", []) if isinstance(row.get("gt_tool_calls"), list) else []
    ep_metrics = ep.get("metrics", {}) if isinstance(ep.get("metrics"), dict) else {}

    solvable = bool(row.get("solvable", False))
    gt_match = _thinkgeo_gt_match(pred_answer, gt_answer) if solvable else False
    refused = any(term in _normalize_text(pred_answer) for term in REFUSAL_TERMS)

    tsr_hit = solvable and gt_match
    hrr_hit = (not solvable) and refused
    answer_ok = tsr_hit or hrr_hit

    route_stats = _sample_route_stats(pred_calls)
    sample_tls = TLS(len(gt_calls), len(pred_calls)) if len(pred_calls) > 0 else 0.0
    ccq = float(ep_metrics.get("ccq", 0.0))
    plan_accuracy = float(ep_metrics.get("plan_accuracy", 0.0))

    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "ok": True,
        "solvable": solvable,
        "is_routing": bool(row.get("is_routing", False)),
        "tsr_hit": tsr_hit,
        "hrr_hit": hrr_hit,
        "answer_correct": answer_ok,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "gt_tool_calls": len(gt_calls),
        "pred_tool_calls": len(pred_calls),
        "sample_tls": sample_tls,
        "subtasks_complete": _subtasks_complete(bundle),
        "mgar_hit": _subtasks_complete(bundle) and answer_ok,
        "plan_accuracy": plan_accuracy,
        "plan_followed": bool(ep_metrics.get("plan_followed", False)),
        "plan_loop_count": int(ep_metrics.get("plan_loop_count", 0)),
        "route_stats": route_stats,
        "ccq": ccq,
        "run_dir": ep.get("run_dir"),
        "episode_metrics": ep_metrics,
        "result": ep.get("result", {}),
        "metrics": ep_metrics,
        "agents_output": bundle.get("agents_output", []),
        "agent_outputs": agent_outputs,
        "agent_tool_calls_by_agent": bundle.get("agent_tool_calls", {}),
        "agent_traces": bundle.get("agent_traces", {}),
        "bundle_path": f"{ep.get('run_dir')}/bundle.json" if ep.get("run_dir") else None,
        "flow_log_path": bundle.get("flow_log_path"),
        "verbose_log_path": bundle.get("verbose_agents_output_path"),
        "agent_call_order": bundle.get("executed_agent_order") or bundle.get("agent_call_order", []),
    }


def _aggregate_oea(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    inst_correct = sum(1 for s in samples if s.get("answer_correct"))
    tool_correct = sum(int(s.get("tool_correct", 0)) for s in samples)
    tool_total = sum(int(s.get("tool_total", 0)) for s in samples)
    argn_correct = sum(int(s.get("argn_correct", 0)) for s in samples)
    argn_total = sum(int(s.get("argn_total", 0)) for s in samples)
    argv_correct = sum(int(s.get("argv_correct", 0)) for s in samples)
    argv_total = sum(int(s.get("argv_total", 0)) for s in samples)

    rouge_vals = [float(s.get("rouge_l_f1", 0.0)) for s in samples]
    avg_rouge = sum(rouge_vals) / len(rouge_vals) if rouge_vals else 0.0

    gt_tool_calls = sum(int(s.get("gt_tool_calls", 0)) for s in samples)
    pred_tool_calls = sum(int(s.get("pred_tool_calls", 0)) for s in samples)
    mgar_hits = sum(1 for s in samples if s.get("mgar_hit"))
    ccq_vals = [float(s.get("ccq", 0.0)) for s in samples]
    avg_ccq = sum(ccq_vals) / len(ccq_vals) if ccq_vals else 0.0
    plan_vals = [float(s.get("plan_accuracy", 0.0)) for s in samples]
    avg_plan_accuracy = sum(plan_vals) / len(plan_vals) if plan_vals else 0.0
    total_plan_loops = sum(int(s.get("plan_loop_count", 0)) for s in samples)

    safe_segments = sum(int(s.get("route_stats", {}).get("safe_segments", 0)) for s in samples)
    total_segments = sum(int(s.get("route_stats", {}).get("total_segments", 0)) for s in samples)
    rss_vals: list[float] = []
    for s in samples:
        rss_vals.extend([float(v) for v in s.get("route_stats", {}).get("rss_values", []) if isinstance(v, (int, float))])

    if total_segments > 0:
        rss_metric = RSS(safe_segments, total_segments)
        rss_mode = "segment_ratio"
    elif rss_vals:
        rss_metric = sum(max(0.0, min(1.0, v)) for v in rss_vals) / len(rss_vals) * 100.0
        rss_mode = "route_rss_proxy"
    else:
        rss_metric = None
        rss_mode = "unavailable"

    return {
        "benchmark_metrics": {
            "Inst": Inst(inst_correct, total),
            "Tool": Tool(tool_correct, tool_total),
            "ArgN": ArgN(argn_correct, argn_total),
            "ArgV": ArgV(argv_correct, argv_total),
            "Summ": Summ(avg_rouge),
        },
        "novel_metrics": {
            "MGAR": MGAR(mgar_hits, total),
            "TLS": TLS(gt_tool_calls, pred_tool_calls) if pred_tool_calls > 0 else 0.0,
            "RSS": rss_metric,
            "RSS_mode": rss_mode,
            "CCQ": avg_ccq,
            "plan_accuracy": avg_plan_accuracy,
            "plan_loops_total": total_plan_loops,
            "CRR": None,
            "MTCS": None,
            "DDF1": None,
            "CDF1": None,
            "ReSR": None,
        },
        "counters": {
            "total": total,
            "inst_correct": inst_correct,
            "tool_correct": tool_correct,
            "tool_total": tool_total,
            "argn_correct": argn_correct,
            "argn_total": argn_total,
            "argv_correct": argv_correct,
            "argv_total": argv_total,
            "gt_tool_calls": gt_tool_calls,
            "pred_tool_calls": pred_tool_calls,
        },
    }


def _aggregate_thinkgeo(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    solvable_total = sum(1 for s in samples if s.get("solvable"))
    unsolvable_total = total - solvable_total

    tsr_correct = sum(1 for s in samples if s.get("tsr_hit"))
    hrr_correct = sum(1 for s in samples if s.get("hrr_hit"))

    routing_samples = [s for s in samples if s.get("solvable") and s.get("is_routing")]
    simple_samples = [s for s in samples if s.get("solvable") and not s.get("is_routing")]
    routing_correct = sum(1 for s in routing_samples if s.get("tsr_hit"))
    simple_correct = sum(1 for s in simple_samples if s.get("tsr_hit"))

    gt_tool_calls = sum(int(s.get("gt_tool_calls", 0)) for s in samples)
    pred_tool_calls = sum(int(s.get("pred_tool_calls", 0)) for s in samples)
    mgar_hits = sum(1 for s in samples if s.get("mgar_hit"))
    ccq_vals = [float(s.get("ccq", 0.0)) for s in samples]
    avg_ccq = sum(ccq_vals) / len(ccq_vals) if ccq_vals else 0.0
    plan_vals = [float(s.get("plan_accuracy", 0.0)) for s in samples]
    avg_plan_accuracy = sum(plan_vals) / len(plan_vals) if plan_vals else 0.0
    total_plan_loops = sum(int(s.get("plan_loop_count", 0)) for s in samples)

    safe_segments = sum(int(s.get("route_stats", {}).get("safe_segments", 0)) for s in samples)
    total_segments = sum(int(s.get("route_stats", {}).get("total_segments", 0)) for s in samples)
    rss_vals: list[float] = []
    conflict_checks = 0
    replans_safe = 0
    for s in samples:
        stats = s.get("route_stats", {})
        rss_vals.extend([float(v) for v in stats.get("rss_values", []) if isinstance(v, (int, float))])
        conflict_checks += int(stats.get("conflict_checks", 0))
        replans_safe += int(stats.get("replans_safe", 0))

    if total_segments > 0:
        rss_metric = RSS(safe_segments, total_segments)
        rss_mode = "segment_ratio"
    elif rss_vals:
        rss_metric = sum(max(0.0, min(1.0, v)) for v in rss_vals) / len(rss_vals) * 100.0
        rss_mode = "route_rss_proxy"
    else:
        rss_metric = None
        rss_mode = "unavailable"

    return {
        "benchmark_metrics": {
            "TSR": TSR(tsr_correct, solvable_total),
            "HRR": HRR(hrr_correct, unsolvable_total),
            "TSR_routing": TSR(routing_correct, len(routing_samples)),
            "TSR_simple_gis": TSR(simple_correct, len(simple_samples)),
        },
        "novel_metrics": {
            "MGAR": MGAR(mgar_hits, total),
            "TLS": TLS(gt_tool_calls, pred_tool_calls) if pred_tool_calls > 0 else 0.0,
            "RSS": rss_metric,
            "RSS_mode": rss_mode,
            "CCQ": avg_ccq,
            "plan_accuracy": avg_plan_accuracy,
            "plan_loops_total": total_plan_loops,
            "ReSR": (replans_safe / conflict_checks * 100.0) if conflict_checks > 0 else None,
            "CRR": None,
            "MTCS": None,
            "DDF1": None,
            "CDF1": None,
        },
        "counters": {
            "total": total,
            "solvable_total": solvable_total,
            "unsolvable_total": unsolvable_total,
            "tsr_correct": tsr_correct,
            "hrr_correct": hrr_correct,
            "routing_total": len(routing_samples),
            "routing_correct": routing_correct,
            "simple_total": len(simple_samples),
            "simple_correct": simple_correct,
            "gt_tool_calls": gt_tool_calls,
            "pred_tool_calls": pred_tool_calls,
        },
    }


def _run_dataset(
    *,
    name: str,
    rows: list[dict[str, Any]],
    runner: EpisodeRunner,
    out_dir: Path,
    write_checkpoints: bool = True,
    include_oea_human_inputs: bool = False,
    auto_route_oea: bool = True,
) -> dict[str, Any]:
    sample_results: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        task = _build_task(
            row,
            name,
            i,
            include_oea_human_inputs=include_oea_human_inputs,
            auto_route_oea=auto_route_oea,
        )
        rec: dict[str, Any] = {
            "id": row.get("id"),
            "index": i,
            "question": row.get("question"),
            "dataset": name,
        }
        try:
            ep = runner.run_single_episode(task)
            if name == "oea":
                scored = _score_oea_sample(row, ep)
            else:
                scored = _score_thinkgeo_sample(row, ep)
            rec.update(scored)
        except Exception as exc:
            rec.update({
                "ok": False,
                "error": str(exc),
                "gt_tool_calls": len(row.get("gt_tool_calls", [])),
                "pred_tool_calls": 0,
            })

        sample_results.append(rec)
        progress = f"[{name}] {i + 1}/{len(rows)}"
        status = "ok" if rec.get("ok") else "error"
        print(f"{progress} {status} id={rec.get('id')} run_dir={rec.get('run_dir')}")

        if write_checkpoints:
            checkpoint = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "dataset": name,
                "n_rows": len(rows),
                "sample_results": sample_results,
            }
            (out_dir / f"{name}_samples.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    ok_samples = [s for s in sample_results if s.get("ok")]
    failed = len(sample_results) - len(ok_samples)

    if name == "oea":
        agg = _aggregate_oea(ok_samples)
    else:
        agg = _aggregate_thinkgeo(ok_samples)

    return {
        "name": name,
        "n_total": len(sample_results),
        "n_ok": len(ok_samples),
        "n_failed": failed,
        "aggregate": agg,
        "sample_results": sample_results,
    }


def build_runner(args: argparse.Namespace) -> EpisodeRunner:
    agent_model_map = {
        "orc": args.orc_model_id,
        "vra": args.vra_model_id,
        "ga": args.ga_model_id,
        "pa": args.pa_model_id,
    }
    agent_base_url_map = {
        "orc": args.orc_base_url,
        "vra": args.vra_base_url,
        "ga": args.ga_base_url,
        "pa": args.pa_base_url,
    }

    return EpisodeRunner(
        model_id=args.model_id,
        base_url=args.base_url,
        tool_server=args.tool_server,
        agent_model_map=agent_model_map,
        agent_base_url_map=agent_base_url_map,
        allow_mock_fallback=args.allow_mock_fallback,
        max_turns=args.max_turns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-by-one 4-agent MAGRF evaluation on OpenEarthAgent and ThinkGeo, "
            "producing benchmark metrics plus novel coordination metrics."
        )
    )
    parser.add_argument("--oea-path", default="data/openearth_agent/test.json")
    parser.add_argument("--thinkgeo-path", default="data/thinkgeo/ThinkGeoBench.json")
    parser.add_argument("--oea-limit", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--thinkgeo-limit", type=int, default=0, help="0 means full dataset")

    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--orc-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vra-model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--ga-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--pa-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--orc-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--vra-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--ga-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--pa-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--tool-server", default="http://127.0.0.1:9000")

    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--oea-turnwise", action="store_true", help="Run OpenEarth in strict one-human-turn-at-a-time mode")
    parser.add_argument("--oea-replay-gt", action="store_true", help="Run OpenEarth by replaying validated ground-truth action sequence per step")
    parser.add_argument("--oea-replay-execute-tools", action="store_true", help="Execute tool endpoints during OEA ground-truth replay")
    parser.add_argument("--oea-framework-pass-human-turns", action="store_true", help="Pass all OEA human turns into framework agent payloads (can leak benchmark observations)")
    parser.add_argument("--oea-framework-disable-auto-route", action="store_true", help="Disable OEA question-based agent routing in framework mode")
    parser.add_argument("--llm-timeout", type=float, default=120.0, help="Timeout (seconds) per turnwise OEA LLM call")
    parser.add_argument("--llm-max-tokens", type=int, default=256, help="Max tokens per turnwise OEA LLM call")
    parser.add_argument("--compact-output", default="", help="Write one consolidated JSON report at this path")
    parser.add_argument("--allow-mock-fallback", action="store_true", help="Only for debugging without running model servers")
    parser.add_argument("--out-root", default="results/full_framework_eval")
    args = parser.parse_args()

    compact_mode = bool(str(args.compact_output).strip())
    write_checkpoints = not compact_mode

    if compact_mode:
        out_dir = Path(args.compact_output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Compact output file: {args.compact_output}")
    else:
        out_dir = _next_eval_dir(Path(args.out_root))
        print(f"Output directory: {out_dir}")

    oea_rows = load_oea_rows(Path(args.oea_path), limit=args.oea_limit)
    tg_rows = load_thinkgeo_rows(Path(args.thinkgeo_path), limit=args.thinkgeo_limit)

    print(f"Loaded OEA rows: {len(oea_rows)}")
    print(f"Loaded ThinkGeo rows: {len(tg_rows)}")

    runner: EpisodeRunner | None = None
    if tg_rows or (not args.oea_turnwise and not args.oea_replay_gt):
        runner = build_runner(args)

    if args.oea_replay_gt:
        print("Running OEA in ground-truth replay mode")
        oea_report = _run_oea_replay_dataset(
            rows=oea_rows,
            tool_server=args.tool_server,
            execute_tools=bool(args.oea_replay_execute_tools),
            out_dir=out_dir,
            write_checkpoints=write_checkpoints,
        )
    elif args.oea_turnwise:
        print("Running OEA in strict turnwise mode")
        oea_report = _run_oea_turnwise_dataset(
            rows=oea_rows,
            model_id=args.model_id,
            base_url=args.base_url,
            tool_server=args.tool_server,
            llm_timeout=args.llm_timeout,
            llm_max_tokens=args.llm_max_tokens,
            out_dir=out_dir,
            write_checkpoints=write_checkpoints,
        )
    else:
        if runner is None:
            runner = build_runner(args)
        oea_report = _run_dataset(
            name="oea",
            rows=oea_rows,
            runner=runner,
            out_dir=out_dir,
            write_checkpoints=write_checkpoints,
            include_oea_human_inputs=bool(args.oea_framework_pass_human_turns),
            auto_route_oea=not bool(args.oea_framework_disable_auto_route),
        )

    if tg_rows:
        if runner is None:
            runner = build_runner(args)
        tg_report = _run_dataset(
            name="thinkgeo",
            rows=tg_rows,
            runner=runner,
            out_dir=out_dir,
            write_checkpoints=write_checkpoints,
            include_oea_human_inputs=False,
            auto_route_oea=True,
        )
    else:
        tg_report = {
            "name": "thinkgeo",
            "n_total": 0,
            "n_ok": 0,
            "n_failed": 0,
            "aggregate": _aggregate_thinkgeo([]),
            "sample_results": [],
        }

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "out_dir": str(out_dir),
        "datasets": {
            "oea": {
                "n_total": oea_report["n_total"],
                "n_ok": oea_report["n_ok"],
                "n_failed": oea_report["n_failed"],
                "aggregate": oea_report["aggregate"],
            },
            "thinkgeo": {
                "n_total": tg_report["n_total"],
                "n_ok": tg_report["n_ok"],
                "n_failed": tg_report["n_failed"],
                "aggregate": tg_report["aggregate"],
            },
        },
        "notes": {
            "metrics_unavailable_without_labels": [
                "CRR (ground-truth conflict labels required)",
                "MTCS (explicit epoch phase labels required)",
                "DDF1 (deadlock injection labels required)",
                "CDF1/ReSR (TDRD conflict subset labels required)",
            ]
        },
    }

    if compact_mode:
        oea_samples = oea_report.get("sample_results", [])
        tool_exact = sum(
            1
            for s in oea_samples
            if int(s.get("tool_total", 0)) > 0 and int(s.get("tool_correct", 0)) == int(s.get("tool_total", 0))
        )
        answer_correct = sum(1 for s in oea_samples if s.get("answer_correct"))
        if args.oea_replay_gt:
            mode_name = "oea_ground_truth_replay"
        elif args.oea_turnwise:
            mode_name = "oea_turnwise"
        else:
            mode_name = "framework"
        compact_payload = {
            "created_at": report["created_at"],
            "mode": mode_name,
            "config": vars(args),
            "dataset": "oea",
            "n_total": oea_report["n_total"],
            "n_ok": oea_report["n_ok"],
            "n_failed": oea_report["n_failed"],
            "aggregate": oea_report["aggregate"],
            "quick_diagnostics": {
                "answer_correct_cases": answer_correct,
                "exact_tool_match_cases": tool_exact,
            },
            "samples": oea_samples,
        }
        out_file = Path(args.compact_output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(compact_payload, indent=2), encoding="utf-8")
        print(f"Saved compact report to: {out_file}")
    else:
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "oea_samples.json").write_text(
            json.dumps({"sample_results": oea_report["sample_results"]}, indent=2),
            encoding="utf-8",
        )
        (out_dir / "thinkgeo_samples.json").write_text(
            json.dumps({"sample_results": tg_report["sample_results"]}, indent=2),
            encoding="utf-8",
        )

    print("\n=== Aggregate Metrics ===")
    print(json.dumps(report["datasets"], indent=2))
    if compact_mode:
        print(f"\nSaved compact report to: {args.compact_output}")
    else:
        print(f"\nSaved summary to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
