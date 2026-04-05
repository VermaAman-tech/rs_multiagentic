#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from datetime import datetime
from uuid import uuid4

from framework.episode_runner import EpisodeRunner


RESULTS_LIVE_CHAT_ROOT = Path("results/agent_live_chat")


def _clean_human_turn(value: Any) -> str:
    text = str(value or "").replace("<AGENT_PROMPT>", "").strip()
    if "Question:" in text:
        text = text.split("Question:", 1)[1].strip()
    return " ".join(text.split())


def _extract_human_turns(item: dict[str, Any]) -> list[str]:
    turns: list[str] = []
    conv = item.get("conversation", []) if isinstance(item.get("conversation"), list) else []
    for msg in conv:
        if not isinstance(msg, dict) or msg.get("from") != "human":
            continue
        clean = _clean_human_turn(msg.get("value", ""))
        if clean:
            turns.append(clean)
    return turns


def _extract_question(turns: list[str]) -> str:
    for t in turns:
        if not t.upper().startswith("OBSERVATION"):
            return t
    return turns[0] if turns else ""


def _load_case(path: Path, case_id: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("idx", i))
        if rid == case_id:
            turns = _extract_human_turns(item)
            return {
                "id": rid,
                "question": _extract_question(turns),
                "human_turns": turns,
            }

    raise ValueError(f"Case id {case_id} not found in {path}")


def _print_run_paths(run_dir: str | None) -> None:
    if not run_dir:
        print("run_dir: <missing>")
        return
    print(f"run_dir: {run_dir}")
    print(f"bundle: {run_dir}/bundle.json")
    print(f"flow_log: {run_dir}/flow_log.json")
    print(f"ordered_actions: {run_dir}/agents_verbose_ordered.jsonl")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _extract_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return ""
        try:
            return _extract_answer_text(json.loads(txt))
        except Exception:
            return txt
    if isinstance(value, dict):
        for k in ("ans", "final_answer", "answer", "response"):
            if k in value and value[k] not in (None, ""):
                return _extract_answer_text(value[k])
        actions = value.get("actions") if isinstance(value.get("actions"), list) else []
        for act in actions:
            if not isinstance(act, dict):
                continue
            if str(act.get("name", "")).strip().lower() == "terminate":
                args = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
                ans = args.get("ans") or args.get("final_answer")
                if ans not in (None, ""):
                    return _extract_answer_text(ans)
        if value.get("raw_output"):
            return _extract_answer_text(value.get("raw_output"))
    return ""


def _load_bundle(run_dir: str | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    p = Path(run_dir) / "bundle.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _build_runner(args: argparse.Namespace) -> EpisodeRunner:
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
    parser = argparse.ArgumentParser(description="Interactive live debug chat for one OEA testcase.")
    parser.add_argument("--oea-path", default="data/openearth_agent/test.json")
    parser.add_argument("--case-id", default="2", help="OEA idx to debug (default: 2)")

    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--orc-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vra-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--ga-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--pa-model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--orc-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--vra-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--ga-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--pa-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--tool-server", default="http://127.0.0.1:9000")
    parser.add_argument("--max-turns", type=int, default=4)

    parser.add_argument("--allow-mock-fallback", action="store_true")
    parser.add_argument("--seed-with-all-case-human-turns", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one episode then exit")
    args = parser.parse_args()

    case = _load_case(Path(args.oea_path), str(args.case_id))
    runner = _build_runner(args)

    session_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    session_dir = RESULTS_LIVE_CHAT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    question = case["question"]
    history = [question]
    if args.seed_with_all_case_human_turns:
        history = [str(x) for x in case.get("human_turns", [])]

    chat_history: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    print(f"Loaded testcase id={case['id']}")
    print(f"Question: {question}")
    print("Live logs will print from ORC/VRA/GA/PA while running.")
    print(f"Session JSON folder: {session_dir}")

    def _save_session() -> None:
        payload = {
            "session_id": session_id,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "case": case,
            "config": {
                "model_id": args.model_id,
                "base_url": args.base_url,
                "tool_server": args.tool_server,
                "max_turns": args.max_turns,
                "seed_with_all_case_human_turns": args.seed_with_all_case_human_turns,
            },
            "chat_history": _json_safe(chat_history),
            "runs": _json_safe(runs),
        }
        (session_dir / "session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run_episode() -> None:
        current_objective = (history[-1] if history else question).strip()
        chat_history.append(
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "role": "user",
                "text": current_objective,
            }
        )

        task = {
            "objective": current_objective,
            "scene_id": f"oea-case-{case['id']}-live",
            "region": "oea",
            "benchmark": "oea",
            "dataset_row_id": case["id"],
            "source": "oea-live-chat",
            "human_inputs": [history[0]] if history else [current_objective],
        }
        ep = runner.run_single_episode(task)
        run_dir = ep.get("run_dir") if isinstance(ep, dict) else None
        bundle = _load_bundle(run_dir)
        answer = _extract_answer_text(ep.get("result", {}) if isinstance(ep, dict) else {})
        if not answer:
            answer = _extract_answer_text(bundle.get("agent_outputs", {}))
        if not answer:
            answer = "Run completed. Inspect run bundle for full details."

        chat_history.append(
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "role": "assistant",
                "text": answer,
            }
        )

        run_payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "task": task,
            "history": list(history),
            "episode": ep,
            "bundle": bundle,
            "chat_history": list(chat_history),
        }
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        run_path = session_dir / f"run_{run_id}.json"
        run_path.write_text(json.dumps(_json_safe(run_payload), indent=2), encoding="utf-8")

        runs.append(
            {
                "ts": run_payload["ts"],
                "run_dir": run_dir,
                "run_artifact": str(run_path),
            }
        )
        _save_session()

        _print_run_paths(ep.get("run_dir"))
        print(f"run_artifact: {run_path}")
        print(f"session_snapshot: {session_dir / 'session.json'}")

    if args.once:
        run_episode()
        return

    print("Commands: /run  /show  /reset  /quit")
    print("Type any other text to append a new human turn before running.")

    while True:
        try:
            user_in = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if user_in == "/quit":
            print("Exiting.")
            return
        if user_in == "/show":
            print("Current human turn history:")
            for i, t in enumerate(history, start=1):
                print(f"  {i}. {t}")
            continue
        if user_in == "/reset":
            history = [question]
            print("History reset to the base testcase question.")
            continue
        if user_in and user_in != "/run":
            history.append(user_in)

        run_episode()


if __name__ == "__main__":
    main()
