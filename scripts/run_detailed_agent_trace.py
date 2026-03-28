#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from framework.episode_runner import EpisodeRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAGRF and emit full 4-agent trace bundle.")
    parser.add_argument(
        "--query",
        default="Plan safest evacuation route to nearest shelter after flood with damaged roads.",
        help="Task objective for the 4-agent pipeline",
    )
    parser.add_argument("--model-id", default="mock-model")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--tool-server", default="http://localhost:9000")
    parser.add_argument("--orc-model-id", default=None)
    parser.add_argument("--vra-model-id", default=None)
    parser.add_argument("--ga-model-id", default=None)
    parser.add_argument("--pa-model-id", default=None)
    parser.add_argument("--orc-base-url", default=None)
    parser.add_argument("--vra-base-url", default=None)
    parser.add_argument("--ga-base-url", default=None)
    parser.add_argument("--pa-base-url", default=None)
    parser.add_argument("--strict-no-mock-fallback", action="store_true")
    args = parser.parse_args()

    agent_model_map = {
        "orc": args.orc_model_id or args.model_id,
        "vra": args.vra_model_id or args.model_id,
        "ga": args.ga_model_id or args.model_id,
        "pa": args.pa_model_id or args.model_id,
    }
    agent_base_url_map = {
        "orc": args.orc_base_url or args.base_url,
        "vra": args.vra_base_url or args.base_url,
        "ga": args.ga_base_url or args.base_url,
        "pa": args.pa_base_url or args.base_url,
    }

    runner = EpisodeRunner(
        model_id=args.model_id,
        base_url=args.base_url,
        tool_server=args.tool_server,
        agent_model_map=agent_model_map,
        agent_base_url_map=agent_base_url_map,
        allow_mock_fallback=not args.strict_no_mock_fallback,
    )
    out = runner.run_episode(args.query)
    run_dir = Path(out["episode"]["run_dir"])

    print(f"Run folder: {run_dir}")
    print(f"Bundle: {run_dir / 'bundle.json'}")
    print(f"Summary: {run_dir / 'summary.md'}")
    print("Final status:", out["final"]["status"])
    print("Final result keys:", list((out["episode"]["result"] or {}).keys()))
    print("Episode JSON:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

