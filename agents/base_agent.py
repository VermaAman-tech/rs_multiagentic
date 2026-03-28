from __future__ import annotations
import json
import httpx
import time
from typing import Any, Dict, List
from dataclasses import dataclass, field
from framework.mpc.message import Message

from datetime import datetime
from pathlib import Path

@dataclass
class AgentResult:
    output: dict[str, Any]
    confidence: float
    tool_calls: list[dict[str, Any]]
    n_turns: int
    trace: list[dict[str, Any]] = field(default_factory=list)

def append_trace_line(trace_step: dict[str, Any]) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "agent_trace.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace_step) + "\n")

class BaseAgent:
    name = "base"

    def __init__(
        self,
        model_id: str = "mock-model",
        base_url: str = "http://localhost:8000/v1",
        tool_server: str = "http://localhost:9000",
        max_turns: int = 15,
        api_key: str = "sk-mock",
        allow_mock_fallback: bool = True,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.tool_server = tool_server
        self.max_turns = max_turns
        self.api_key = api_key
        self.allow_mock_fallback = allow_mock_fallback

    def _get_system_prompt(self) -> str:
        return f"You are the {self.name} agent. Use your tools to solve the task."

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Call the LLM via OpenAI-compatible API.
        Falls back to a mock response if the server is unreachable.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.0,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]
        except httpx.ConnectError:
            if not self.allow_mock_fallback:
                raise RuntimeError(
                    f"LLM server unreachable at {self.base_url}. "
                    "Set allow_mock_fallback=True only for offline tests."
                )
            # Server not running — return mock for testing
            return {
                "role": "assistant",
                "content": json.dumps({
                    "thought": "No LLM server available. Returning mock output.",
                    "actions": [{"name": "Terminate", "arguments": {"ans": "mock result"}}]
                }),
                "tool_calls": []
            }
        except httpx.TimeoutException:
            return {
                "role": "assistant",
                "content": json.dumps({
                    "thought": "LLM request timed out.",
                    "actions": [{"name": "Terminate", "arguments": {"ans": "timeout"}}]
                }),
                "tool_calls": []
            }
        except Exception as e:
            return {
                "role": "assistant",
                "content": json.dumps({
                    "thought": f"LLM call failed: {e}",
                    "actions": [{"name": "Terminate", "arguments": {"ans": "error"}}]
                }),
                "tool_calls": []
            }

    def _call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the tool server. Returns the result or error dict."""
        url = f"{self.tool_server}/tools/{tool_name}"
        try:
            resp = httpx.post(url, json=arguments, timeout=60.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            return {"error": f"Tool server not reachable at {self.tool_server}"}
        except httpx.TimeoutException:
            return {"error": f"Tool {tool_name} timed out"}
        except BaseException as e:
            return {"error": str(e)}

    def _extract_inline_tool_call(self, content: str) -> dict | None:
        """
        Extract a tool call from assistant text output (for models that
        don't use the OpenAI tool_calls format but embed JSON in text).
        """
        if not content:
            return None
        try:
            parsed = json.loads(content)
            # Check for {actions: [{name, arguments}]} format
            if "actions" in parsed and isinstance(parsed["actions"], list):
                for act in parsed["actions"]:
                    if act.get("name") and act["name"] != "Terminate":
                        return act
            # Check for {name, arguments} format
            if parsed.get("name") and parsed["name"] != "Terminate":
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _extract_thought_actions(self, content: str) -> tuple[str | None, list[dict[str, Any]]]:
        if not content:
            return None, []
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                thought = parsed.get("thought")
                actions = parsed.get("actions") if isinstance(parsed.get("actions"), list) else []
                return thought, actions
        except (json.JSONDecodeError, TypeError):
            pass
        return None, []

    def handle_task(self, message: Message, tools_schema: list[dict] = None) -> AgentResult:
        if tools_schema is None:
            tools_schema = []

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Task instructions:\n{json.dumps(message.payload)}"}
        ]

        turns = 0
        all_tool_calls_logged = []
        trace: list[dict[str, Any]] = []
        response_msg: dict[str, Any] = {"content": ""}

        while turns < self.max_turns:
            turns += 1
            llm_t0 = time.perf_counter()
            response_msg = self._call_llm(messages, tools_schema)
            llm_ms = (time.perf_counter() - llm_t0) * 1000.0
            messages.append(response_msg)
            content = response_msg.get("content", "")
            thought, actions = self._extract_thought_actions(content)

            # Check for OpenAI-style tool_calls
            tool_calls = response_msg.get("tool_calls") or []

            # Also check for inline tool calls in text content
            if not tool_calls:
                inline = self._extract_inline_tool_call(content)
                if inline:
                    tool_calls = [{
                        "id": f"call_inline_{turns}",
                        "function": {
                            "name": inline["name"],
                            "arguments": json.dumps(inline.get("arguments", {}))
                        }
                    }]

            trace_step: dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent": self.name,
                "task_id": message.payload.get("scene_id", "unknown"),
                "turn": turns,
                "llm_latency_ms": round(llm_ms, 2),
                "assistant_raw_content": content,
                "thought": thought,
                "actions_in_text": actions,
                "tool_calls": [],
                "terminated": False
            }

            if not tool_calls:
                trace_step["terminated_no_tool_call"] = True
                trace_step["terminated"] = True
                trace.append(trace_step)
                append_trace_line(trace_step)
                
                print(f"[AGENT:{self.name} | TURN {turns} | {round(llm_ms, 2)}ms]")
                if thought:
                    print(f"  THOUGHT : {json.dumps(thought)}")
                print(f"  TERMINATED (no tools)")
                print()
                break

            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{turns}")
                fn = tc.get("function", {})
                t_name = fn.get("name")
                t_args_str = fn.get("arguments", "{}")
                try:
                    t_args = json.loads(t_args_str) if isinstance(t_args_str, str) else t_args_str
                except (json.JSONDecodeError, TypeError):
                    t_args = {}

                tool_t0 = time.perf_counter()
                t_res = self._call_tool(t_name, t_args)
                tool_ms = (time.perf_counter() - tool_t0) * 1000.0
                all_tool_calls_logged.append({
                    "tool": t_name,
                    "args": t_args,
                    "output": t_res,
                    "latency_ms": round(tool_ms, 2),
                    "turn": turns,
                })
                trace_step["tool_calls"].append({
                    "id": tc_id,
                    "tool": t_name,
                    "args": t_args,
                    "output": t_res,
                    "latency_ms": round(tool_ms, 2),
                })
                
                # Console output
                print(f"[AGENT:{self.name} | TURN {turns} | {round(llm_ms, 2)}ms]")
                if thought:
                    print(f"  THOUGHT : {json.dumps(thought)}")
                print(f"  TOOL CALL: {t_name}({json.dumps(t_args)})")
                res_str = json.dumps(t_res)
                if len(res_str) > 150: res_str = res_str[:147] + "..."
                print(f"  TOOL RES : {res_str}")
                print(f"  LATENCY  : tool={round(tool_ms, 2)}ms")
                print()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": t_name,
                    "content": json.dumps(t_res)
                })
            trace.append(trace_step)
            append_trace_line(trace_step)

        # Final extraction
        final_content = response_msg.get("content", "")
        output_dict = {"raw_output": final_content}

        # Try to parse structured output from the final response
        try:
            parsed_final = json.loads(final_content)
            if isinstance(parsed_final, dict):
                output_dict.update(parsed_final)
        except (json.JSONDecodeError, TypeError):
            pass

        # Preserve task metadata from payload
        for k in ("objective", "scene_id", "region"):
            if k in message.payload:
                output_dict.setdefault(k, message.payload[k])

        return AgentResult(
            output=output_dict,
            confidence=0.8,
            tool_calls=all_tool_calls_logged,
            n_turns=turns,
            trace=trace,
        )
