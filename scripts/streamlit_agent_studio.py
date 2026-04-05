#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pydeck as pdk
import streamlit as st

from framework.episode_runner import EpisodeRunner


RESULTS_STUDIO_ROOT = Path("results/agent_studio")


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


def _build_runner(cfg: dict[str, Any]) -> EpisodeRunner:
    agent_model_map = {
        "orc": cfg["orc_model_id"],
        "vra": cfg["vra_model_id"],
        "ga": cfg["ga_model_id"],
        "pa": cfg["pa_model_id"],
    }
    agent_base_url_map = {
        "orc": cfg["orc_base_url"],
        "vra": cfg["vra_base_url"],
        "ga": cfg["ga_base_url"],
        "pa": cfg["pa_base_url"],
    }

    return EpisodeRunner(
        model_id=cfg["model_id"],
        base_url=cfg["base_url"],
        tool_server=cfg["tool_server"],
        agent_model_map=agent_model_map,
        agent_base_url_map=agent_base_url_map,
        allow_mock_fallback=cfg["allow_mock_fallback"],
        max_turns=int(cfg["max_turns"]),
    )


class _QueueWriter:
    def __init__(self, q: queue.Queue[str]) -> None:
        self._q = q
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._q.put(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._q.put(self._buffer)
            self._buffer = ""


def _health_check(model_base: str, model_id: str, tool_server: str) -> dict[str, tuple[bool, str]]:
    status: dict[str, tuple[bool, str]] = {}

    try:
        r = httpx.get(f"{tool_server.rstrip('/')}/health", timeout=10.0)
        r.raise_for_status()
        status["tool_server"] = (True, str(r.text)[:140])
    except Exception as exc:
        status["tool_server"] = (False, str(exc))

    try:
        headers: dict[str, str] = {}
        if "huggingface.co" in str(model_base).lower() or "hf.space" in str(model_base).lower():
            hf_token = (
                str(os.getenv("HF_TOKEN", "")).strip()
                or str(os.getenv("HUGGINGFACEHUB_API_TOKEN", "")).strip()
                or str(os.getenv("HUGGING_FACE_HUB_TOKEN", "")).strip()
            )
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0,
            "max_tokens": 2,
        }
        r = httpx.post(
            f"{model_base.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers if headers else None,
            timeout=20.0,
        )
        r.raise_for_status()
        status["model_server"] = (True, "chat/completions ok")
    except Exception as exc:
        status["model_server"] = (False, str(exc))

    return status


def _load_bundle(run_dir: str | None, ep_bundle: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(ep_bundle, dict) and ep_bundle:
        return ep_bundle

    if run_dir:
        p = Path(run_dir) / "bundle.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _load_ordered_jsonl(run_dir: str | None) -> list[dict[str, Any]]:
    if not run_dir:
        return []
    p = Path(run_dir) / "agents_verbose_ordered.jsonl"
    if not p.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


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
            parsed = json.loads(txt)
            return _extract_answer_text(parsed)
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
            name = str(act.get("name", "")).strip().lower()
            if name == "terminate":
                args = act.get("arguments", {}) if isinstance(act.get("arguments"), dict) else {}
                ans = args.get("ans") or args.get("final_answer")
                if ans not in (None, ""):
                    return _extract_answer_text(ans)
        if "raw_output" in value and value["raw_output"] not in (None, ""):
            return _extract_answer_text(value["raw_output"])
    return ""


def _session_dir() -> Path:
    sid = st.session_state.get("session_id")
    if not sid:
        sid = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        st.session_state.session_id = sid
    p = RESULTS_STUDIO_ROOT / "sessions" / sid
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_run_artifact(run_payload: dict[str, Any]) -> Path:
    sdir = _session_dir()
    rid = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    path = sdir / f"run_{rid}.json"
    path.write_text(json.dumps(_json_safe(run_payload), indent=2), encoding="utf-8")
    return path


def _save_session_snapshot(cfg: dict[str, Any]) -> Path:
    sdir = _session_dir()
    path = sdir / "session.json"
    payload = {
        "session_id": st.session_state.get("session_id"),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "config": _json_safe(cfg),
        "chat_history": _json_safe(st.session_state.get("chat_history", [])),
        "runs": _json_safe(st.session_state.get("runs", [])),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _collect_geo_points(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        lat = obj.get("lat")
        lon = obj.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            out.append({"lat": float(lat), "lon": float(lon), "source": "latlon"})

        bbox = obj.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
            minx, miny, maxx, maxy = bbox
            out.append(
                {
                    "lat": float((miny + maxy) / 2.0),
                    "lon": float((minx + maxx) / 2.0),
                    "source": "bbox_center",
                }
            )

        for v in obj.values():
            _collect_geo_points(v, out)
        return

    if isinstance(obj, list):
        for v in obj:
            _collect_geo_points(v, out)


def _draw_agent_canvas(executed: list[str], tool_counts: dict[str, int]) -> None:
    payload = {
        "executed": executed,
        "tool_counts": tool_counts,
    }
    html = f"""
    <div style='background:linear-gradient(135deg,#0b1320,#102a43,#1f3f5b);padding:12px;border-radius:14px;'>
      <canvas id='agentCanvas' width='980' height='260' style='width:100%;max-width:980px;height:260px;border-radius:12px;'></canvas>
    </div>
    <script>
      const payload = {json.dumps(payload)};
      const canvas = document.getElementById('agentCanvas');
      const ctx = canvas.getContext('2d');
      const nodes = [
        {{id:'orc', label:'ORC'}},
        {{id:'vra', label:'VRA'}},
        {{id:'ga', label:'GA'}},
        {{id:'pa', label:'PA'}},
      ];
      const x0 = 130;
      const y = 130;
      const dx = 240;
      const pos = {{}};
      nodes.forEach((n, i) => {{ pos[n.id] = {{x:x0 + i*dx, y:y}}; }});

      function clear() {{
        const g = ctx.createLinearGradient(0,0,980,260);
        g.addColorStop(0,'#08111f');
        g.addColorStop(1,'#122f49');
        ctx.fillStyle = g;
        ctx.fillRect(0,0,980,260);
      }}

      function edge(a,b,active,index) {{
        const p1 = pos[a], p2 = pos[b];
        ctx.strokeStyle = active ? '#3dd5f3' : 'rgba(170,190,220,0.28)';
        ctx.lineWidth = active ? 4 : 2;
        ctx.beginPath();
        ctx.moveTo(p1.x + 42, p1.y);
        ctx.lineTo(p2.x - 42, p2.y);
        ctx.stroke();
        if (active) {{
          ctx.fillStyle = '#3dd5f3';
          ctx.font = '14px Space Grotesk, sans-serif';
          ctx.fillText(String(index + 1), (p1.x + p2.x)/2 - 4, p1.y - 10);
        }}
      }}

      function node(id, label) {{
        const p = pos[id];
        const done = payload.executed.includes(id);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 40, 0, Math.PI*2);
        ctx.fillStyle = done ? '#13b497' : 'rgba(20,33,61,0.9)';
        ctx.fill();
        ctx.strokeStyle = done ? '#9fffe0' : '#5a7ca3';
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.fillStyle = '#f4f8ff';
        ctx.font = 'bold 16px Space Grotesk, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, p.x, p.y + 6);

        const tc = payload.tool_counts[id] || 0;
        ctx.font = '12px Space Grotesk, sans-serif';
        ctx.fillStyle = '#d3ecff';
        ctx.fillText('tools: ' + tc, p.x, p.y + 56);
      }}

      clear();
      edge('orc','vra', payload.executed.includes('orc') && payload.executed.includes('vra'), 0);
      edge('vra','ga', payload.executed.includes('vra') && payload.executed.includes('ga'), 1);
      edge('ga','pa', payload.executed.includes('ga') && payload.executed.includes('pa'), 2);
      node('orc','ORC');
      node('vra','VRA');
      node('ga','GA');
      node('pa','PA');
    </script>
    """
    st.components.v1.html(html, height=300)


def _render_geo_canvas(bundle: dict[str, Any]) -> None:
    points: list[dict[str, Any]] = []
    _collect_geo_points(bundle.get("agent_tool_calls", {}), points)
    if not points:
        st.info("No geospatial coordinates detected yet from tool outputs in this run.")
        return

    lat = sum(p["lat"] for p in points) / len(points)
    lon = sum(p["lon"] for p in points) / len(points)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_radius=200,
        get_fill_color=[28, 199, 194, 190],
        pickable=True,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=7, pitch=25),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "lat: {lat}\nlon: {lon}\nsource: {source}"},
    )
    st.pydeck_chart(deck, use_container_width=True)


def _run_episode_with_live_stream(task: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str], str | None]:
    log_q: queue.Queue[str] = queue.Queue()
    done = threading.Event()
    result_box: dict[str, Any] = {}

    def worker() -> None:
        writer = _QueueWriter(log_q)
        try:
            runner = _build_runner(cfg)
            with redirect_stdout(writer), redirect_stderr(writer):
                ep = runner.run_single_episode(task)
            result_box["episode"] = ep
        except Exception as exc:
            result_box["error"] = str(exc)
        finally:
            writer.flush()
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    lines: list[str] = []
    ph = st.empty()

    while not done.is_set() or not log_q.empty():
        got = False
        while True:
            try:
                line = log_q.get_nowait()
            except queue.Empty:
                break
            lines.append(line)
            got = True

        if got:
            ph.code("\n".join(lines[-300:]), language="text")
        time.sleep(0.12)

    ph.code("\n".join(lines[-300:]) if lines else "No stdout captured.", language="text")
    if "error" in result_box:
        return None, lines, result_box["error"]
    return result_box.get("episode"), lines, None


def _init_state() -> None:
    if "runs" not in st.session_state:
        st.session_state.runs = []
    if "mission_prompt" not in st.session_state:
        st.session_state.mission_prompt = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
          html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
          }
          .stApp {
            background: radial-gradient(circle at 10% 10%, #0f2740 0%, #08111f 35%, #05080f 100%);
            color: #eaf4ff;
          }
          .hero {
            border: 1px solid rgba(100,170,220,0.25);
            background: linear-gradient(140deg, rgba(27,72,103,0.45), rgba(15,27,47,0.85));
            padding: 18px;
            border-radius: 16px;
            margin-bottom: 10px;
          }
          .metric-card {
            border: 1px solid rgba(100,170,220,0.2);
            background: linear-gradient(160deg, rgba(16,42,67,0.8), rgba(8,20,34,0.85));
            border-radius: 14px;
            padding: 10px 12px;
          }
          .small-note {
            color: #b8d8ee;
            font-size: 0.92rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="GeoAgent Studio", page_icon="G", layout="wide", initial_sidebar_state="expanded")
    _inject_css()
    _init_state()

    st.markdown(
        """
        <div class='hero'>
          <h2 style='margin-bottom:6px;'>GeoAgent Studio - ORC / VRA / GA / PA</h2>
          <div class='small-note'>
            Live mission chat, full 4-agent execution traces, ordered tool actions, and geospatial canvas.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Mission Control")
        model_id = st.text_input("Model ID", value="Qwen/Qwen3-4B-Instruct-2507")
        base_url = st.text_input("Base URL", value="http://127.0.0.1:8002/v1")
        tool_server = st.text_input("Tool Server", value="http://127.0.0.1:9000")
        max_turns = st.number_input("Max turns per agent", min_value=1, max_value=15, value=4)
        allow_mock_fallback = st.checkbox("Allow mock fallback", value=False)
        st.caption("For HF router, set HF_TOKEN in your shell environment before launching Streamlit.")

        st.markdown("---")
        st.subheader("Case Preset")
        oea_path = st.text_input("OEA path", value="data/openearth_agent/test.json")
        case_id = st.text_input("Case ID", value="2")
        seed_all_turns = st.checkbox("Seed with all testcase human turns", value=False)

        st.markdown("---")
        if st.button("Run Health Check", use_container_width=True):
            hs = _health_check(base_url, model_id, tool_server)
            for k, (ok, msg) in hs.items():
                if ok:
                    st.success(f"{k}: {msg}")
                else:
                    st.error(f"{k}: {msg}")

    cfg = {
        "model_id": model_id,
        "base_url": base_url,
        "tool_server": tool_server,
        "orc_model_id": model_id,
        "vra_model_id": model_id,
        "ga_model_id": model_id,
        "pa_model_id": model_id,
        "orc_base_url": base_url,
        "vra_base_url": base_url,
        "ga_base_url": base_url,
        "pa_base_url": base_url,
        "allow_mock_fallback": allow_mock_fallback,
        "max_turns": int(max_turns),
    }

    c1, c2, c3 = st.columns([2.4, 1, 1])
    with c1:
        prompt = st.text_area(
            "Mission Prompt",
            value=st.session_state.mission_prompt,
            height=110,
            placeholder="Tell the geospatial multi-agent system what to do...",
        )
    with c2:
        if st.button("Load Testcase", use_container_width=True):
            try:
                case = _load_case(Path(oea_path), case_id)
                st.session_state.case_payload = case
                st.session_state.mission_prompt = case["question"]
                st.success(f"Loaded testcase {case['id']}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with c3:
        if st.button("Clear Runs", use_container_width=True):
            st.session_state.runs = []
            st.session_state.chat_history = []
            st.success("Cleared run history.")
            _save_session_snapshot(cfg)

    run_clicked = st.button("Run Full 4-Agent Pipeline", type="primary", use_container_width=True)

    if run_clicked:
        st.session_state.mission_prompt = prompt
        if not prompt.strip():
            st.error("Enter a mission prompt first.")
        else:
            st.session_state.chat_history.append(
                {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "role": "user",
                    "text": prompt.strip(),
                }
            )
            human_inputs = [prompt.strip()]
            case = st.session_state.get("case_payload")
            if case and str(case.get("id")) == str(case_id) and seed_all_turns:
                case_turns = [str(x) for x in case.get("human_turns", []) if str(x).strip()]
                primary = next((t for t in case_turns if not t.upper().startswith("OBSERVATION")), "")
                if primary:
                    human_inputs = [primary]
                elif case_turns:
                    human_inputs = [case_turns[0]]

            task = {
                "objective": prompt.strip(),
                "scene_id": f"studio-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                "region": "studio",
                "benchmark": "studio",
                "dataset_row_id": str(case_id),
                "source": "streamlit-studio",
                "human_inputs": human_inputs,
            }

            st.subheader("Live Agent Logs")
            ep, logs, err = _run_episode_with_live_stream(task, cfg)
            if err:
                st.error(f"Pipeline failed: {err}")
                st.session_state.chat_history.append(
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "role": "assistant",
                        "text": f"Pipeline failed: {err}",
                    }
                )
                _save_session_snapshot(cfg)
            else:
                run_dir = ep.get("run_dir") if isinstance(ep, dict) else None
                bundle = _load_bundle(run_dir, ep.get("bundle") if isinstance(ep, dict) else None)
                ordered = _load_ordered_jsonl(run_dir)

                answer_text = _extract_answer_text(ep.get("result", {}) if isinstance(ep, dict) else {})
                if not answer_text:
                    answer_text = _extract_answer_text(bundle.get("agent_outputs", {}))
                if not answer_text:
                    answer_text = "Run completed. Inspect per-agent reasoning and tool outputs below."

                st.session_state.chat_history.append(
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "role": "assistant",
                        "text": answer_text,
                    }
                )

                run_payload = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "session_id": st.session_state.get("session_id"),
                    "prompt": prompt.strip(),
                    "task": task,
                    "run_dir": run_dir,
                    "episode": ep,
                    "bundle": bundle,
                    "ordered": ordered,
                    "logs": logs,
                    "chat_history": list(st.session_state.chat_history),
                }
                artifact_path = _save_run_artifact(run_payload)

                st.session_state.runs.append(
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "prompt": prompt.strip(),
                        "task": task,
                        "run_dir": run_dir,
                        "episode": ep,
                        "bundle": bundle,
                        "ordered": ordered,
                        "logs": logs,
                        "artifact_path": str(artifact_path),
                    }
                )
                session_path = _save_session_snapshot(cfg)
                st.success("Run finished.")
                st.caption(f"Saved run artifact: {artifact_path}")
                st.caption(f"Saved session snapshot: {session_path}")

    if not st.session_state.runs:
        st.info("No runs yet. Load testcase 2 (optional), enter a mission prompt, then run.")
        return

    latest = st.session_state.runs[-1]
    bundle = latest.get("bundle", {}) if isinstance(latest.get("bundle"), dict) else {}
    metrics = latest.get("episode", {}).get("metrics", {}) if isinstance(latest.get("episode"), dict) else {}

    st.markdown("### Chat Transcript")
    for row in st.session_state.chat_history[-20:]:
        role = str(row.get("role", "assistant")).upper()
        text = str(row.get("text", ""))
        ts = str(row.get("ts", ""))
        if role == "USER":
            st.markdown(f"**USER** [{ts}]  ")
            st.write(text)
        else:
            st.markdown(f"**ASSISTANT** [{ts}]  ")
            st.write(text)

    st.markdown("### Latest Run")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><b>Run Dir</b><br>{latest.get('run_dir')}</div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><b>Total Tool Calls</b><br>{metrics.get('total_tool_calls')}</div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><b>Plan Accuracy</b><br>{metrics.get('plan_accuracy')}</div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><b>Safety</b><br>{metrics.get('safety_valid')}</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Agent Canvas", "Per-Agent Reasoning", "Tool Inspector", "Ordered Sequence", "Flow + Raw"])

    with tabs[0]:
        executed = bundle.get("executed_agent_order") or bundle.get("agent_call_order") or []
        calls = bundle.get("agent_tool_calls", {}) if isinstance(bundle.get("agent_tool_calls"), dict) else {}
        counts = {ag: len(calls.get(ag, [])) for ag in ["orc", "vra", "ga", "pa"]}
        _draw_agent_canvas(executed, counts)
        st.markdown("#### Geospatial Canvas")
        _render_geo_canvas(bundle)

    with tabs[1]:
        traces = bundle.get("agent_traces", {}) if isinstance(bundle.get("agent_traces"), dict) else {}
        for agent in ["orc", "vra", "ga", "pa"]:
            with st.expander(f"{agent.upper()} reasoning", expanded=(agent == "orc")):
                rows = traces.get(agent, []) if isinstance(traces.get(agent), list) else []
                if not rows:
                    st.caption("No trace rows for this agent in this run.")
                    continue
                for step in rows:
                    turn = step.get("turn")
                    st.markdown(f"Turn {turn}")
                    thought = step.get("thought")
                    if thought not in (None, ""):
                        st.write(str(thought))
                    calls_local = step.get("tool_calls", []) if isinstance(step.get("tool_calls"), list) else []
                    st.caption(f"tool_calls: {len(calls_local)}")

    with tabs[2]:
        traces = bundle.get("agent_traces", {}) if isinstance(bundle.get("agent_traces"), dict) else {}
        for agent in ["vra", "ga", "pa"]:
            steps = traces.get(agent, []) if isinstance(traces.get(agent), list) else []
            if not steps:
                continue
            st.markdown(f"#### {agent.upper()} tool calls")
            for step in steps:
                turn = step.get("turn")
                calls_local = step.get("tool_calls", []) if isinstance(step.get("tool_calls"), list) else []
                for idx, call in enumerate(calls_local, start=1):
                    tool_name = call.get("tool")
                    with st.expander(f"{agent.upper()} turn {turn} - {tool_name} [{idx}]", expanded=False):
                        st.json(
                            {
                                "args": call.get("args", {}),
                                "normalized_args": call.get("normalized_args", {}),
                                "latency_ms": call.get("latency_ms"),
                                "output": call.get("output", {}),
                            },
                            expanded=False,
                        )

    with tabs[3]:
        ordered = latest.get("ordered", []) if isinstance(latest.get("ordered"), list) else []
        if not ordered:
            st.caption("No ordered sequence file found yet.")
        else:
            st.dataframe(ordered, use_container_width=True, height=360)

    with tabs[4]:
        flow = bundle.get("flow_log", {}) if isinstance(bundle.get("flow_log"), dict) else {}
        events = flow.get("events", []) if isinstance(flow.get("events"), list) else []
        st.markdown("#### Flow events")
        if events:
            st.dataframe(events, use_container_width=True, height=300)
        else:
            st.caption("No flow events found in bundle.")

        st.markdown("#### Raw bundle")
        st.json(bundle, expanded=False)

    st.markdown("### Run History")
    for i, run in enumerate(reversed(st.session_state.runs[-8:]), start=1):
        with st.expander(f"Run {i} - {run.get('run_dir')} - {run.get('ts')}"):
            st.write(run.get("prompt"))
            if run.get("artifact_path"):
                st.caption(f"artifact: {run.get('artifact_path')}")
            if run.get("logs"):
                st.code("\n".join(run.get("logs", [])[-120:]), language="text")


if __name__ == "__main__":
    main()
