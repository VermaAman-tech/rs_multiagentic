# MAGRF 4-Agent Framework — Outstanding Tasks

Comprehensive issue analysis from full codebase review.
Generated 2026-04-01.

---

## 1. ORC — Orchestrator Does Not Reason Dynamically

**Status:** CRITICAL
**Location:** `framework/episode_runner.py` (lines 294–313, 858–863), `agents/orc.py`

### Problem

The ORC is supposed to be the system's brain — it should analyze each query, decide
which agents to call, in what order, and dynamically reroute when an agent fails.
Currently it does **none** of that:

- `_decompose_task()` (line 294) hard-codes three static sub-task strings regardless of
  the query content. A query like *"Which fire station and police station are closest to
  each other in Banff National Park?"* should dispatch GA → PA only. Instead it always
  fires VRA → GA → PA, sending VRA to do "visual-temporal evidence extraction" when there
  is zero imagery involved.
- `_resolve_agent_sequence()` (line 302) defaults to `["vra", "ga", "pa"]` unless the
  task dict explicitly overrides it — but nothing ever sets that override.
- When GA crashes with a 400 error (as shown in your logs at turn 2), the runner just
  continues the fixed sequence and never retries or reroutes.
- The `Orchestrator` class itself (orc.py) has no planning logic of its own — it only has
  an `aggregate()` method for merging results after the fact.

### Tasks

- [x] **T1.1** Give ORC an actual LLM-based planning step: before dispatching agents, call
      ORC's LLM with the objective and available agent capabilities. ORC should return a
      JSON plan with `{agent_sequence, sub_tasks, rationale}`. Use this plan instead of
      the hardcoded decomposition.
- [x] **T1.2** Implement **dynamic rerouting**: after each agent completes (or fails),
      pass its result/error back to ORC's LLM to decide the next step. If an agent
      returned an error or timeout, ORC should decide: retry, skip, reroute to different
      agent, or abort.
- [x] **T1.3** Replace the static `_decompose_task()` with the output of ORC's planning
      LLM call. The sub-task descriptions should be specific to the actual objective.
- [x] **T1.4** Add an ORC `plan()` method to `Orchestrator` class that actually calls the
      LLM and returns a structured plan. Move planning logic from episode_runner into ORC.

---

## 2. Agent Error Handling — No Recovery from LLM/Tool Failures

**Status:** CRITICAL
**Location:** `agents/base_agent.py` (lines 83–124), `framework/episode_runner.py`

### Problem

From the logs, GA turn 2 fails with a 400 Bad Request and immediately terminates — the
LLM mock fallback generates a `Terminate` action with `"ans": "error"`. The ORC then
proceeds as if GA completed successfully. Correct behavior:

- ORC should inspect the agent result for error indicators
- On error, decide: retry with different parameters, skip agent, or abort the plan
- The 400 error itself may come from oversized context being sent to the LLM

### Tasks

- [x] **T2.1** Add error detection in `_run_agent_from_task()` — check for error
      indicators in the result (`"ans": "error"`, `"ans": "timeout"`, error thoughts,
      empty tool_calls). Tag the result as `error_result=True`.
- [x] **T2.2** After each agent completes, if the result is an error, invoke ORC's
      dynamic replanning (from T1.2) to decide recovery action.
- [x] **T2.3** Implement retry logic: allow `_execute_task` to retry an agent up to N
      times with exponential backoff.
- [x] **T2.4** Improve `_call_llm` error handling in base_agent.py — the generic
      `except Exception` block at line 116 catches 400 Bad Request (which likely means
      the request payload was malformed/too large) and silently terminates. Instead, log
      the actual error, truncate context, and retry.

---

## 3. Token Limit / Context Explosion — Agents Send Full Payload to LLM

**Status:** CRITICAL
**Location:** `agents/base_agent.py` (lines 447–450, 560–566), `framework/protocols/compression.py`

### Problem

When a tool returns a huge payload (e.g., `GetAreaBoundary` returning a WKT polygon with
thousands of coordinate pairs), the system:

1. Stores the full raw response in tool messages via `_tool_result_for_llm` (line 564)
2. Appends all of these to message history
3. Sends the entire history to the LLM

The `_tool_result_for_llm` method (line 321) truncates at 3200 chars — but this is
**per-message**, not per-conversation. After multiple tool calls, the accumulated context
far exceeds the 32K model limit, causing 400 errors.

The `_trim_message_history` method (line 360) keeps 18 messages max — but 18 messages x
3200 chars each = 57,600 chars (approx 14,400+ tokens) of tool content alone, plus system
prompts and assistant messages.

Compression (`compress_context`) only runs on episodic memory at the
episode level — it never touches the in-conversation LLM message history.

### Tasks

- [x] **T3.1** Implement per-conversation token budget tracking in BaseAgent. Before
      each LLM call, estimate total tokens in messages and prune/compress if over budget.
- [x] **T3.2** Reduce `_MAX_TOOL_MSG_CHARS` from 3200 to ~1600 for tool results. For
      geospatial outputs (WKT, large arrays), extract only the crucial metadata and
      discard the raw geometry.
- [x] **T3.3** Reduce `_MAX_MESSAGES` from 18 to ~10 for the in-flight conversation.
      More aggressive tail trimming protects against context overflow.
- [x] **T3.4** Add a conversation-level compression step: before sending to LLM, if
      total estimated tokens exceeds (model_max * 0.7), compress oldest tool responses to
      summaries.
- [x] **T3.5** Make `_tool_result_for_llm` smarter about geospatial data: extract bbox
      extents from WKT instead of head/tail truncation. For POIs lists, return count and
      first 5 items instead of all.

---

## 4. Memory and Context Optimization — Episodic Memory is Not Used by Agents

**Status:** HIGH
**Location:** `framework/memory/episodic_memory.py`, `framework/episode_runner.py` (lines
242–267), `agents/base_agent.py`

### Problem

Episodic memory is populated during the run (line 174–178), and context keys are selected
per agent (line 242–267). However, **agents never read episodic memory**. The
`_context_keys_for()` method generates keys, but these keys are only sent in the
`task_payload` as metadata — agents never query the memory to inject prior context.

This means:
- PA never sees VRA's damage polygons or GA's road scores from memory
- GA never sees VRA's flood extent
- Each agent runs blind to what others already found

Working memory (`working_memory.py`) exists but is never instantiated or used anywhere.

### Tasks

- [x] **T4.1** Actually inject episodic memory context into the agent's LLM messages.
      When dispatching to an agent, read the relevant memory keys and include a condensed
      summary as a `system` or `user` message.
- [x] **T4.2** Connect `WorkingMemory` to each agent for per-agent scratchpad tracking.
- [x] **T4.3** Implement selective memory injection: before dispatching PA, inject only
      the VRA damage summary and GA road-condition summary, not the raw tool outputs.
- [x] **T4.4** Add memory TTL and garbage collection — stale keys from previous episodes
      should not leak into current context. (Currently `set_items({})` at line 651 resets
      everything, but this is too aggressive.)

---

## 5. GetAreaBoundary Schema — Missing `buffer_m` Parameter

**Status:** HIGH
**Location:** `tools/schemas.py` (line 121–125), `tools/gis/area_boundary.py`

### Problem

The GA agent's tool schema includes `buffer_m` as a parameter (ga.py line 19), but:
- The Pydantic schema `GetAreaBoundaryInput` (schemas.py line 121) only accepts `area_name`
- The `area_boundary.py` implementation reads `req.buffer` not `req.buffer_m`
- So when GA calls `GetAreaBoundary(area_name="Banff...", buffer_m=3000)`, the `buffer_m`
  parameter is silently dropped by Pydantic validation, and no buffer is applied

For the Banff query specifically, the 3000m buffer is critical to the answer.

### Tasks

- [x] **T5.1** Add `buffer_m: Optional[int] = 0` to `GetAreaBoundaryInput` in schemas.py.
- [x] **T5.2** Update `area_boundary.py` to read `req.buffer_m` instead of `req.buffer`.
- [x] **T5.3** Audit all tool schemas to verify they match both the agent tool definitions
      and the tool implementations.

---

## 6. AddPoisLayer — Missing POI Categories for the Query

**Status:** HIGH
**Location:** `tools/gis/pois_layer.py`, `agents/ga.py`

### Problem

The Banff query requires finding fire stations and police stations. The `pois_layer.py`
tool's `_query_to_osm_tags()` only maps "shelter/hospital", "road/street", and "building".
There is no mapping for "fire_station" or "police" — it falls through to the generic
`{"amenity": True}` which downloads ALL amenities (massive payload) and wastes tokens.

Also, POIs are returned with `"name": req.poi_category` (generic) instead of the actual
POI name from OSM data. This makes it impossible for the agent to distinguish fire
stations from police stations.

### Tasks

- [x] **T6.1** Expand `_query_to_osm_tags()` to map common POI categories:
      fire_station, police, school, pharmacy, gas station, etc.
- [x] **T6.2** Return actual OSM names in the POI result, not just the query category.
      Use `row.get("name", req.poi_category)`.
- [x] **T6.3** Add bbox extraction from boundary WKT so that `AddPoisLayer` can receive
      the actual Banff bbox instead of a placeholder.

---

## 7. Safety Protocol — Always Passes Unless Explicitly Enforced

**Status:** MEDIUM
**Location:** `framework/protocols/safety.py`

### Problem

`validate_safety_payload()` returns `(True, [])` unless the payload contains
`enforce_safety_keys=True`. No task ever sets this flag, so safety **always passes**.
Yet the log shows `Safety: FAIL — missing []` — this contradiction happens because the
`task_success` metric at line 1013 also checks `has_answer` which can be false when the
agent returns `"ans": "error"` or `"ans": "mock result"`.

The safety check should actually validate that required spatial/route outputs are present
when a disaster/routing query is being answered.

### Tasks

- [x] **T7.1** Make safety validation context-aware: determine required keys from the
      query type (disaster requires damage_polygons+flood_extent, routing requires
      impassable_roads).
- [x] **T7.2** Fix the contradictory safety report: `Safety: FAIL — missing []` should
      show what actually failed (empty brackets means nothing is missing, but it says FAIL).
- [x] **T7.3** Consolidate TSR logic: `task_success` in episode_runner should use a
      single clear condition rather than a chain of heuristics.

---

## 8. GA Error Recovery — 400 Error Kills the Agent Silently

**Status:** HIGH
**Location:** `agents/base_agent.py` (lines 116–124)

### Problem

From the logs:
```
[AGENT:ga | TURN 2 | 278.76ms]
  THOUGHT: "LLM call failed: Client error '400 Bad Request' ..."
  TERMINATED (no tools)
```

GA called `GetAreaBoundary` on turn 1 successfully but the boundary WKT was enormous.
On turn 2, the conversation history (system + user + assistant + tool result) exceeded
the LLM context window. The 400 error from vLLM means "too many tokens" — but the agent
catches this as a generic exception and terminates with `"ans": "error"`.

### Tasks

- [x] **T8.1** Detect 400/413 errors specifically in `_call_llm` and trigger in-place
      context compression before retrying.
- [x] **T8.2** After context overflow, aggressively truncate the longest tool result
      message and retry once.
- [x] **T8.3** Add a `max_context_chars` check before calling the LLM — if messages
      exceed the limit, proactively trim.

---

## 9. PA Agent — Wrong Tools for the Query

**Status:** HIGH
**Location:** `agents/pa.py`, `framework/episode_runner.py`

### Problem

For the Banff "closest fire/police station" query, PA is dispatched to "generate safe
evacuation routes." PA has `EvacuationRoutePlanner` and `ComputeDistance` — but it
should be using `ComputeDistance` between POI pairs, not route planning.

The real problem is upstream (ORC not decomposing correctly), but PA's system prompt
also has no concept of "proximity analysis" — it only thinks about evacuation routes.

### Tasks

- [x] **T9.1** Broaden PA's system prompt to include proximity/spatial analysis tasks,
      not just evacuation routing.
- [x] **T9.2** Allow PA to receive POI data from GA through memory injection (ties to T4).
- [x] **T9.3** When ORC plans correctly (T1), PA should receive a sub-task like
      "Compute pairwise distances between fire stations and police stations, find minimum."

---

## 10. VRA Agent — Dispatched When Unnecessary

**Status:** MEDIUM
**Location:** `framework/episode_runner.py` (line 312)

### Problem

The default sequence always starts with VRA, even for purely geospatial queries that
don't involve imagery. The Banff query, for instance, needs only GA (boundary + POIs)
and PA (distance computation). VRA wastes turns running `TemporalStackLoader` and
`ChangeDetection` on mock imagery that has nothing to do with the question.

### Tasks

- [x] **T10.1** ORC's planning step (T1.1) should classify the query type and only
      dispatch VRA for queries involving imagery, change detection, or visual analysis.
- [x] **T10.2** Add query-type classification to ORC's planning: categories like
      geospatial_proximity, disaster_assessment, change_detection, visual_qa.

---

## 11. Conflict Resolution — Runs on Wrong Agent Pair

**Status:** MEDIUM
**Location:** `agents/orc.py` (lines 44–58), `framework/episode_runner.py` (lines 869–881)

### Problem

`resolve_conflict` is always called with the first two agents' results (typically VRA
and GA). But conflict resolution should only apply when two agents claim contradictory
facts about the same phenomenon. Running it on VRA (vision) vs GA (GIS) output for
non-overlapping tasks produces meaningless results.

### Tasks

- [x] **T11.1** Only invoke conflict resolution when agents produce overlapping claims
      (e.g., both VRA and GA claim different flood extents).
- [x] **T11.2** Add semantic conflict detection: compare output keys/domains before
      resolving.

---

## 12. Compression Protocol — Incomplete

**Status:** MEDIUM
**Location:** `framework/protocols/compression.py`

### Problem

- Only runs when `token_estimate > 25,600` in episodic memory (episode_runner line 904)
- Drops Prithvi embeddings entirely, summarizes spectral arrays — but doesn't compress
  the biggest offender: WKT geometry strings from GetAreaBoundary
- CCQ is always 1.0 because all safety facts are trivially preserved (none existed)

### Tasks

- [x] **T12.1** Add WKT/geometry compression: replace full WKT strings with bbox
      summaries (extent, area, centroid).
- [x] **T12.2** Add POI list compression: retain count + top-N only.
- [x] **T12.3** Reconsider the threshold: 25,600 chars (not tokens) is about 6,400
      tokens — far below where context pressure happens.

---

## 13. Deadlock Detector — Never Actually Prevents Deadlock

**Status:** LOW
**Location:** `framework/protocols/deadlock.py`, `framework/episode_runner.py`

### Problem

The deadlock detector tracks wait-for edges (orc to agent), but since execution is
sequential (or explicitly parallel VRA+GA), no actual deadlock cycles can form. The
detector runs at line 191–193 only on ACK timeout, which itself only fires when the
broker is empty (line 161) — but the broker is populated synchronously, so it is
always non-empty.

### Tasks

- [x] **T13.1** Revisit if deadlock detection is needed at all in the current
      architecture. If kept, make it useful for the parallel VRA+GA case.
- [x] **T13.2** Fix the ACK timeout flow: the `_await_ack` method at line 156 calls
      `_drain_for("orc")` which already drains the queue — so the ACK published at
      line 154 was just consumed by the drain. The self-ACK architecture is broken
      (agent emits ACK -> broker -> immediately drained -> found).

---

## 14. Message Broker — Self-Delivery Issue

**Status:** MEDIUM
**Location:** `framework/episode_runner.py` (lines 141–154, 156–171)

### Problem

The `_emit_ack` method publishes an ACK from the agent to ORC. Then `_await_ack`
immediately drains the ORC mailbox looking for that ACK. Since publish/drain are
synchronous in-memory operations, the ACK is always found instantly. The timeout path
is never exercised, and the deadlock detector is never engaged meaningfully.

### Tasks

- [x] **T14.1** Restructure ACK flow: separate the publish and drain into the actual
      task execution lifecycle so that ACK represents genuine agent readiness, not a
      pre-execution formality.
- [x] **T14.2** Consider simplifying the MPC protocol for single-process mode — the
      full pub/sub abstraction adds complexity without benefit when all agents run in
      the same process.

---

## 15. Tool Server — Schema/Implementation Mismatches

**Status:** MEDIUM
**Location:** `tools/schemas.py`, various tool implementations

### Tasks

- [x] **T15.1** `GetAreaBoundaryInput` missing `buffer_m` (see T5.1).
- [x] **T15.2** `GetAreaBoundaryOutput` only has `boundary_wkt`, but the implementation
      also returns `gpkg_path` and `success` — these should be in the schema.
- [x] **T15.3** `AddPoisLayerOutput` has `pois: List[Dict]` but actual returns also
      include `success` — add it.
- [x] **T15.4** `ComputeDistanceOutput` only has `distance_meters`, but implementation
      also returns `success` — missing from schema.
- [x] **T15.5** All tool implementations silently succeed on error (returning fake
      data). This masks failures — errors should propagate clearly.

---

## 16. Agent Prompts — Not Task-Adaptive

**Status:** HIGH
**Location:** `agents/ga.py`, `agents/pa.py`, `agents/vra.py`

### Problem

Agent system prompts are static and generic. For the Banff query, GA's prompt says
"Build GIS layers that support routing and risk-aware planning" and PA's says
"Turn geospatial evidence into safe and actionable route plans." Neither prompt
is relevant to a simple proximity query.

### Tasks

- [x] **T16.1** Make agent prompts task-adaptive: ORC's plan should include a
      task-specific instruction that gets prepended to the agent's system prompt.
- [x] **T16.2** Add query-type-specific prompt templates for common task patterns:
      proximity, change detection, damage assessment, route planning.

---

## 17. Token Counting — Grossly Inaccurate

**Status:** MEDIUM
**Location:** `agents/base_agent.py` (line 83), `framework/memory/episodic_memory.py` (line 33)

### Problem

Token estimation is `len(text) // 4` everywhere. This is a crude heuristic that
underestimates for JSON/code content and overestimates for CJK text. Combined with
the 25,600-char compression trigger, actual token usage is poorly tracked.

### Tasks

- [x] **T17.1** Use `tiktoken` or a proper tokenizer matched to the Qwen model for
      accurate token counting.
- [x] **T17.2** Add a global token budget tracker that sums actual LLM request +
      response tokens and enforces limits.

---

## 18. Tests — Incomplete Coverage

**Status:** MEDIUM
**Location:** `tests/`

### Problem

Tests exist but are minimal:
- `test_agents.py` — 1 file, basic agent instantiation
- `test_single_epoch.py` — smoke test only
- No tests for: ORC planning, error recovery, context compression, tool normalization,
  memory injection, safety validation

### Tasks

- [x] **T18.1** Add unit tests for ORC's planning/replanning logic (once T1 is done).
- [x] **T18.2** Add integration tests for error recovery: agent failure then ORC reroute.
- [x] **T18.3** Add context-overflow tests: verify LLM call doesn't fail with large tool
      outputs.
- [x] **T18.4** Add tool schema validation tests: every GA/PA/VRA tool definition must
      match the corresponding Pydantic schema.

---

## 19. Logging — Missing Structured Error Reporting

**Status:** LOW
**Location:** Throughout codebase

### Problem

- Console output uses `print()` directly — no log levels, no structured format
- Error details are lost in broad `except Exception` blocks
- Agent traces go to `logs/agent_trace.jsonl` but aren't connected to episode-level
  error diagnostics

### Tasks

- [x] **T19.1** Replace `print()` with `logging` module, add log levels (DEBUG, INFO,
      WARNING, ERROR).
- [x] **T19.2** Add structured error reporting: when an agent fails, log the full
      error context (payload size, token estimate, tool history, error message).
- [x] **T19.3** Aggregate per-episode error counts in the run bundle for quick triage.

---

## 20. Configuration — Not Used at Runtime

**Status:** LOW
**Location:** `configs/agents.yaml`, `configs/models.yaml`

### Problem

Config files exist (`agents.yaml`, `models.yaml`, `tools.yaml`) with useful settings
like `max_react_turns`, `working_memory_tokens`, `episodic_memory_soft_limit_tokens`,
`thinking_mode`, etc. — but **none of these are loaded or used by the code**. All values
are hardcoded.

### Tasks

- [x] **T20.1** Add config loading at startup: parse YAML configs and inject values
      into agent constructors and episode runner.
- [x] **T20.2** Respect `max_react_turns` from config instead of the hardcoded `max_turns`
      constructor parameter.
- [x] **T20.3** Respect `working_memory_tokens` to cap agent working memory.
- [x] **T20.4** Respect `thinking_mode` to toggle CoT in vLLM calls.

---

## Priority Summary

| Priority | Task IDs | Description |
|----------|----------|-------------|
| **P0 — CRITICAL** | T1.1–T1.4 | ORC must reason and plan dynamically |
| **P0 — CRITICAL** | T2.1–T2.4 | Error recovery instead of silent failure |
| **P0 — CRITICAL** | T3.1–T3.5 | Token/context overflow causing 400 errors |
| **P1 — HIGH** | T4.1–T4.4 | Memory injection so agents see each other's output |
| **P1 — HIGH** | T5.1–T5.3 | GetAreaBoundary buffer_m parameter |
| **P1 — HIGH** | T6.1–T6.3 | POI category mapping for real queries |
| **P1 — HIGH** | T8.1–T8.3 | GA-specific 400 error recovery |
| **P1 — HIGH** | T9.1–T9.3 | PA task-awareness |
| **P1 — HIGH** | T16.1–T16.2 | Task-adaptive prompts |
| **P2 — MEDIUM** | T7, T10–T12, T14–T15, T17 | Safety, VRA gating, compression, MPC, schemas, token counting |
| **P3 — LOW** | T13, T18–T20 | Deadlock, tests, logging, config |

---

## Execution Order Recommendation

1. **Phase 1 — Stop the Bleeding (T3 + T5 + T8)**
   Fix the 400 errors first: reduce context sizes, fix tool schemas, add LLM retry on
   context overflow. This makes the framework runnable.

2. **Phase 2 — Intelligent Orchestration (T1 + T2 + T10)**
   Give ORC real planning + error recovery. This is the core behavioral fix that will
   make queries like Banff work correctly.

3. **Phase 3 — Memory and Context Flow (T4 + T6 + T9 + T16)**
   Wire up inter-agent context, fix POI tools, make PA and prompts task-aware.

4. **Phase 4 — Polish (T7 + T11 + T12 + T14 + T15 + T17)**
   Safety, conflict, compression, MPC, schemas, token tracking.

5. **Phase 5 — Quality (T13 + T18 + T19 + T20)**
   Tests, logging, config, deadlock.
