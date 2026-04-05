# MAGRF Pipeline v3 — Root-Cause Audit & Fix Plan

> **Objective**: Make the pipeline produce correct, evidence-grounded answers
> for every OEA case, with per-agent tool-call traces that can be compared
> 1:1 against the ground-truth action sequences.

---

## 1  Executive Failure Summary (eval_027)

| Case | GT Answer (excerpt) | Pipeline Answer | Root Cause |
|------|---------------------|-----------------|------------|
| 0 (Banff) | `Banff Fire Station ↔ RCMP Banff Detachment (836.74 m)` | Same ✅ but **0/5 tool match** | Canonical shortcut bypasses tool chain; `pred_actions` = `[Terminate, Terminate]` instead of `[GetAreaBoundary, AddPoisLayer, AddPoisLayer, ComputeDistance, Terminate]` |
| 1 (Garbage) | `2975.0 sq m` | Same ✅ but **0/3 tool match** | Canonical shortcut hardcodes answer; no `TextToBbox` or `Calculator` recorded |
| 2 (Helicopter) | `296.4 px → 42 m` | Same ✅ but **0/3 tool match** | Canonical shortcut hardcodes answer |
| 3 (NDBI Nizwa) | `moderate growth 0.14%…` | `No satellite imagery…` ❌ | `TemporalStackLoader` returns `success: false` because mock data files don't exist; all 3 agents parrot the same error; **no actual tools called** |
| 4 (Assiniboine) | `Terrace→Assiniboine…` | Same ✅ but **0/5 tool match** | Canonical shortcut |

**Key insight**: The answer text "matches" ground truth only because hardcoded
canonical shortcuts in `ga.py` and `vra.py` reproduce the GT answer verbatim.
No actual agent reasoning or tool execution happens. The pipeline is a lookup
table for 4/5 cases and a total failure for case 3.

---

## 2  Detailed Root Causes

### 2.1  Canonical / Hardcoded Answer Shortcuts

| File | Function | Line(s) | What it does |
|------|----------|---------|-------------|
| `agents/ga.py` | `_trial_canonical_oea()` | 209-283 | If the objective keyword-matches "banff+fire+police+closest" or "assiniboine+restaurant+nearest+park", it returns a **hardcoded answer** with `tool_calls=[]`, bypassing the ReAct loop entirely |
| `agents/ga.py` | `handle_task()` | 441-450 | Calls `_trial_canonical_oea()` **before** any LLM/tool interaction |
| `agents/vra.py` | `_trial_bbox_solver()` | 362-610 | For "domestic garbage + GSD=0.5" or "helicopter + GSD=0.14153", returns a **hardcoded answer** with the exact GT values. For other cases, it does call `TextToBbox` but the deterministic postprocessing may still override the LLM loop |
| `agents/vra.py` | `handle_task()` | 612-630 | Calls `_trial_bbox_solver()` before any LLM interaction |
| `agents/pa.py` | `_trial_visual_passthrough()` | 339-426 | Parrots VRA's answer for garbage/helicopter queries without calling PA tools |

**Impact**: The predicted action sequence logged is just `[Terminate, Terminate]`
(one per agent that runs). Ground truth expects 4-5 tool calls. Even though
`answer_match=True`, the `Tool` metric is 0/5 → **0% Tool accuracy**.

### 2.2  Case 3 (NDBI): Full Pipeline Failure

**Flow**: ORC classifies as `spectral_change` → routes to `[vra, ga, pa]`.

1. **VRA** (turn 1): The LLM recognizes the task needs `GetAreaBoundary` → `AddIndexLayer` → `ComputeIndexChange`.
   But the LLM immediately terminates with "No satellite imagery available" because:
   - The system prompt tells it to use `TemporalStackLoader` for temporal queries
   - `TemporalStackLoader` checks for `image_paths` → none passed for non-image cases
   - It checks `local_stack_path` → not set
   - It checks mock paths `data/mock/epoch_0.tif` → don't exist
   - Returns `success: false`
   - LLM decides the task is impossible and terminates

2. **GA** (turn 1): Gets the same conclusion from VRA context → terminates with same error
3. **PA** (turn 1): Same

**Root fix needed**: Case 3 is a pure GIS / spectral query. The pipeline should:
1. Call `GetAreaBoundary` for "Nizwa Province, Oman"
2. Call `AddIndexLayer` for NDBI on the 2019 raster
3. Call `AddIndexLayer` for NDBI on the 2024 raster
4. Call `ComputeIndexChange` between them
5. `Terminate` with the change statistics

This requires **actual geospatial data** or at minimum, the tool server needs
to handle the case where boundary + index tools work even without
`TemporalStackLoader`. The current `TemporalStackLoader` is a gate that blocks
all downstream processing.

### 2.3  ORC Routing Issues

- `_classify_query_type()` dispatches spectral queries to `[vra, ga, pa]`. 
  For Case 3 (NDBI), VRA is unnecessary — the query doesn't involve images.
  The correct sequence is `[ga, pa]` or just `[ga]` since GA owns `GetAreaBoundary`
  and VRA owns `AddIndexLayer`/`ComputeIndexChange`.
  
  **Actual issue**: The tool ownership is split: `GetAreaBoundary` is GA-only,
  but `AddIndexLayer`/`ComputeIndexChange` are VRA-only. For spectral change
  queries that require both boundary AND index computation, **both agents must
  participate** but with correct sub-task descriptions, not both trying the
  same thing.

### 2.4  `agents_output` Trace Logging

The benchmark evaluation (`_extract_pred_actions`) reads from `bundle.agents_output`.
The `_build_agents_output()` method in `episode_runner.py` only logs actions from
`trace[*].tool_calls` or `trace[*].actions_in_text`. When agents use canonical
shortcuts (which set `tool_calls=[]`), no real tool names appear, so `pred_actions`
is empty except for `Terminate` from `actions_in_text`.

### 2.5  Token Budget / max_tokens

```python
# base_agent.py line 189
"max_tokens": max(256, min(2048, self.model_max_context_tokens // 4)),
```

With `model_max_context_tokens=32768`, this gives `max_tokens=2048`. This should
be sufficient but truncation can still occur for complex JSON responses. The
`synthesize_answer()` call uses the same limit.

### 2.6  `_answer_match_with_tolerance()` Is Too Lenient

For Case 3, `answer_match=True` despite ROUGE=0.171 because the function
uses `_extract_numbers()` which picks up any numeric overlap. The GT has
numbers like `0.14`, `0.66` etc. and the pred has none, but `_extract_numbers`
finds `2019`, `2024` in both texts → numeric overlap → `match=True`.
This masks failures.

### 2.7  `_build_agents_output()` Missing Thought Traces

The function only captures `actions` not `thoughts`. For comparison with GT,
we need the full ReAct chain: thought + action per turn.

---

## 3  Fix Plan — Phased Execution

### Phase 1: Remove Canonical Shortcuts (CRITICAL)

**Goal**: Force all agents to use the LLM ReAct loop and execute real tools.

#### 1.1  `agents/ga.py`
- **DELETE** `_trial_canonical_oea()` method (lines 209-283)
- **DELETE** `_build_trial_shortcut_result()` method (lines 162-207)
- **MODIFY** `handle_task()`: Remove the `canonical = ...` call. Keep
  `_trial_restaurant_park_assignment()` temporarily since it does call real
  tools (GetAreaBoundary, AddPoisLayer, ComputeDistance) — but ensure it
  logs tool calls into the trace correctly.

#### 1.2  `agents/vra.py`
- **DELETE** the canonical hardcoded blocks inside `_trial_bbox_solver()`:
  - Lines 374-407 (garbage GSD=0.5 canonical)
  - Lines 409-449 (helicopter GSD=0.14153 canonical)
- **KEEP** the dynamic `TextToBbox` + postprocessing path (lines 451+) since
  it actually calls tools. But ensure the `tool_calls` list is populated in
  the trace.

#### 1.3  `agents/pa.py`
- **DELETE** `_trial_visual_passthrough()` method (lines 339-426)
- **MODIFY** `handle_task()`: Remove the `visual_shortcut = ...` call.
  PA should run its normal ReAct loop.

### Phase 2: Fix Agent Trace Logging for Tool Accuracy

**Goal**: Ensure `agents_output` in the bundle contains the real tool calls
made by each agent, so `_extract_pred_actions()` can produce the correct
action sequence for comparison.

#### 2.1  `episode_runner.py`: `_build_agents_output()`

Currently extracts from `trace.tool_calls` → `actions`. The issue: when a
canonical shortcut runs, `tool_calls=[]` so no actions are logged. After
removing canonicals (Phase 1), this should naturally improve. However:

- Also ensure that the `thought` is logged alongside each action for full
  ReAct trace visibility.
- Add an `observation` field from the tool output so we can see the full
  chain: thought → action → observation.

#### 2.2  Bundle structure enhancement

Each entry in `agents_output` should include:
```json
{
  "from": "gpt",
  "agent": "ga",
  "turn": 1,
  "value": "{\"thought\":\"...\",\"actions\":[{\"name\":\"GetAreaBoundary\",\"arguments\":{\"area_name\":\"Banff\"}}]}"
}
```

Currently this is correct structurally, but we need to also emit a
corresponding `observation` row per tool call.

### Phase 3: Fix Case 3 (NDBI / Spectral Change)

#### 3.1  Tool availability for spectral tasks

The GT for Case 3 is:
```
GetAreaBoundary → AddIndexLayer(2019) → AddIndexLayer(2024) → ComputeIndexChange → Terminate
```

Current tool ownership:
- `GetAreaBoundary` → GA only
- `AddIndexLayer` → VRA only
- `ComputeIndexChange` → VRA only

**Fix options**:
1. **Option A (Recommended)**: Add `AddIndexLayer` and `ComputeIndexChange` to GA's tool list.
   This way for `spectral_change` queries, GA can handle the entire chain:
   boundary → index → change.
2. **Option B**: Keep tools split but ensure ORC passes `GetAreaBoundary`
   results from GA to VRA via episodic memory. This requires VRA to actually
   use the boundary to compute indices.

#### 3.2  `TemporalStackLoader` should not be needed for NDBI

The GT doesn't use `TemporalStackLoader` — it uses `GetAreaBoundary` directly.
The NDBI is computed from GeoTIFF rasters identified by year, not from a
temporal stack. The agent just needs to:
1. Get boundary bbox
2. Use `AddIndexLayer` with `geotiff_path` pointing to year-specific rasters

The problem is that the raster files need to exist. For benchmarking, the tool
server should handle missing rasters gracefully with `success: false` and the
agent should report this rather than hallucinating.

#### 3.3  Sub-task descriptions for spectral queries

ORC should generate sub-tasks like:
- GA: "Get boundary of Nizwa Province using GetAreaBoundary with buffer"
- VRA: "Using the boundary bbox from GA, compute NDBI for 2019 and 2024
  using AddIndexLayer, then compute change using ComputeIndexChange"

### Phase 4: Fix OEA Evaluation Metrics

#### 4.1  `_answer_match_with_tolerance()` — tighten numeric matching

Current logic: if ANY extracted numbers from pred/ref overlap within 5%,
it's a match. This is too loose.

**Fix**: Require ALL key numbers to match, not just 70% of the first N
numbers. Filter out year-like numbers (2019, 2024) from the match check.

#### 4.2  `_evaluate_oea_case()` — improve action alignment

Currently does strict sequential comparison: `gt[i]` vs `pred[i]`. If the
predicted sequence has extra or skipped steps, all subsequent matches fail.

**Fix**: Use a longest common subsequence (LCS) approach for tool name
matching, similar to how ROUGE-L works. This way, if pred has an extra
intermediate step, it doesn't cascade-break all subsequent matches.

#### 4.3  Inst metric

Currently `inst_correct = 1 if pred_answer.strip() else 0`. This counts ANY
non-empty answer as correct. Should be: `1 if answer_match else 0`.

### Phase 5: Pipeline Integrity

#### 5.1  Image path propagation

Ensure `task["image_paths"]` flows correctly through:
`run_agent_pipeline_benchmarks._build_task()` → 
`episode_runner._task_payload()` → 
`message.payload["image_paths"]` → 
agent's `_build_task_adaptive_system_prompt()`

Currently this chain works (verified in code), but agents sometimes ignore
the provided paths and try to fabricate them.

#### 5.2  `max_tokens` increase

Change from:
```python
max(256, min(2048, self.model_max_context_tokens // 4))
```
To:
```python
max(512, min(4096, self.model_max_context_tokens // 4))
```

This gives 4096 tokens for a 32K context model, preventing JSON truncation
in complex responses.

#### 5.3  Thinking token stripping

Already implemented in `_strip_thinking_tokens()`. Verify it handles Qwen3's
`<think>...</think>` tags and that the stripped output is used for JSON parsing.

### Phase 6: ORC Synthesis Robustness

#### 6.1  `synthesize_answer()` edge cases

When ALL agents return errors (Case 3), the synthesized answer should:
1. Acknowledge which agents ran and what the errors were
2. State clearly that the task could not be completed
3. NOT parrot the same error from all 3 agents

Currently it does the right thing (returns the error), but the answer is
misleading for evaluation because `answer_match` says `True` due to
lenient matching.

---

## 4  Implementation Priority

```
┌─────────────────────────────────────────────┐
│ Phase 1: Remove canonical shortcuts         │ ← Do FIRST
│   - ga.py, vra.py, pa.py                   │   (blocks everything)
├─────────────────────────────────────────────┤
│ Phase 2: Fix trace logging                  │ ← Do SECOND
│   - episode_runner.py                       │   (needed for eval)
├─────────────────────────────────────────────┤
│ Phase 3: Fix Case 3 (spectral chain)        │ ← Do THIRD
│   - Tool ownership, sub-tasks               │
├─────────────────────────────────────────────┤
│ Phase 4: Fix evaluation metrics             │ ← Do FOURTH
│   - run_agent_pipeline_benchmarks.py        │
├─────────────────────────────────────────────┤
│ Phase 5: Pipeline integrity                 │ ← Do FIFTH
│   - max_tokens, image paths                 │
├─────────────────────────────────────────────┤
│ Phase 6: Synthesis robustness               │ ← LAST
│   - orc.py synthesize_answer()              │
└─────────────────────────────────────────────┘
```

---

## 5  Verification Checklist

After applying all fixes, re-run the benchmark and verify:

- [ ] **Case 0 (Banff)**: `pred_actions` includes `[GetAreaBoundary, AddPoisLayer, AddPoisLayer, ComputeDistance, Terminate]` → Tool accuracy ≥ 3/5
- [ ] **Case 1 (Garbage)**: `pred_actions` includes `[TextToBbox, Calculator, Terminate]` → Tool accuracy ≥ 2/3
- [ ] **Case 2 (Helicopter)**: `pred_actions` includes `[TextToBbox, Calculator, Terminate]` → Tool accuracy ≥ 2/3
- [ ] **Case 3 (NDBI)**: `pred_actions` includes `[GetAreaBoundary, AddIndexLayer, AddIndexLayer, ComputeIndexChange, Terminate]` → Tool accuracy ≥ 3/5 AND answer contains percentages
- [ ] **Case 4 (Assiniboine)**: `pred_actions` includes `[GetAreaBoundary, AddPoisLayer, AddPoisLayer, ComputeDistance, Terminate]` → Tool accuracy ≥ 3/5
- [ ] No `answer_match=True` for clearly wrong answers (Case 3 error message)
- [ ] Each agent's trace shows explicit thought → action → observation chain
- [ ] `agents_verbose_ordered.jsonl` contains per-turn ReAct traces

---

## 6  Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `agents/ga.py` | 1 | Remove `_trial_canonical_oea()`, `_build_trial_shortcut_result()`, update `handle_task()` |
| `agents/vra.py` | 1 | Remove canonical hardcoded blocks in `_trial_bbox_solver()`, update `handle_task()` |
| `agents/pa.py` | 1 | Remove `_trial_visual_passthrough()`, update `handle_task()` |
| `framework/episode_runner.py` | 2 | Enhance `_build_agents_output()` with thoughts+observations |
| `agents/orc.py` | 3 | Add spectral tools to GA registry, improve sub-task descriptions |
| `agents/ga.py` | 3 | Add `AddIndexLayer`, `ComputeIndexChange` to `GA_TOOLS` |
| `scripts/run_agent_pipeline_benchmarks.py` | 4 | Fix `_answer_match_with_tolerance()`, `_evaluate_oea_case()`, `Inst` metric |
| `agents/base_agent.py` | 5 | Increase `max_tokens` cap |
