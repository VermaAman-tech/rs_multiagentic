# Plan 4: Restore Multi-Agent Pipeline Integrity

## Problem Statement

The MAGRF pipeline in eval_029 shows **0% instruction accuracy** and **0% tool accuracy** because agents bypass
the ReAct loop via hardcoded canonical shortcut methods. Additionally, the Terminate answer is redacted in
`agents_output`, causing the evaluation extractor to find an empty predicted answer. The spectral change
pipeline (Case 3: NDBI) fails because ORC misroutes spectral tasks and the TemporalStackLoader blocks.

## Root Causes

| # | Root Cause | Impact | File(s) |
|---|-----------|--------|---------|
| 1 | `GA._trial_restaurant_park_assignment()` bypasses LLM ReAct loop with hardcoded logic | 0 LLM turns for restaurant-park cases, only `[Terminate]` action trace | `agents/ga.py` L193-347, L352-354 |
| 2 | `VRA._trial_bbox_solver()` bypasses LLM ReAct loop for garbage/helicopter tasks | Same as above for vision tasks | `agents/vra.py` L362-532, L539-550 |
| 3 | `_sanitize_logged_action()` redacts Terminate answer in `agents_output` | Eval extractor sees `<redacted>` → empty pred_answer → always 0% Inst | `framework/episode_runner.py` L586-599 |
| 4 | ORC spectral routing inconsistency | Spectral tasks routed to GA but VRA also owns spectral tools, causing confusion | `agents/orc.py`, `configs/agents.yaml` |
| 5 | `max_turns=4` in eval command too low for multi-step ReAct chains | Agents hit turn cap before completing tool chains like GetAreaBoundary→AddPoisLayer→ComputeDistance | CLI default in eval script |
| 6 | Mock fallback returns fake data when LLM is unreachable | Hides connection failures behind `"mock result"` answers | `agents/base_agent.py` L365-392 |
| 7 | `_answer_match_with_tolerance` ROUGE-L threshold too lenient at 0.75 | False positives inflate Inst metric | `scripts/run_agent_pipeline_benchmarks.py` L367 |

## Proposed Changes (6 Phases)

---

### Phase 1: Remove Canonical Shortcuts (Critical)

#### [MODIFY] [ga.py](file:///mnt/media1/maram/rs_multiagentic_project/rs_multiagentic/agents/ga.py)
- **DELETE** `_trial_restaurant_park_assignment()` method (L193-347)
- **MODIFY** `handle_task()` (L349-355) → remove shortcut call, always delegate to `super().handle_task()`

#### [MODIFY] [vra.py](file:///mnt/media1/maram/rs_multiagentic_project/rs_multiagentic/agents/vra.py)
- **DELETE** `_trial_bbox_solver()` method (L362-532)
- **MODIFY** `handle_task()` (L534-552) → remove shortcut calls, always delegate to `super().handle_task()`

---

### Phase 2: Fix Terminate Answer Redaction in Trace (Critical)

#### [MODIFY] [episode_runner.py](file:///mnt/media1/maram/rs_multiagentic_project/rs_multiagentic/framework/episode_runner.py)
- **MODIFY** `_sanitize_logged_action()` (L586-599): Stop redacting the `ans` key in Terminate actions
  - The current logic replaces all answer keys with `<redacted>` to avoid "leaking" ground truth in logs
  - But this breaks the evaluation extractor which reads `agents_output` → `gpt` turns → `actions` → Terminate args
  - Fix: Remove the redaction entirely, or only redact when the answer literally matches the ground truth

---

### Phase 3: Fix ORC Spectral Routing and Tool Ownership

#### [MODIFY] [orc.py](file:///mnt/media1/maram/rs_multiagentic_project/rs_multiagentic/agents/orc.py)
- GA already has AddIndexLayer + ComputeIndexChange in its GA_TOOLS, and ORC already routes spectral→GA
- The actual failure was TemporalStackLoader blocking (which GA doesn't use), so this is already correct
- Verify `_AGENT_CAPABILITIES` dictionary includes AddIndexLayer/ComputeIndexChange for GA

#### [MODIFY] [configs/agents.yaml](file:///mnt/media1/maram/rs_multiagentic_project/rs_multiagentic/configs/agents.yaml)
- **ADD** `AddIndexLayer` and `ComputeIndexChange` to GA's tool list (currently missing from YAML, though present in GA_TOOLS code)

---

### Phase 4: Pipeline Robustness

#### [MODIFY] [base_agent.py](file:///mnt/media1/maram/rs_multiagentic_project/rs_multiagentic/agents/base_agent.py)
- **MODIFY** `_call_llm()` `max_tokens` formula (L207): Increase from `max(512, min(4096, ...))` to `max(1024, min(8192, ...))`
- **MODIFY** mock fallback (L378-392): When `allow_mock_fallback=False`, raise error; when True, log clearly

#### [MODIFY] [run_agent_pipeline_benchmarks.py](file:///mnt/media1/maram/rs_multiagentic_project/rs_multiagentic/scripts/run_agent_pipeline_benchmarks.py)
- **MODIFY** default `--max-turns` from 4 to 15 (L435)
- **MODIFY** `_answer_match_with_tolerance` ROUGE-L threshold from 0.75 to 0.50 (L367) for stricter matching

---

### Phase 5: Enhanced Trace Logging

- `_build_agents_output()` already captures per-turn thought/actions/observations
- Ensure the `_sanitize_logged_action` fix from Phase 2 propagates clean Terminate answers

---

### Phase 6: Run Evaluation

- Run 20-case OEA evaluation with `--max-turns 15` and `--strict-no-mock-fallback`
- Verify: Inst > 0%, Tool > 0%, action sequences contain real tool calls not just `[Terminate]`

## Verification Plan

### Automated Tests
```bash
python scripts/run_agent_pipeline_benchmarks.py \
  --oea-limit 20 \
  --thinkgeo-limit 0 \
  --max-turns 15 \
  --strict-no-mock-fallback
```

### Manual Verification
- Check `manifest.json` → each case shows `pred_actions` with real tool names
- Check `oea_eval` → `tool_correct > 0` for any case
- Check `final_answer` → not empty, not `<redacted>`
