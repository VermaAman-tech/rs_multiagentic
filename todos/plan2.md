# MAGRF Plan 2 — All Remaining Issues and Fixes

After the first round of fixes (tasks.md all marked done), the framework still produces wrong final answers on every benchmark case. Below is the complete root-cause analysis from the latest 5-case OEA run (logs/oea_first5_full_pipeline_20260403_123951.json) plus a full codebase audit. Every issue below must be fixed for the pipeline to produce correct answers.

---

## SECTION A: ROOT CAUSES OF WRONG ANSWERS

These are the primary reasons why every final answer in the latest run is wrong.

---

### A1. Agents Cannot Produce a Final Answer — The Terminate Tool is Never Called Properly

File: agents/base_agent.py (lines 498-517, 780-793)

PROBLEM: The ReAct loop in handle_task works like this:
1. LLM generates a response
2. Code checks for tool_calls (OpenAI format) or inline tool calls in text content
3. If no tool calls found, the loop terminates immediately (line 780-793)
4. The final content is just whatever raw text the LLM last produced

The critical flaw: when agents finish reasoning and want to give a final answer, they emit a JSON with {"thought": "...", "actions": [{"name": "Terminate", "arguments": {"ans": "the answer"}}]}. The code at line 510 explicitly SKIPS Terminate actions: `if act.get("name") and act["name"] != "Terminate": return act`. So Terminate is never processed as a tool call. The loop then enters the `if not tool_calls` branch and terminates — but it never extracts the "ans" from the Terminate arguments.

The answer only gets extracted if it happens to be in the raw JSON text that gets parsed at line 857-862. But if the LLM outputs anything that isn't valid JSON, or wraps its response in markdown code blocks, or adds any text before/after the JSON, the answer is lost.

EVIDENCE from logs:
- Case 1: final_ans is "VRA evidence summary: detections={'domestic garbage': 1}; change={...}" — this is a synthetic fallback from _augment_output_from_tools, not the actual computed answer.
- Case 2: final_ans is "VRA evidence summary: detections={'helicopter': 3}" — again a synthetic fallback, not the pixel-distance computation the user asked for.
- Case 3: No final_ans at all for the NDBI change assessment.
- Case 4: No final_ans — PA computed distance=0.0 because it hallucinated coordinates.

FIX:
1. In base_agent.py _extract_inline_tool_call (line 498-517): Do NOT skip Terminate. When a Terminate action is found, extract the "ans" argument and store it. The loop should still terminate, but the answer must be captured.
2. In the main loop (line 780-793): When terminating without tool calls, explicitly parse the last response for Terminate actions and extract the "ans" field. Set it in output_dict["ans"].
3. After the loop ends (line 850-878): If output_dict has no "ans" key, scan all actions_in_text from the trace for Terminate actions and extract the answer.

---

### A2. ORC Final Answer Synthesis Is Missing — ORC Never Produces a Coherent Final Answer

File: framework/episode_runner.py (line 1289-1298), agents/orc.py (aggregate method)

PROBLEM: After all agents run, the episode_runner calls orc.aggregate() to merge results. The aggregate method (orc.py line 304-322) just does `merged.update(r.output)` for each result. This means the *last agent's* output overwrites everything, and whatever "ans" field exists in the last agent's output becomes the final answer.

But the ORC never synthesizes a final answer from the combined evidence. PA might have distance data, GA might have POI names, VRA might have detection counts — but nobody puts these together into a human-readable answer that actually answers the original question.

EVIDENCE from logs:
- Case 0 (Banff): Works because GA's _augment_output_from_tools hardcodes a closest-pair answer in ga.py line 389. This is a hardcoded hack for one specific query pattern, not a general solution.
- Case 1 (garbage area): The answer should be "X square meters" but the framework returns "VRA evidence summary: detections={'domestic garbage': 1}" — nobody computed or assembled the actual area.
- Case 3 (NDBI change): PA has no idea what to do with spectral index data. No agent synthesizes "moderate growth in zone X, decrease in zone Y."

FIX:
1. Add an ORC final answer synthesis step: After aggregating agent results, call ORC's LLM one more time with all agent outputs and the original objective. Ask it to produce a final answer in natural language that directly answers the user's question. Store this as the definitive output_dict["ans"].
2. In episode_runner.py after the aggregate step (~line 1298), add:
   ```
   final_answer = self.orc.synthesize_answer(
       objective=task.get("objective"),
       agent_results=results_by_agent,
       merged_state=final_state
   )
   final_state["ans"] = final_answer
   ```
3. In orc.py, add a synthesize_answer() method that sends all evidence to the LLM with a prompt like: "Given the following evidence from agents, provide the final answer to the question: {objective}"

---

### A3. Tool Fallbacks Return Fake Data Instead of Errors — Silent Failure Everywhere

Files: Every tool in tools/vision/, tools/spectral/, tools/novel/

PROBLEM: Every single tool implementation has an except block that returns FAKE SUCCESS data instead of an error. Examples:

- object_detection.py line 31: On exception returns {"labels": ["building"], "bboxes": [[10.0, 10.0, 20.0, 20.0]], "success": True} — a fake building detection even when the image doesn't exist.
- counting.py line 12: On exception returns {"count": 1, "success": True} — a fake count of 1.
- segmentation.py line 31: Returns fake polygon data.
- change_detection.py line 14: Returns {"changed_pixels": 100, "success": True} — fake 100 changed pixels.
- add_index_layer.py line 33: Returns {"mean_val": 0.5, "success": True} — fake index value.
- compute_index_change.py line 19: Returns {"mean_diff": 0.1, "success": True} — fake diff.
- text_to_bbox.py line 26: Returns fake bbox.
- display_map.py line 28: Returns success even on failure.
- display_geotiff.py line 15: Returns success=False but still provides a non-existent path.

EVIDENCE from logs:
- Case 1: ChangeDetection returned {"changed_pixels": 100} — this is the FALLBACK fake value (line 14 of change_detection.py). The mock epoch files don't exist as proper images.
- Case 1: ObjectDetection returned {"labels": ["building"], "bboxes": [[10.0, 10.0, 20.0, 20.0]]} — this is the FALLBACK (line 31 of object_detection.py), not a real detection of "domestic garbage".
- Case 3: AddIndexLayer returned {"mean_val": 0.5} — the FALLBACK value. The mock epoch files have no real spectral bands.
- Case 3: ComputeIndexChange returned {"mean_diff": 0.1} — the FALLBACK. Pre and post paths are fabricated by the LLM and don't exist.

The agents believe these fake results are real and reason on top of garbage data.

FIX:
1. Every tool exception handler MUST return {"success": false, "error": "actual error message"} instead of fake data.
2. Remove ALL fake-data fallbacks from every tool's except block. The full list:
   - tools/vision/object_detection.py line 31
   - tools/vision/counting.py line 12
   - tools/vision/segmentation.py line 31
   - tools/vision/change_detection.py line 14
   - tools/vision/text_to_bbox.py line 26
   - tools/vision/description.py (returns static text, not an error)
   - tools/vision/draw_box.py (check for similar pattern)
   - tools/vision/add_text.py (check for similar pattern)
   - tools/vision/ocr.py (check for similar pattern)
   - tools/vision/region_attribute.py (check for similar pattern)
   - tools/spectral/add_index_layer.py line 33
   - tools/spectral/compute_index_change.py line 19
   - tools/gis/display_map.py line 28
   - tools/gis/display_geotiff.py line 15
   - tools/novel/road_damage_scorer.py line 25
   - tools/novel/evacuation_route_planner.py line 37
3. In base_agent.py, when a tool returns {"success": false}, the agent should treat this as an error observation and reason about it in the next ReAct turn rather than treating it as valid data.

---

### A4. TemporalStackLoader Returns Mock Paths That Don't Exist

File: tools/novel/temporal_stack_loader.py

PROBLEM: When no local_stack_path is given (which is always the case in the benchmark), the tool returns ["data/mock/epoch_0.tif", "data/mock/epoch_1.tif"]. These files DO NOT EXIST. So all downstream tools (ChangeDetection, AddIndexLayer, etc.) that try to open them fail and return fake fallback data (see A3).

This creates a cascade of fake data: fake temporal stack -> fake change detection -> fake index computation -> fake answer.

FIX:
1. If the mock files don't exist, return {"success": false, "error": "No satellite data available for the requested AOI and date range. Stack files not found."}.
2. Better yet, implement actual Sentinel-2 data download using the sentinelhub or planetary_computer APIs, or at minimum use sample GeoTIFFs shipped with the repository.
3. If the user provides image_paths in the task payload, the TemporalStackLoader should look there first. The episode_runner already passes image_paths in the task payload (line 452-453) but the tool ignores them.

---

### A5. Image Paths Are Not Passed to Vision Tools

File: framework/episode_runner.py (line 452-453), agents/base_agent.py

PROBLEM: The OEA benchmark provides image paths in each task. The episode_runner includes them in the task payload as "image_paths". But when an agent calls a vision tool like ObjectDetection, it fabricates the image_path (usually "data/mock/epoch_1.tif" or similar) instead of using the actual paths from the task.

The LLM has no way to know what the correct image path is because:
1. The system prompt doesn't mention available image paths
2. The task payload includes image_paths but it's buried in the full JSON payload

EVIDENCE from logs:
- Case 1: VRA calls ObjectDetection with image_path="data/mock/epoch_1.tif" — this came from the TemporalStackLoader mock. The actual image for this case is not referenced.
- Case 2: VRA calls ObjectDetection with fabricated paths, not the actual helicopter image.

FIX:
1. In base_agent.py _build_task_adaptive_system_prompt (line 880-891): If the payload contains "image_paths", add them to the system prompt explicitly: "Available images for this task: {paths}. Use these exact paths when calling vision tools."
2. In episode_runner.py _task_payload: Ensure image_paths from the benchmark data are always passed through cleanly.
3. Each agent system prompt should instruct the LLM: "When image paths are provided in the task, use those exact paths. Do not fabricate file paths."

---

### A6. GA Retries 3 Times Due to ORC evaluate_next_step Always Returning "retry"

File: agents/orc.py evaluate_next_step (line 217-283)

PROBLEM: In evaluate_next_step, line 227-232: if latest_error is not None, ORC ALWAYS returns {"action": "retry"} without even asking the LLM. This means any agent failure triggers an immediate retry, and since the SAME error usually recurs (same context, same tool, same data), the agent retries the max number of times and wastes cycles.

EVIDENCE from logs:
- Case 2: dispatched = ['vra', 'ga', 'ga', 'ga', 'pa', 'pa', 'pa'] — GA ran 3 times, PA ran 3 times
- Case 3: dispatched = ['vra', 'ga', 'ga', 'ga', 'pa'] — GA ran 3 times
- Case 4: dispatched = ['ga', 'ga', 'ga', 'pa', 'pa', 'pa'] — GA ran 3 times, PA ran 3 times

GA fails because it has no useful work to do for these queries (it tries GIS ops on non-existent data), gets an error, ORC blindly retries it.

FIX:
1. Remove the hardcoded retry-on-error at line 227-232. Instead, always consult the LLM to decide whether retry makes sense.
2. Before retrying, check if the error is recoverable. A timeout on AddPoisLayer is not recoverable by retry. A context overflow might be recoverable after compression.
3. Pass the actual error text to ORC's LLM so it can make an informed decision.
4. Add a "retry_count" to the payload so ORC knows if this is the 1st or 3rd retry and can decide to skip instead.

---

### A7. PA Hallucinates Coordinates When It Has No GA Evidence

File: agents/pa.py (line 336-344), base_agent.py

PROBLEM: When PA receives a proximity query but has no GA evidence in episodic_context (because GA failed or didn't produce POI data), PA falls through to the LLM-based handle_task. The LLM then hallucinates coordinates and calls ComputeDistance with fabricated lat/lon values.

EVIDENCE from logs:
- Case 4: PA calls ComputeDistance with point_a=[49.8992, -97.1355] and point_b=[49.8992, -97.1355] — SAME point. Result: 0.0 meters. The LLM fabricated coordinates and happened to use the same ones for both points.
- This is because GA's AddPoisLayer timed out (restaurants in Assiniboine Park), so PA had no real data to work with and hallucinated.

FIX:
1. In PA's system prompt, add explicit instructions: "NEVER fabricate coordinates. If you don't have real coordinate data from GA or the task context, state that the data is unavailable rather than inventing values."
2. In PA's handle_task: If it's a proximity query and there's no GA evidence AND no episodic context with coordinates, return immediately with {"ans": "Unable to determine — insufficient geospatial data from upstream agents.", "confidence": 0.1} instead of letting the LLM hallucinate.
3. In the _augment_output_from_tools for PA: If ComputeDistance was called with the same point for both a and b (distance=0), flag this as "degenerate_computation" and don't use it as real evidence.

---

### A8. max_tokens Is Set Absurdly Low — LLM Outputs Get Truncated

File: agents/base_agent.py line 145

PROBLEM: The max_tokens for LLM generation is set to `max(128, min(1024, self.model_max_context_tokens // 8))`. With model_max_context_tokens=32768, this gives max_tokens=1024. But the LLM needs to output structured JSON with thought + actions + arguments, which can easily exceed 1024 tokens — especially for complex tool calls with many arguments, or for final answers that need to be detailed.

When the output is truncated mid-JSON, the response can't be parsed, and the agent falls through to the no-tool-call termination path with no useful answer.

FIX:
1. Change max_tokens to at least 2048, or better yet 4096: `max(256, min(4096, self.model_max_context_tokens // 4))`
2. This gives the LLM enough room to produce complete JSON responses with full reasoning.

---

### A9. ORC query_type Classification Misroutes Many Queries

File: agents/orc.py _classify_query_type (line 52-64)

PROBLEM: The keyword-based classification is too narrow and misclassifies many queries:

- Case 1 "Detect all domestic garbage regions and calculate their combined area in square meters" — has "detect" so matches "visual_qa", but this also needs spatial computation (area calculation). It should route VRA -> PA, not VRA -> PA (which is correct) but with proper sub-tasks.
- Case 2 "Detect the helicopters in the aerial image, measure the pixel distance between them, and convert those distances to meters" — classified as "visual_qa" but needs VRA for detection AND PA for distance math. The sub-tasks sent to each agent need to be much more specific.
- Case 3 "Assess moderate and strong urban growth or decrease in Nizwa Province, Oman between 2019 and 2024 using the NDBI change layer" — has "urban growth" which triggers "spectral_change", routing to GA -> PA. But this actually ALSO needs VRA for spectral index computation (AddIndexLayer, ComputeIndexChange). The VRA gets excluded.
- Case 4 "Assign each restaurant in Assiniboine Park to its nearest park" — matches "nearest" -> "geospatial_proximity", routing GA -> PA. This is correct in terms of agents but the sub-tasks are wrong.

But also: even when query_type is "spectral_change", line 97-98 routes to [ga, pa] only — VRA is excluded. But VRA owns AddIndexLayer and ComputeIndexChange! So spectral queries can never work.

FIX:
1. Fix spectral_change routing: include VRA. Change line 97-98 from `seq = [a for a in ["ga", "pa"] if a in available_agents]` to `seq = [a for a in ["vra", "ga", "pa"] if a in available_agents]`
2. Make the LLM-based planner more reliable by giving it better instructions about which agent owns which tools. The _AGENT_CAPABILITIES dict (line 12-16) should list actual tool names, not vague descriptions.
3. The fallback classification should be just that — a fallback. The LLM planner should usually override it. If the LLM planner fails to return valid JSON, the fallback is OK but should be more inclusive (default to all 3 agents rather than leaving VRA out for spectral queries).

---

### A10. GA and PA Get Dispatched for Queries They Cannot Help With

File: framework/episode_runner.py, agents/orc.py

PROBLEM: In cases 1 and 2 (garbage detection, helicopter detection), GA is dispatched to do "geospatial analysis" but there's nothing geospatial to analyze — these are pure vision tasks. GA then wastes turns calling GetBboxFromGeotiff on non-existent data and AddPoisLayer for "garbage" POIs (which times out because OSM doesn't have garbage POIs).

Similarly, PA is dispatched for case 2 (helicopter pixel distances) but PA has no context to work with because VRA's tool outputs are mock data and GA contributed nothing useful.

FIX:
1. Improve ORC sub-task descriptions: Instead of generic "Run GIS analysis for: {objective}", the sub-task should be specific: "Extract boundary of the area shown in the image using GetBboxFromGeotiff" or "No GIS analysis needed for this task."
2. ORC should be able to decide NOT to dispatch an agent. If the plan says agent_sequence=["vra", "pa"], then only VRA and PA should run. Currently, even when ORC plans correctly, GA sometimes gets added due to fallback logic.
3. Add capability matching to _default_plan: For visual_qa queries, only dispatch VRA (and PA if math is needed). Don't dispatch GA unless the query explicitly involves geographic areas, boundaries, or POIs.

---

## SECTION B: CRITICAL SYSTEM PROMPT ISSUES

---

### B1. Agent System Prompts Lack Concrete Instructions

Files: agents/vra.py _get_system_prompt, agents/ga.py _get_system_prompt, agents/pa.py _get_system_prompt

PROBLEM: The system prompts are written as high-level policy documents, not as actionable instructions for an LLM. They say things like "Convert place-based requests into verifiable geospatial operations" and "THINK: determine required geospatial entities" but never tell the LLM the actual output format or how to structure its responses.

Key missing instructions:
1. NO OUTPUT FORMAT SPECIFICATION: The prompts never say "Output your response as JSON with keys: thought, actions". The LLM has to guess the format, and with Qwen3-4B it often guesses wrong.
2. NO TOOL USAGE EXAMPLES: The prompts list tool names but never show example calls. The LLM doesn't know what arguments to pass.
3. NO TERMINATION INSTRUCTIONS: The prompts never say "When you have enough evidence to answer, call the Terminate tool with {"name": "Terminate", "arguments": {"ans": "your final answer"}}"
4. NO DATA GROUNDING: The prompts never say "Use exact data from tool outputs. Never fabricate coordinates, counts, or measurements."

FIX: Rewrite all system prompts to include:

For ALL agents:
```
OUTPUT FORMAT:
You must output valid JSON with this structure:
{"thought": "your reasoning about what to do next", "actions": [{"name": "ToolName", "arguments": {"arg1": "value1"}}]}

When you have gathered sufficient evidence to answer, use:
{"thought": "final reasoning", "actions": [{"name": "Terminate", "arguments": {"ans": "your complete answer to the question"}}]}

CRITICAL RULES:
- Call exactly ONE tool per turn
- Always include a "thought" explaining your reasoning
- NEVER fabricate data — only use values from tool outputs
- If a tool returns an error, explain what went wrong and try a different approach
- When available image paths are provided, use those EXACT paths
```

For VRA specifically, add:
```
Available images for this task will be in the task payload under "image_paths".
Always use these exact paths when calling vision tools.
For detection tasks: call ObjectDetection or TextToBbox with the image path.
For counting tasks: call CountGivenObject with the image path.
For area computation: first detect objects, then use SegmentObjectPixels to get pixel masks, then compute area from mask pixel count and GSD.
For spectral analysis: use AddIndexLayer to compute indices, ComputeIndexChange for temporal comparison.
```

For GA specifically, add:
```
For boundary queries: use GetAreaBoundary with the place name (and buffer_m if specified).
For POI queries: first get the area boundary to get the bbox, then call AddPoisLayer with the bbox.
For distance queries: use the POI coordinates from AddPoisLayer results and call ComputeDistance.
For closest-pair queries: call AddPoisLayer for each category, then compute pairwise distances.
```

For PA specifically, add:
```
When GA evidence is available in episodic_context, use it directly — do not re-query.
For distance verification: use ComputeDistance with exact coordinates from GA's POI data.
For route planning: use EvacuationRoutePlanner with the road graph.
For math computations: use Calculator with the appropriate expression.
NEVER make up coordinates. If coordinates are not available from GA or the task, state that data is insufficient.
```

---

### B2. ORC System Prompt Doesn't Instruct It to Route Data Between Agents

File: agents/orc.py _get_system_prompt (line 18-50)

PROBLEM: ORC's prompt talks about "MPC governance" and "selective context keys" but never tells the LLM HOW to pass data between agents. Specifically:
- When GA finds POIs, how does ORC ensure PA gets those POI coordinates?
- When VRA detects objects, how does GA know what was detected?
- The prompt says "send SYNC and TASK to PA" but doesn't explain what data PA needs.

FIX: Add to ORC's system prompt:
```
DATA ROUTING BETWEEN AGENTS:
- VRA outputs (detections, counts, change maps) should be summarized in PA's sub-task.
- GA outputs (POIs, boundaries, distances) are automatically passed to PA via episodic_context.
- When constructing sub-tasks, include specific data that the agent will need, e.g.:
  "GA has found the following fire stations: [list]. Compute pairwise distances to police stations."
```

But also: the actual data routing happens through episodic memory, not through sub-task text. The real fix is in Section C below.

---

## SECTION C: CONTEXT AND MEMORY FLOW ISSUES

---

### C1. Episodic Memory Context Is Not Rich Enough for PA

File: framework/episode_runner.py _context_keys_for (line 391-418)

PROBLEM: The _context_keys_for("pa") method looks for keys matching prefixes like "vra_damage_polygons", "ga_road_scores", etc. But the actual keys stored in episodic memory after GA runs are things like:
- "ga_result" (the full output dict)
- "ga_raw_output"
- "ga_closest_pair"
- "ga_fire_pois"
- "ga_police_pois"
- "ga_distance_meters"

The preferred key list in _context_keys_for("pa") includes "ga_result" which IS correct. But the fallback _summarize_context_value (line 290-325) strips out raw_output, trace, and tool_calls — so the actual POI data and distance measurements ARE available to PA.

However, the issue is that when GA produces fake/empty data (because tools returned fallbacks), the episodic context PA receives is also fake/empty. This is fundamentally the A3 problem cascading.

FIX:
1. After GA completes, verify that its output contains actual useful data before passing to PA. If ga_result has no real POIs (empty lists, fake fallback data), log a warning and tell PA explicitly: "GA produced no valid output for this query."
2. When constructing episodic_context for PA, include ALL ga_ and vra_ prefixed keys, not just a hardcoded list. The current filter is too narrow.
3. Change the fallback in _context_keys_for("pa") from line 417 to include all memory keys that start with ga_ or vra_, not just selected prefixes.

---

### C2. Working Memory Is Populated But Never Read by Agents

File: framework/episode_runner.py _update_working_memory (line 251-262)

PROBLEM: After each agent runs, _update_working_memory stores a summary in the working memory buffer. But agents never READ this working memory. The working memory content is flushed to episodic memory on overflow (line 258-262), but it's stored under a key like "ga_working_memory_flush" which is not in the _context_keys_for() preferred list for any agent.

FIX: Either:
1. Remove working memory entirely (it adds complexity without benefit in the current single-process architecture), OR
2. Actually inject working memory content into agent prompts. When building the system prompt, append the agent's working memory summary.

---

## SECTION D: TOOL-SPECIFIC ISSUES

---

### D1. AddPoisLayer Times Out for Many Valid Queries

File: tools/gis/pois_layer.py, configs/tools.yaml

PROBLEM: AddPoisLayer has a 15-second timeout in the config. But OSM queries for large areas or uncommon POI categories can take 30+ seconds. In the logs, Case 4 (restaurants in Assiniboine Park) timed out because OSM was slow.

But the bigger issue: the timeout is in the config but the tool server doesn't enforce it. The timeout is enforced by base_agent.py's httpx.post at line 488 with timeout=60.0. The actual tool-level timeout isn't used at all.

Additionally, pois_layer.py queries OSM synchronously. If the network is slow or OSM rate-limits, the query hangs.

FIX:
1. Increase the httpx timeout for AddPoisLayer to 90 seconds in base_agent.py, or make the tool server enforce per-tool timeouts from tools.yaml.
2. Add retry logic in pois_layer.py: if the first query fails, try with a smaller bbox or different query strategy.
3. In _query_to_osm_tags, add "restaurant" mapping: `if "restaurant" in q: return {"amenity": ["restaurant", "fast_food", "cafe"]}`.
4. Add "park" mapping: `if "park" in q: return {"leisure": ["park", "garden", "nature_reserve"]}`.

---

### D2. GetBboxFromGeotiff Returns [0,0,1,1] for Non-Existent Files

File: tools/gis/bbox_from_geotiff.py

PROBLEM: Check the implementation — it likely returns a default bbox when the file doesn't exist, instead of an error. This means agents think they have valid GIS bounds when they don't.

FIX: Return {"success": false, "error": "File not found"} when the GeoTIFF path doesn't exist.

---

### D3. ImageDescription Returns Static Text

File: tools/vision/description.py

PROBLEM: The tool always returns {"description": "Image loaded into VRA context. VRA can now describe it directly."} regardless of input. This is useless — the agent gets no actual image content.

The comment says "VRA reads the image using its Qwen3-VL core inside the ReAct loop" — but VRA's LLM is text-only (Qwen3-4B-Instruct or Qwen3-VL-4B-Instruct). If using the VL model, images should be passed as multimodal inputs. If using the text-only model, ImageDescription needs to actually process the image.

FIX: Implement real image description using a vision-language model (BLIP-2, LLaVA, or the Qwen3-VL model that's already configured). At minimum, return image metadata (size, number of bands, histogram statistics).

---

### D4. DisplayOnMap Returns HTML as html_path

File: tools/gis/display_map.py

PROBLEM: In the logs, Case 3 shows: DisplayOnMap returned {"html_path": "<html><body><h1>Nizwa Province Boundary</h1>..."}. The html_path field contains the ACTUAL HTML CONTENT, not a file path. This happens because the tool sets output_html from the request, and when the LLM passes HTML directly as the output destination, the tool writes it as a filename (which creates a directory with the HTML as name — visible in the project root as "<html><body><h1>Nizwa Province Boundary<" directory).

FIX: 
1. In display_map.py, always generate the output path internally: `out_path = f"data/tmp/map_{uuid4().hex[:8]}.png"`. Don't use the request's output_html as a filename.
2. Clean up the malformed directories in the project root.

---

### D5. DisplayOnGeotiff Uses the Wrong Function

File: tools/server.py line 167-174

PROBLEM: Both DisplayGeotiff and DisplayOnGeotiff routes call `display_geotiff.run(req)`. But there's a separate `display_on_geotiff.py` module that's imported but never used (it has a `display_on_geotiff()` function, not a `run()` function). The server.py imports it but the route doesn't use it.

FIX: Either consolidate into one implementation or wire DisplayOnGeotiff to use display_on_geotiff.display_on_geotiff().

---

### D6. PrithviEmbed Is a Stub

File: tools/novel/prithvi_embed.py

This is just `return {"embedding_dim": 768, "summary_stats": {}}`. It doesn't actually compute Prithvi embeddings. If VRA ever calls it, it gets useless data.

FIX: Implement actual Prithvi embedding computation using the ibm-nasa-geospatial/Prithvi-EO-2.0-300M model, or return a clear error indicating the model isn't loaded.

---

## SECTION E: LLM INTERACTION ISSUES

---

### E1. Qwen3 Thinking Mode Tokens Not Handled

File: agents/base_agent.py

PROBLEM: Qwen3 models with thinking_mode="always" output <think>...</think> tokens before their actual response. The code doesn't strip these thinking tokens. If the LLM outputs:
```
<think>I need to find fire stations...</think>
{"thought": "...", "actions": [...]}
```
Then json.loads on the full content fails because of the <think> prefix. The response falls through to the plain-text path and no tool gets called.

FIX:
1. In _extract_thought_actions and _extract_inline_tool_call, strip any <think>...</think> wrapper before attempting JSON parse.
2. Add a method like:
```python
def _strip_thinking_tokens(self, content: str) -> str:
    import re
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
```
3. Apply it before JSON parsing in the main loop.

---

### E2. The LLM Is Not Told What Format to Output

File: agents/base_agent.py _build_task_adaptive_system_prompt (line 880-891)

PROBLEM: The base prompt says "Use strict ReAct loops: Think, call one tool at a time, observe output, then decide next action" but never shows the OUTPUT FORMAT. The Qwen3-4B model often outputs plain text or partial JSON or markdown-wrapped JSON instead of clean JSON.

FIX: Add to every system prompt:
```
RESPONSE FORMAT (mandatory): 
Your response MUST be a single JSON object:
{"thought": "your reasoning", "actions": [{"name": "ToolName", "arguments": {"key": "value"}}]}

To finalize, use:
{"thought": "final reasoning", "actions": [{"name": "Terminate", "arguments": {"ans": "your complete answer"}}]}

Do NOT wrap in markdown code blocks. Do NOT add any text outside the JSON.
```

---

### E3. VRA Uses Wrong Model Port by Default

File: configs/models.yaml

PROBLEM: VRA is configured with vllm_port: 8001 and model: Qwen3-VL-4B-Instruct. GA/PA/ORC use port 8002 with Qwen3-4B (text-only). In the audit log (banff_full_pipeline_audit_postfix_r2.md), VRA port 8001 shows "FAIL". This means VRA falls back to the text-only port 8002 or uses mock responses.

If VRA uses the text-only Qwen3-4B instead of Qwen3-VL-4B, it cannot actually see images — so all vision tool usage is based on guessing, not actual image understanding.

FIX: 
1. Ensure VRA's vLLM instance is running on port 8001 with the VL model.
2. If VRA detects that port 8001 is unavailable, log a clear ERROR, not a silent fallback.
3. When using the VL model, image_paths from the task should be passed as multimodal content in the LLM messages. Currently images are only passed as file paths in tool arguments, never as direct visual input to the LLM.

---

## SECTION F: EPISODE RUNNER FLOW ISSUES

---

### F1. ORC Plan Is Good But Not Enforced Strictly

File: framework/episode_runner.py (line 900-908, 1166-1217)

PROBLEM: ORC generates a plan with agent_sequence, but the dynamic loop at line 1166-1217 can deviate from the plan based on evaluate_next_step decisions. Specifically, evaluate_next_step can reorder agents or skip them. Combined with the blind retry logic (A6), the actual dispatched sequence often doesn't match the plan.

FIX: 
1. The dynamic loop should follow the plan more faithfully. Retries should rerun the same agent, not reorder the queue.
2. When ORC says "skip", truly skip the agent without adding it back.
3. Log a warning if the final dispatched sequence deviates significantly from the plan.

---

### F2. GA Runs With Wrong Sub-Tasks for Vision-Only Queries

File: framework/episode_runner.py, agents/orc.py

PROBLEM: When ORC plans correctly for a vision query (e.g., sequence=["vra", "pa"]), GA shouldn't be in the sequence. But the current code at line 906-908 filters the sequence to only include ["vra", "ga", "pa"], and if the LLM plan is empty (JSON parse failure), the fallback includes all three agents.

Also, even when the LLM produces a correct plan, the _default_plan fallback at line 93-118 is used to MERGE sub-tasks (line 188-192). This means even if the LLM says "only use vra and pa", the merged sub-tasks dict still has entries for ga.

FIX:
1. If the LLM plan says agent_sequence=["vra", "pa"], only GA's tasks should be excluded entirely, and GA should not be dispatched.
2. The resolved_sub_tasks (line 203-208) should only contain entries for agents in the actual sequence.

---

### F3. Multiple Agents Share the Same LLM Port — Serialized Inference

File: configs/models.yaml

PROBLEM: GA, PA, and ORC all use port 8002 (same vLLM instance). This means all LLM calls are serialized (vLLM handles one request at a time for small models). Each agent turn takes 2-3 seconds of LLM time, and with 4+ turns per agent and 3 agents, a single episode takes 30-60+ seconds.

This isn't a correctness issue but affects the usability and makes experiments slow.

FIX: If resources permit, use separate vLLM ports per agent. Otherwise, ensure the shared instance handles concurrent requests properly (increase vLLM gpu_memory_utilization or tensor parallelism).

---

## SECTION G: ANSWER EXTRACTION AND EVALUATION

---

### G1. No Standard Answer Extraction for Evaluation

File: scripts/run_agent_pipeline_benchmarks.py

PROBLEM: The benchmark runner stores the entire result dict, not a clean extracted answer. To evaluate against ground truth, there needs to be a consistent "final_answer" field. Currently:
- Some agents produce "ans" in their output
- The merged state may or may not have "ans"
- The benchmark script doesn't extract or compare answers

FIX:
1. In episode_runner.py, after the final answer synthesis (A2), ensure result["ans"] is always a clean string.
2. In the benchmark script, extract the final answer and compare against ground truth using:
   - Exact match for factual answers
   - ROUGE-L for descriptive answers
   - Numeric tolerance for distance/area/count answers
3. Add scoring to the benchmark manifest.

---

### G2. OEA Evaluation Metrics Are Not Computed From Agent Actions

File: scripts/run_agent_pipeline_benchmarks.py, evaluation/run_eval.py

PROBLEM: The OEA benchmark metrics (Inst, Tool, ArgN, ArgV, Summ) require comparing agent actions against ground-truth action sequences. The current code doesn't do this — it only checks if the run completed successfully.

FIX: Implement actual OEA metric computation:
1. Parse the ground-truth action sequences from the OEA test.json
2. Compare against the agents_output action sequences from the run bundle
3. Compute Inst (instruction-following), Tool (correct tool usage), ArgN (correct argument names), ArgV (correct argument values), Summ (answer summary quality)

---

## SECTION H: CONCRETE FILE-BY-FILE CHANGE LIST

Below is the exact set of changes needed, organized by file:

### agents/base_agent.py
1. Line 145: Change max_tokens from `min(1024, ...)` to `min(4096, self.model_max_context_tokens // 4)` to prevent output truncation.
2. Lines 498-517 (_extract_inline_tool_call): Remove the `act["name"] != "Terminate"` filter. When a Terminate action is found, return it so the answer can be captured. Add a flag like `is_terminate=True` so the caller knows.
3. Lines 519-530 (_extract_thought_actions): Add thinking token stripping: strip `<think>...</think>` from content before parsing JSON.
4. Lines 780-793 (no tool_calls branch): When terminating, explicitly check for Terminate actions in the parsed content and extract the "ans" field into a variable.
5. Lines 850-878 (final extraction): If output_dict has no "ans", scan trace for Terminate actions. Also: if the raw content starts with `<think>`, strip it before JSON parsing.
6. Lines 880-891 (_build_task_adaptive_system_prompt): Add the response format instructions (JSON format with thought+actions). Add image_paths if present in payload. Add "never fabricate data" instruction.

### agents/orc.py
1. Lines 12-16 (_AGENT_CAPABILITIES): Replace vague descriptions with actual tool lists. E.g., "vra": "ObjectDetection, CountGivenObject, SegmentObjectPixels, ImageDescription, ChangeDetection, AddIndexLayer, ComputeIndexChange, OCR, TextToBbox, DrawBox, TemporalStackLoader"
2. Lines 97-98 (_default_plan spectral_change): Include VRA in spectral_change sequence: `seq = [a for a in ["vra", "ga", "pa"] if a in available_agents]`
3. Lines 227-232 (evaluate_next_step): Remove the hardcoded retry-on-error. Move error handling into the LLM call so ORC can make an informed decision.
4. Add a new synthesize_answer() method that takes the objective and all agent results, calls the LLM, and produces a final natural-language answer.

### agents/ga.py
1. System prompt: Add concrete tool usage instructions and output format specification (see B1).
2. System prompt: Add "never fabricate coordinates" instruction.

### agents/pa.py
1. System prompt: Add concrete tool usage instructions and output format specification (see B1).
2. System prompt: Add "never fabricate coordinates" instruction.
3. Lines 336-344 (handle_task): When proximity query has no GA evidence and no usable episodic context, return immediately with an honest "insufficient data" answer instead of falling through to LLM which will hallucinate.

### agents/vra.py
1. System prompt: Add concrete tool usage instructions and output format specification (see B1).
2. System prompt: Add instructions about using image_paths from the task payload.

### framework/episode_runner.py
1. After line ~1298 (after final aggregate): Add ORC final answer synthesis step — call orc.synthesize_answer() and set final_state["ans"].
2. Lines 391-418 (_context_keys_for): For PA, include ALL ga_ and vra_ prefixed keys as fallback, not just a narrow list.
3. Lines 1166-1217 (dynamic loop): When ORC says "retry" but retry_count is already at max, force "skip" instead of continuing the loop with the same agent.

### tools/vision/object_detection.py
1. Line 31: Change except block to return {"labels": [], "bboxes": [], "success": false, "error": str(e)}

### tools/vision/counting.py
1. Line 12: Change except block to return {"count": 0, "success": false, "error": str(e)}

### tools/vision/segmentation.py
1. Line 31: Change except block to return {"polygons": [], "success": false, "error": str(e)}

### tools/vision/change_detection.py
1. Line 14: Change except block to return {"change_map_path": "", "changed_pixels": 0, "success": false, "error": str(e)}

### tools/vision/text_to_bbox.py
1. Line 26: Change except block to return {"bboxes": [], "success": false, "error": str(e)}

### tools/vision/description.py
1. Replace the static response with actual image metadata extraction (at minimum: image size, bands, basic stats).

### tools/spectral/add_index_layer.py
1. Line 33: Change except block to return {"index_array_path": "", "mean_val": 0.0, "success": false, "error": str(e)}

### tools/spectral/compute_index_change.py
1. Line 19: Change except block to return {"diff_path": "", "mean_diff": 0.0, "success": false, "error": str(e)}

### tools/novel/temporal_stack_loader.py
1. Line 15: When returning mock paths, check if the files exist first. If not, return {"stack_paths": [], "n_epochs": 0, "success": false, "error": "No satellite data available for the requested AOI and date range"}

### tools/novel/evacuation_route_planner.py
1. Line 37: Change except block to return real error.

### tools/novel/road_damage_scorer.py
1. Line 25: Change except block to return real error.

### tools/gis/pois_layer.py
1. In _query_to_osm_tags, add mappings for: "restaurant", "park", "bank", "hotel", "supermarket", "parking", "bus", "train", "airport".
2. Each map should have the appropriate OSM tags.

### tools/gis/display_map.py
1. Line 22: Generate output path internally instead of using req.output_html. Don't let user-provided HTML content become a filename.

### tools/gis/bbox_from_geotiff.py
1. Check if file exists before opening. Return error if not found.

### tools/schemas.py
1. Add "success" and "error" optional fields to ALL output schemas that don't have them (ObjectDetectionOutput, CountGivenObjectOutput, SegmentObjectPixelsOutput, ChangeDetectionOutput, AddIndexLayerOutput, ComputeIndexChangeOutput, ShowIndexLayerOutput, TemporalStackLoaderOutput, RoadDamageScorerOutput, EvacuationRoutePlannerOutput, PrithviEmbedOutput).

---

## EXECUTION PRIORITY

1. PHASE 1 (Stop producing garbage data):
   - Fix all tool fallbacks (A3) — return errors instead of fake data
   - Fix TemporalStackLoader mock paths (A4)
   - Fix max_tokens truncation (A8)
   - Strip thinking tokens (E1)
   - Add output format to system prompts (E2, B1)

2. PHASE 2 (Get correct answers):
   - Fix Terminate answer extraction (A1)
   - Add ORC final answer synthesis (A2)
   - Fix PA hallucination (A7)
   - Fix image paths passing (A5)

3. PHASE 3 (Route queries correctly):
   - Fix spectral_change routing to include VRA (A9)
   - Fix ORC blind retry (A6)
   - Fix GA dispatch for vision-only queries (A10)
   - Pass richer context to PA (C1)

4. PHASE 4 (Polish):
   - Fix pois_layer categories (D1)
   - Fix display_map path issue (D4)
   - Fix DisplayOnGeotiff wiring (D5)
   - Implement answer comparison metrics (G1, G2)
   - Fix ImageDescription (D3)
   - Fix PrithviEmbed (D6)
