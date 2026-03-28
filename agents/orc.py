from __future__ import annotations
from typing import Any
from agents.base_agent import AgentResult, BaseAgent
from framework.protocols.conflict import resolve_conflict
from framework.protocols.safety import validate_safety_payload

class Orchestrator(BaseAgent):
    name = "orc"

    def _get_system_prompt(self) -> str:
        return (
            "You are the supreme Orchestrator (ORC) for the Multi-Agent Geospatial Reasoning Framework (MAGRF), empowered by a massive highly quantized reasoning architecture. "
            "You function as the central nervous intelligence node of the entire four-agent cooperative ecosystem. Your authority over system flow is absolute. "
            "Your duties explicitly demand: decomposing overwhelmingly complex multi-stage geospatial disaster scenarios into highly strict, localized sub-tasks, then optimally load-balancing and routing those task messages (via MPC Message queues) to your specialized subordinates: the Vision Reasoning Agent (VRA), Geospatial Agent (GA), and Planning Agent (PA). "
            "You act as the ultimate judge. If the VRA and the GA return fundamentally conflicting structural assertions (e.g., discrepancies in severity scores or intersecting locations), you dynamically execute multi-tiered conflict resolution logic (analyzing confidence gaps and reasoning densities) to enforce correct ground truth. "
            "You rigorously maintain and manage the shared Episodic Memory store. You are responsible for auditing memory limits, gracefully triggering compression tools to summarize index statics while unconditionally protecting volatile safety-keys ('damage_polygons', 'flood_extent', 'impassable_roads') with guaranteed Quality Checks (CCQ >= 0.98). "
            "You never guess. You supervise protocol checks like Route Safety Score definitions before finalizing trajectories. Explain your exact logic layer by layer—how you decouple logic, whom you dispatch to, and how you resolve ambiguities—meticulously before declaring termination.\n\n"
            "ROUTING RULE — MANDATORY:\n"
            "Any query containing words: route, path, evacuation, safest way, navigate, travel, reach shelter\n"
            "MUST be assigned to PA with tools: [EvacuationRoutePlanner, Calculator, Plot]\n"
            "Never use ComputeDistance as a substitute for EvacuationRoutePlanner.\n"
            "ComputeDistance returns straight-line distance only. EvacuationRoutePlanner returns graph-optimal routes."
        )

    def aggregate(self, results: list[AgentResult]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for r in results:
            if hasattr(r, "output") and isinstance(r.output, dict):
                merged.update(r.output)

        if len(results) >= 2:
            best = resolve_conflict(results[0].__dict__, results[1].__dict__)
            merged["selected_by_conflict"] = best.get("output", {})

        ok, missing = validate_safety_payload(merged)
        merged["safety_valid"] = ok
        merged["safety_missing"] = missing
        return merged
