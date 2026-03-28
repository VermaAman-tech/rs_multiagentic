ABLATIONS = {
    'A1_no_deadlock': {'deadlock_enabled': False},
    'A2_no_conflict': {'conflict_resolution': False},
    'A3_no_compression': {'compression': False},
    'A4_no_prithvi': {'use_prithvi': False},
    'A5_single_agent': {'agents': ['orc']},
    'A6_no_react': {'react': False},
    'A7_no_memory': {'episodic_memory': False},
    'A8_no_novel_tools': {
        "name": "A8_no_novel_tools",
        "description": "Remove N1/N2/N3/N4. Use only original 24 tools.",
        "disabled_tools": ["TemporalStackLoader", "RoadDamageScorer",
                           "EvacuationRoutePlanner", "PrithviEmbed"],
        "fallback_tools": {
            "EvacuationRoutePlanner": "ComputeDistance",
            "RoadDamageScorer": None,
            "TemporalStackLoader": "ChangeDetection",
        }
    },
}
