from framework.protocols.conflict import resolve_conflict


def test_conflict_resolve_by_confidence():
    a = {"output": "a", "confidence": 0.9}
    b = {"output": "b", "confidence": 0.5}
    assert resolve_conflict(a, b)["output"] == "a"
