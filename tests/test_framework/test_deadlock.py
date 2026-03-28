from framework.protocols.deadlock import DeadlockDetector, break_cycle


def test_deadlock_detect_and_break():
    d = DeadlockDetector()
    d.add_wait("vra", "ga")
    d.add_wait("ga", "vra")
    assert d.detect_cycle() is not None
    assert break_cycle(d) is not None
