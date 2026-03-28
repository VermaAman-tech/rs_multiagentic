from framework.memory.episodic_memory import EpisodicMemory
from framework.memory.working_memory import WorkingMemory


def test_episodic_memory_versioning():
    em = EpisodicMemory()
    v1 = em.write("k", {"a": 1})
    v2 = em.write("k", {"a": 2})
    assert v1 == 1
    assert v2 == 2
    assert em.read("k")["a"] == 2


def test_working_memory_flush_on_limit():
    wm = WorkingMemory(token_limit=5)
    assert wm.append("abc") is False
    assert wm.append("def") is True
    assert wm.view() == ""
