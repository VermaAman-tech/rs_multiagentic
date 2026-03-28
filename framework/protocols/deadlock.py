from __future__ import annotations

from collections import defaultdict


class DeadlockDetector:
    def __init__(self) -> None:
        self.wait_for: dict[str, set[str]] = defaultdict(set)

    def add_wait(self, waiter: str, waiting_on: str) -> None:
        self.wait_for[waiter].add(waiting_on)

    def remove_wait(self, waiter: str, waiting_on: str) -> None:
        self.wait_for[waiter].discard(waiting_on)

    def detect_cycle(self) -> list[str] | None:
        seen: set[str] = set()
        stack: set[str] = set()
        parent: dict[str, str] = {}

        def dfs(node: str) -> list[str] | None:
            seen.add(node)
            stack.add(node)
            for nxt in self.wait_for.get(node, set()):
                if nxt not in seen:
                    parent[nxt] = node
                    cyc = dfs(nxt)
                    if cyc:
                        return cyc
                elif nxt in stack:
                    cyc = [nxt, node]
                    cur = node
                    while cur in parent and parent[cur] != nxt:
                        cur = parent[cur]
                        cyc.append(cur)
                    cyc.append(nxt)
                    return list(reversed(cyc))
            stack.remove(node)
            return None

        for n in list(self.wait_for.keys()):
            if n not in seen:
                cyc = dfs(n)
                if cyc:
                    return cyc
        return None


def break_cycle(detector: DeadlockDetector) -> dict[str, str] | None:
    cyc = detector.detect_cycle()
    if not cyc:
        return None
    if len(cyc) >= 2:
        detector.remove_wait(cyc[0], cyc[1])
    return {"broken_from": cyc[0], "broken_to": cyc[1], "cycle": "->".join(cyc)}
