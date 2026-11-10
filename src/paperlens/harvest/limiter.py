"""Request limiter: "at most one request per `min_interval` seconds".

What it does and why (plain language)
-------------------------------------
arXiv's Terms of Use allow one request every 3 seconds (verified 2026-10-31
on info.arxiv.org/help/api/tou.html; they apply to OAI-PMH too and to ALL
our machines combined). This class hands out time slots.

The trick (and the reason it is safe with many threads/processes later):
  1. take the lock
  2. reserve the NEXT free slot  (slot = max(now, next_free); next_free = slot + interval)
  3. release the lock
  4. THEN sleep until the slot
Sleeping OUTSIDE the lock means waiting callers do not block each other from
reserving; each one gets its own, later slot. Slots can never overlap, so the
spacing guarantee holds.

Phase 0: threads in ONE process (threading.Lock).
Phase 2 (E1): the same logic moves to a lock + shared memory so it covers ALL
processes. The reserve-then-sleep shape stays identical.

`penalize(seconds)` pushes the next free slot out (used for Retry-After and
error backoff), so ALL waiting and future callers respect it.

`history` records every granted slot (clock units) as evidence for NFR8.
"""
from __future__ import annotations

import threading
import time


class MinIntervalLimiter:
    def __init__(self, min_interval: float = 3.0, clock=time.monotonic, sleep=time.sleep):
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        self.min_interval = float(min_interval)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_free = float("-inf")
        self.history: list = []

    def wait(self) -> float:
        """Block until this caller's slot. Returns the seconds it waited."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_free)
            self._next_free = slot + self.min_interval
            self.history.append(slot)
        delay = slot - now
        if delay > 0:
            self._sleep(delay)          # outside the lock, on purpose
        return max(delay, 0.0)

    def penalize(self, seconds: float) -> None:
        """Do not allow any request for at least `seconds` from now."""
        if seconds <= 0:
            return
        with self._lock:
            self._next_free = max(self._next_free, self._clock() + seconds)

    def min_gap_seen(self) -> float | None:
        """Smallest spacing between consecutive granted slots (None if < 2)."""
        h = sorted(self.history)
        if len(h) < 2:
            return None
        return min(b - a for a, b in zip(h, h[1:]))
