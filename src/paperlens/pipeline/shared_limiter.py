"""Rate limiter and credit budget shared by ALL processes.

What it does and why (plain language)
-------------------------------------
arXiv's rule (1 request per 3 s) is for ALL our processes combined, and the
OpenAlex API has a daily credit budget. A per-process limiter cannot enforce
that. These two objects live in shared memory with one lock each, so every
process sees the same state.

SharedLimiter: same idea as harvest/limiter.py. Under the lock, reserve the
next free time slot; release the lock; THEN sleep until the slot. Sleeping
outside the lock means waiting processes do not hold each other up, and slots
can never overlap. Same API (wait / penalize), so OAIClient accepts either.

SharedBudget: an atomic counter. try_spend(n) succeeds only if n credits
remain, so the total spent can never exceed the limit, however many processes
race (NFR8: 0 calls over the budget).

The time source is time.monotonic(), a system-wide clock on Linux, macOS and
Windows, so timestamps from different processes are comparable.

Create in the PARENT and pass the object to children as a Process argument
(spawn pickles synchronisation primitives only while starting the child).
"""
from __future__ import annotations

import multiprocessing as mp
import time


class SharedLimiter:
    def __init__(self, min_interval: float, ctx=None, history_size: int = 4096,
                 clock=time.monotonic, sleep=time.sleep):
        ctx = ctx or mp.get_context("spawn")
        self.min_interval = float(min_interval)
        self._lock = ctx.Lock()
        self._next_free = ctx.Value("d", float("-inf"), lock=False)
        self._count = ctx.Value("q", 0, lock=False)
        self._history = ctx.Array("d", history_size, lock=False)     # every granted slot, for evidence
        self._size = history_size
        self._clock, self._sleep = clock, sleep

    def wait(self) -> float:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_free.value)
            self._next_free.value = slot + self.min_interval
            n = self._count.value
            if n < self._size:
                self._history[n] = slot
            self._count.value = n + 1
        delay = slot - now
        if delay > 0:
            self._sleep(delay)                                         # outside the lock, on purpose
        return max(delay, 0.0)

    def penalize(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self._next_free.value = max(self._next_free.value, self._clock() + seconds)

    def history(self) -> list:
        with self._lock:
            n = min(self._count.value, self._size)
            return sorted(self._history[:n])

    def min_gap_seen(self):
        h = self.history()
        return min(b - a for a, b in zip(h, h[1:])) if len(h) > 1 else None


class SharedBudget:
    def __init__(self, total: int, ctx=None):
        ctx = ctx or mp.get_context("spawn")
        if total < 0:
            raise ValueError("budget must be >= 0")
        self.total = int(total)
        self._lock = ctx.Lock()
        self._spent = ctx.Value("q", 0, lock=False)

    def try_spend(self, n: int = 1) -> bool:
        if n < 0:
            raise ValueError("cannot spend a negative amount")
        with self._lock:
            if self._spent.value + n > self.total:
                return False
            self._spent.value += n
            return True

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent.value

    @property
    def remaining(self) -> int:
        return self.total - self.spent
