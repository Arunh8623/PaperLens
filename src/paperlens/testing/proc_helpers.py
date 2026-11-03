"""Child-process helpers for tests (must live in the package: 'spawn' children import them)."""
import time


def hammer_limiter(limiter, n):
    """Wait on the shared limiter n times; return nothing (evidence is in limiter.history())."""
    for _ in range(n):
        limiter.wait()


def spend_budget(budget, tries, out_queue):
    won = 0
    for _ in range(tries):
        if budget.try_spend(1):
            won += 1
    out_queue.put(won)


def hold_slot_then_sleep(limiter, seconds):
    limiter.wait()
    time.sleep(seconds)
