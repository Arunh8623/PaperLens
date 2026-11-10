import multiprocessing as mp

import pytest

from paperlens.pipeline.shared_limiter import SharedBudget, SharedLimiter
from paperlens.testing.proc_helpers import hammer_limiter, spend_budget

CTX = mp.get_context("spawn")


def test_limiter_single_process_spacing():
    import time
    lim = SharedLimiter(0.05)
    t0 = time.monotonic()
    for _ in range(5):
        lim.wait()
    assert time.monotonic() - t0 >= 4 * 0.05 - 0.01              # really slept, not just booked slots
    assert lim.min_gap_seen() >= 0.05 - 1e-9
    assert len(lim.history()) == 5


def test_limiter_penalize_pushes_next_slot():
    lim = SharedLimiter(0.01)
    lim.wait()
    lim.penalize(0.2)
    lim.wait()
    h = lim.history()
    assert h[1] - h[0] >= 0.19


def test_limiter_across_processes_never_closer_than_interval():
    import time
    lim = SharedLimiter(0.05, ctx=CTX)
    procs = [CTX.Process(target=hammer_limiter, args=(lim, 8)) for _ in range(4)]
    t0 = time.monotonic()
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
        assert p.exitcode == 0
    assert time.monotonic() - t0 >= 31 * 0.05 - 0.01             # 32 calls cannot finish faster than this
    assert len(lim.history()) == 32
    assert lim.min_gap_seen() >= 0.05 - 1e-6                     # 4 processes, still one global rhythm


def test_limiter_slot_survives_a_killed_process_conservatively():
    lim = SharedLimiter(0.05, ctx=CTX)
    p = CTX.Process(target=hammer_limiter, args=(lim, 50))
    p.start()
    import time
    time.sleep(0.5)
    p.kill()
    p.join(10)
    lim.wait()                                                    # a fresh process-independent caller still works
    assert lim.min_gap_seen() >= 0.05 - 1e-6


def test_budget_is_exact_under_racing_processes():
    budget = SharedBudget(100, ctx=CTX)
    q = CTX.Queue()
    procs = [CTX.Process(target=spend_budget, args=(budget, 60, q)) for _ in range(4)]
    for p in procs:
        p.start()
    won = sum(q.get(timeout=30) for _ in procs)
    for p in procs:
        p.join(30)
    assert won == 100 and budget.spent == 100 and budget.remaining == 0
    assert budget.try_spend(1) is False


def test_budget_edge_cases():
    b = SharedBudget(3)
    assert b.try_spend(3) and not b.try_spend(1) and b.remaining == 0
    assert SharedBudget(0).try_spend(0) is True
    with pytest.raises(ValueError):
        b.try_spend(-1)
    with pytest.raises(ValueError):
        SharedBudget(-1)
