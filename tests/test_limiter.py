import threading
import time

import pytest

from paperlens.harvest.limiter import MinIntervalLimiter
from tests.fakes import FakeTime


def test_sequential_waits_are_spaced_exactly():
    ft = FakeTime()
    lim = MinIntervalLimiter(3.0, ft.clock, ft.sleep)
    assert lim.wait() == 0                       # first call is free
    for _ in range(4):
        lim.wait()
    gaps = [b - a for a, b in zip(lim.history, lim.history[1:])]
    assert gaps == [3.0] * 4 and lim.min_gap_seen() == 3.0
    assert ft.sleeps == [3.0] * 4


def test_no_extra_delay_if_caller_was_slow():
    ft = FakeTime()
    lim = MinIntervalLimiter(3.0, ft.clock, ft.sleep)
    lim.wait()
    ft.now += 10                                 # caller did other work for 10 s
    assert lim.wait() == 0 and ft.sleeps == []


def test_penalize_delays_next_request():
    ft = FakeTime()
    lim = MinIntervalLimiter(3.0, ft.clock, ft.sleep)
    lim.wait()
    lim.penalize(20)
    lim.wait()
    assert lim.history[1] - lim.history[0] == 20
    lim.penalize(0)
    lim.penalize(-5)                             # ignored
    lim.wait()
    assert lim.history[2] - lim.history[1] == 3.0


def test_penalize_never_shortens():
    ft = FakeTime()
    lim = MinIntervalLimiter(10.0, ft.clock, ft.sleep)
    lim.wait()
    lim.penalize(1)                              # smaller than the interval
    lim.wait()
    assert lim.history[1] - lim.history[0] == 10.0


def test_bad_interval():
    with pytest.raises(ValueError):
        MinIntervalLimiter(-1)
    assert MinIntervalLimiter(3.0).min_gap_seen() is None


def test_threads_get_distinct_spaced_slots_and_do_not_block_each_other():
    """Reserve-then-sleep-outside-the-lock: 8 threads arriving together get
    slots 0, I, 2I ... 7I and all finish in about 7*I, not 8 sequential waits
    stacked on a held lock."""
    interval, n = 0.05, 8
    lim = MinIntervalLimiter(interval)
    wake = []
    start = time.monotonic()

    def work():
        lim.wait()
        wake.append(time.monotonic() - start)

    ts = [threading.Thread(target=work) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    slots = sorted(lim.history)
    assert all(b - a >= interval - 1e-9 for a, b in zip(slots, slots[1:]))
    assert len(set(slots)) == n
    assert max(wake) >= (n - 1) * interval - 0.03        # nobody woke before its slot (30 ms slack: Windows clock ticks ~15 ms)
    assert max(wake) < (n - 1) * interval + 1.0           # and nobody waited extra
