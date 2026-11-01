import pytest

from paperlens.store.fsutil import retry_fs


class Flaky:
    def __init__(self, fail_times, exc=PermissionError):
        self.calls, self.fail_times, self.exc = 0, fail_times, exc

    def __call__(self, *a):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc("locked by antivirus")
        return ("ok", a)


def test_retries_on_windows_until_it_works_with_backoff():
    f, waits = Flaky(3), []
    assert retry_fs(f, 1, 2, on_windows=True, sleep=waits.append, delay=0.05) == ("ok", (1, 2))
    assert f.calls == 4 and waits == [0.05, 0.1, 0.2]


def test_gives_up_after_attempts_and_raises_the_real_error():
    f, waits = Flaky(99), []
    with pytest.raises(PermissionError):
        retry_fs(f, on_windows=True, sleep=waits.append, attempts=4)
    assert f.calls == 4 and len(waits) == 3


def test_never_retries_off_windows():
    f, waits = Flaky(1), []
    with pytest.raises(PermissionError):
        retry_fs(f, on_windows=False, sleep=waits.append)
    assert f.calls == 1 and waits == []


def test_other_errors_are_not_retried():
    f = Flaky(5, exc=FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        retry_fs(f, on_windows=True, sleep=lambda s: None)
    assert f.calls == 1


def test_backoff_is_capped_at_one_second():
    f, waits = Flaky(10), []
    retry_fs(f, on_windows=True, sleep=waits.append, attempts=12, delay=0.4)
    assert max(waits) == 1.0


def test_success_first_time_does_not_sleep():
    waits = []
    assert retry_fs(lambda: 5, on_windows=True, sleep=waits.append) == 5 and waits == []
