import os
import signal
import subprocess
import sys
import time

import pytest

from paperlens.store.lock import DirLock, IndexLocked

HOLDER = r"""
import sys, time
from paperlens.store.lock import DirLock
l = DirLock(sys.argv[1]); l.acquire()
print("locked", flush=True)
time.sleep(60)
"""


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    return e


def test_second_lock_in_same_process_fails_then_succeeds_after_release(tmp_path):
    a, b = DirLock(tmp_path), DirLock(tmp_path)
    a.acquire()
    with pytest.raises(IndexLocked):
        b.acquire()
    a.release()
    b.acquire()
    b.release()
    b.release()                                       # releasing twice is harmless


def test_double_acquire_on_same_object_is_a_bug(tmp_path):
    a = DirLock(tmp_path)
    with a:
        with pytest.raises(RuntimeError):
            a.acquire()


@pytest.mark.skipif(os.name == "nt", reason="uses SIGKILL (POSIX only)")
def test_lock_held_by_other_process_and_freed_when_it_is_killed(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", HOLDER, str(tmp_path)], env=env(),
                         stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        with pytest.raises(IndexLocked):
            DirLock(tmp_path).acquire()
        p.send_signal(signal.SIGKILL)                  # no cleanup code runs
        p.wait(timeout=10)
        t0 = time.time()
        with DirLock(tmp_path):                        # the OS released it for us
            pass
        assert time.time() - t0 < 2
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
