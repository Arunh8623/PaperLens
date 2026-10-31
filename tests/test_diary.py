import os
import subprocess
import sys

import pytest

from paperlens.store import diary
from paperlens.store.crashpoints import CRASH_EXIT_CODE

PAYLOADS = [b"", b"a", b"hello\n\r\n\x00world", b"x" * 300, bytes(range(256))]


def image(payloads):
    return b"".join(diary.frame(p) for p in payloads)


def record_ends(payloads):
    ends, pos = [], 0
    for p in payloads:
        pos += diary.HEADER.size + len(p)
        ends.append(pos)
    return ends


def test_clean_roundtrip():
    r = diary.scan_bytes(image(PAYLOADS))
    assert r.payloads == PAYLOADS and not r.torn and r.reason == "clean"


def test_empty_file_is_clean():
    r = diary.scan_bytes(b"")
    assert r.payloads == [] and not r.torn


def test_truncate_at_every_byte_offset():
    """A crash can cut the file anywhere. Good prefix must be exactly the
    records that are fully present."""
    data = image(PAYLOADS)
    ends = record_ends(PAYLOADS)
    for t in range(len(data) + 1):
        r = diary.scan_bytes(data[:t])
        whole = sum(1 for e in ends if e <= t)
        assert r.payloads == PAYLOADS[:whole], f"cut at {t}"
        assert r.good_bytes == (ends[whole - 1] if whole else 0), f"cut at {t}"
        assert r.torn == (t not in [0] + ends), f"cut at {t}"


def test_flip_every_byte_stops_at_that_record():
    data = bytearray(image(PAYLOADS))
    ends = record_ends(PAYLOADS)
    for i in range(len(data)):
        bad = bytearray(data)
        bad[i] ^= 0xFF
        r = diary.scan_bytes(bytes(bad))
        idx = next(k for k, e in enumerate(ends) if i < e)  # record holding byte i
        assert r.payloads == PAYLOADS[:idx], f"flip at {i}"
        assert r.torn


def test_absurd_length_is_rejected():
    import struct
    junk = struct.pack("<II", 0xFFFFFFFF, 0) + b"zz"
    r = diary.scan_bytes(diary.frame(b"ok") + junk)
    assert r.payloads == [b"ok"] and r.reason == "bad_length"


def test_recover_tail_truncates_and_appends_cleanly(tmp_path):
    p = tmp_path / "wal-00000000000000000001.log"
    data = image(PAYLOADS)
    p.write_bytes(data + diary.frame(b"half")[:-2])  # torn last record
    res = diary.recover_tail(p)
    assert res.torn and res.payloads == PAYLOADS
    assert p.stat().st_size == len(data)
    with diary.DiaryWriter(p) as w:
        w.put(b"new")
        w.commit()
    assert diary.scan_file(p).payloads == PAYLOADS + [b"new"]


def test_newlines_survive_byte_for_byte(tmp_path):
    """Guards the Windows 0x0A -> 0x0D0A bug (needs O_BINARY).
    On Linux this always passes; the real check is running it on Windows."""
    p = tmp_path / "d.log"
    payload = b"\n" * 10 + b"\r\n" + b"\n"
    with diary.DiaryWriter(p) as w:
        w.put(payload)
        w.commit()
    assert p.read_bytes() == diary.frame(payload)


def test_wal_names_sort_by_seq(tmp_path):
    for s in (10, 2, 300):
        (tmp_path / diary.wal_name(s)).write_bytes(b"")
    (tmp_path / "other.txt").write_bytes(b"")
    assert [s for s, _ in diary.list_wals(tmp_path)] == [2, 10, 300]
    assert diary.parse_wal_name("wal-xyz.log") is None


# ---- real process death at named crash points -----------------------------
CHILD = r"""
import sys
from paperlens.store.diary import DiaryWriter
w = DiaryWriter(sys.argv[1])
for i in range(5):
    w.put(b"rec%d" % i)
    if i == 1:
        w.commit()
w.commit()
"""


def run_child(path, crash_at, after=1):
    env = dict(os.environ, PAPERLENS_CRASH_AT=crash_at, PAPERLENS_CRASH_AFTER=str(after))
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    env["PYTHONPATH"] = os.path.abspath(src) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", CHILD, str(path)], env=env,
                          capture_output=True, timeout=30)


def test_crash_after_wal_write_keeps_written_records(tmp_path):
    p = tmp_path / "d.log"
    r = run_child(p, "after_wal_write", after=3)
    assert r.returncode == CRASH_EXIT_CODE
    # Process death (not power loss): the OS still has the 3 written records.
    assert diary.scan_file(p).payloads == [b"rec0", b"rec1", b"rec2"]


def test_crash_after_fsync_has_committed_records(tmp_path):
    p = tmp_path / "d.log"
    r = run_child(p, "after_wal_fsync", after=1)  # first commit is after rec1
    assert r.returncode == CRASH_EXIT_CODE
    assert diary.scan_file(p).payloads[:2] == [b"rec0", b"rec1"]


def test_no_crash_when_env_unset(tmp_path):
    p = tmp_path / "d.log"
    env = dict(os.environ)
    env.pop("PAPERLENS_CRASH_AT", None)
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    r = subprocess.run([sys.executable, "-c", CHILD, str(p)], env=env, capture_output=True, timeout=30)
    assert r.returncode == 0
    assert len(diary.scan_file(p).payloads) == 5
