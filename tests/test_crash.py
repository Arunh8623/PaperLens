import os
import random
import signal
import subprocess
import time

import pytest

from paperlens.store.crashpoints import CRASH_EXIT_CODE, KNOWN_POINTS
from paperlens.store.fsck import fsck
from paperlens.store.index import IndexStore
from paperlens.synth import generate_papers
from tests import crashlib as cl


def test_known_points_are_the_six_from_the_brief():
    assert KNOWN_POINTS == ("after_wal_write", "after_wal_fsync", "flush_after_tmp_written",
                            "flush_after_segment_rename", "flush_after_manifest",
                            "flush_after_wal_delete")


# crash point -> which occurrence to die on (so there is real history before the crash)
CASES = [
    ("after_wal_write", 23), ("after_wal_write", 1),
    ("after_wal_fsync", 5), ("after_wal_fsync", 1),
    ("flush_after_tmp_written", 1), ("flush_after_tmp_written", 3),
    ("flush_after_segment_rename", 1), ("flush_after_segment_rename", 2),
    ("flush_after_manifest", 1), ("flush_after_manifest", 3),
    ("flush_after_wal_delete", 1), ("flush_after_wal_delete", 2),
]


@pytest.mark.parametrize("point,after", CASES)
@pytest.mark.parametrize("kind", ["plain", "messy"])
def test_crash_at_every_write_step_loses_and_duplicates_nothing(tmp_path, point, after, kind):
    ix, ack = tmp_path / "ix", tmp_path / "ack.txt"
    n, seed = 150, 3
    r = cl.run_child(ix, ack, n, seed, kind, crash_at=point, crash_after=after)
    # If the point were not wired into the code the child would exit 0: that must fail the test.
    assert r.returncode == CRASH_EXIT_CODE, f"{point} never fired (exit {r.returncode}): {r.stderr[-500:]}"
    first_write = point.startswith("after_wal") and after == 1     # dies before anything can be acked
    if kind == "plain" and not first_write:
        assert cl.read_acked(ack), "nothing acknowledged before the crash: weak test"
    cl.check_after_crash(ix, ack, kind)
    cl.finish_and_verify(ix, ack, n, seed, kind)


def test_crash_during_flush_leaves_expected_debris(tmp_path):
    """After a crash at each flush step, the on-disk state is what the design says."""
    ix = tmp_path / "ix"
    cl.run_child(ix, tmp_path / "a1", 100, 1, "plain", crash_at="flush_after_tmp_written")
    assert any(n.startswith(".tmp-seg-") for n in os.listdir(ix / "segments"))
    rep = fsck(ix)
    assert rep.ok and any("leftover" in w for w in rep.warnings)

    ix2 = tmp_path / "ix2"
    cl.run_child(ix2, tmp_path / "a2", 100, 1, "plain", crash_at="flush_after_segment_rename")
    assert "seg-000001" in os.listdir(ix2 / "segments")
    assert not (ix2 / "MANIFEST").exists()                  # segment exists but is NOT published
    with IndexStore(ix2) as st:                              # recovery removes the orphan, replays diary
        assert st.stats()["recovery"]["removed_unlisted"] == 1 and st.stats()["segments"] == 0
        assert st.stats()["recovery"]["replayed"] > 0

    ix3 = tmp_path / "ix3"
    cl.run_child(ix3, tmp_path / "a3", 100, 1, "plain", crash_at="flush_after_manifest")
    assert len(os.listdir(ix3 / "wal")) == 1                 # old diary not deleted yet
    with IndexStore(ix3) as st:                              # entries <= last_flushed_seq are skipped
        s = st.stats()
        assert s["segments"] == 1 and s["recovery"]["replayed"] == 0
        assert s["recovery"]["old_wals_removed"] == 1


def test_torn_diary_tail_is_cut_and_acknowledged_data_survives(tmp_path):
    ix, ack = tmp_path / "ix", tmp_path / "ack.txt"
    cl.run_child(ix, ack, 60, 2, "plain", flush_every=10**6)       # no flush: everything is in the diary
    wal = ix / "wal" / os.listdir(ix / "wal")[0]
    size = wal.stat().st_size
    junk = b"\x40\x00\x00\x00\x12\x34\x56\x78partial-record"      # a record cut off mid-write
    with open(wal, "ab") as f:
        f.write(junk)
    assert fsck(ix).warnings                                  # fsck sees it, does not change it
    assert wal.stat().st_size == size + len(junk)
    with IndexStore(ix) as st:
        assert st.stats()["recovery"]["torn_bytes_cut"] == len(junk)
        assert len(st.live_keys()) == 60
    assert wal.stat().st_size == size


def test_corrupt_acknowledged_diary_entry_in_the_middle_truncates_later_entries(tmp_path):
    """Documents the design's known limit: the diary is a log, so a bad record
    in the MIDDLE of the newest file ends the log there (everything after it is
    cut). Bit rot of acknowledged data is a media-failure problem, not a crash."""
    ix = tmp_path / "ix"
    with IndexStore(ix) as st:
        for p in generate_papers(20, 4):
            st.put(p)
    wal = ix / "wal" / os.listdir(ix / "wal")[0]
    data = bytearray(wal.read_bytes())
    data[len(data) // 2] ^= 0xFF
    wal.write_bytes(bytes(data))
    with IndexStore(ix) as st:
        assert 0 < len(st.live_keys()) < 20
        assert st.stats()["recovery"]["torn_bytes_cut"] > 0


def test_damaged_manifest_is_refused_not_guessed(tmp_path):
    ix = tmp_path / "ix"
    with IndexStore(ix) as st:
        st.put(next(iter(generate_papers(1, 1)))); st.flush()
    mp = ix / "MANIFEST"
    mp.write_bytes(mp.read_bytes()[:-5])
    from paperlens.store.index import IndexCorrupt
    with pytest.raises(IndexCorrupt):
        IndexStore(ix)
    assert not fsck(ix).ok


def test_missing_listed_segment_is_an_error(tmp_path):
    import shutil
    from paperlens.store.index import IndexCorrupt
    ix = tmp_path / "ix"
    with IndexStore(ix) as st:
        st.put(next(iter(generate_papers(1, 1)))); st.flush()
    shutil.rmtree(ix / "segments" / "seg-000001")
    with pytest.raises(IndexCorrupt, match="missing"):
        IndexStore(ix)
    assert any("missing" in e for e in fsck(ix).errors)


def test_fsck_catches_a_flipped_byte_in_every_segment_file(tmp_path):
    ix = tmp_path / "ix"
    with IndexStore(ix) as st:
        for p in generate_papers(25, 5):
            st.put(p)
        st.flush()
    assert fsck(ix).ok
    from paperlens.store.segment import DATA_FILES
    for fname in DATA_FILES:
        path = ix / "segments" / "seg-000001" / fname
        orig = path.read_bytes()
        if not orig:
            continue
        flipped = bytearray(orig); flipped[len(flipped) // 2] ^= 0x01
        path.write_bytes(bytes(flipped))
        assert not fsck(ix).ok, f"fsck missed corruption in {fname}"
        path.write_bytes(orig)
    assert fsck(ix).ok


def test_fsck_refuses_when_a_writer_is_running(tmp_path):
    with IndexStore(tmp_path / "ix"):
        assert "locked" in fsck(tmp_path / "ix").errors[0]


# ------------------------------------------------------------ random SIGKILL (POSIX only)
@pytest.mark.skipif(os.name == "nt", reason="SIGKILL is POSIX only; Windows uses crash points / Process.kill")
@pytest.mark.parametrize("kind,seed", [("plain", 11), ("messy", 12)])
def test_random_sigkills_during_ingest(tmp_path, kind, seed):
    """Start the ingest, kill -9 it at random moments, check the invariants every
    time, resume (re-sending everything), repeat; finish and compare to ground truth."""
    rng = random.Random(seed)
    ix, ack = tmp_path / "ix", tmp_path / "ack.txt"
    n = 1500
    kills = 0
    for _round in range(6):
        # skip_acked: resend only what was never acknowledged, so every round makes real progress
        p = subprocess.Popen(cl.child_cmd(ix, ack, n, seed, kind, 5, 30, pause=0.01, skip_acked=True),
                             env=cl.child_env(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(rng.uniform(0.3, 0.8))             # python start-up is ~0.1-0.2 s; then it ingests
        if p.poll() is None:
            p.send_signal(signal.SIGKILL)
            kills += 1
        p.wait(timeout=30)
        if p.returncode not in (0, -signal.SIGKILL):
            pytest.fail(f"child failed on its own: {p.stderr.read().decode()[-500:]}")
        if os.path.exists(ix / "MANIFEST") or os.path.exists(ix / "wal"):
            cl.check_after_crash(ix, ack, kind)
    assert kills >= 3, "the kills mostly missed (machine too fast/slow): test not meaningful"
    cl.finish_and_verify(ix, ack, n, seed, kind, batch=5, flush_every=30)
