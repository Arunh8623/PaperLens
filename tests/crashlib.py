"""Shared helpers for tests that crash a real child process."""
import os
import subprocess
import sys

from paperlens.store.crashpoints import CRASH_EXIT_CODE
from paperlens.store.fsck import fsck
from paperlens.store.index import IndexStore
from paperlens.synth import expected_final
from paperlens.testing.crash_child import make_stream

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


def child_env(**extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PAPERLENS_CRASH_AT", None)
    env.pop("PAPERLENS_CRASH_AFTER", None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def child_cmd(index, ackfile, n, seed, kind, batch, flush_every, pause=0, skip_acked=False):
    cmd = [sys.executable, "-m", "paperlens.testing.crash_child", str(index), str(ackfile),
           str(n), str(seed), kind, str(batch), str(flush_every), str(pause)]
    return cmd + ["skip"] if skip_acked else cmd


def run_child(index, ackfile, n=200, seed=1, kind="plain", batch=7, flush_every=40,
              crash_at=None, crash_after=1, timeout=120):
    extra = {}
    if crash_at:
        extra = {"PAPERLENS_CRASH_AT": crash_at, "PAPERLENS_CRASH_AFTER": crash_after}
    return subprocess.run(child_cmd(index, ackfile, n, seed, kind, batch, flush_every),
                          env=child_env(**extra), capture_output=True, text=True, timeout=timeout)


def read_acked(ackfile):
    """Keys whose line was completely written (a half-written last line is ignored)."""
    if not os.path.exists(ackfile):
        return []
    data = open(ackfile, "rb").read().decode("utf-8")
    lines = data.split("\n")
    return [ln for ln in lines[:-1] if ln]        # last element has no newline -> incomplete


def check_after_crash(index, ackfile, kind="plain"):
    """The durability invariants. Returns the live key set."""
    rep = fsck(index)                              # BEFORE recovery: only warnings allowed
    assert rep.ok, f"fsck errors after crash: {rep.errors}"
    acked = read_acked(ackfile)
    with IndexStore(index) as st:
        live = st.live_keys()
        assert fsck_clean_view(st)
        arxiv_ids = [k.rsplit("@", 1)[0] for k in live]
        assert len(arxiv_ids) == len(set(arxiv_ids)), "two live versions of one paper"
        if kind == "plain":
            missing = [k for k in acked if k not in live]
            assert not missing, f"LOST acknowledged papers: {missing[:5]} (+{len(missing)})"
            assert st.stats()["live_docs"] == len(live)
    rep2 = fsck(index)                             # AFTER recovery: fully clean
    assert rep2.ok and not rep2.warnings, (rep2.errors, rep2.warnings)
    return live


def fsck_clean_view(st):
    return st.stats()["live_docs"] == len(st.live_keys())


def finish_and_verify(index, ackfile, n, seed, kind, batch=7, flush_every=40):
    """Re-send EVERYTHING to the recovered index (what the supervisor would do);
    the end state must equal the ground truth exactly once each."""
    r = run_child(index, ackfile, n, seed, kind, batch, flush_every)
    assert r.returncode == 0, r.stderr[-800:]
    truth = {p.key for p in expected_final(make_stream(n, seed, kind)).values()}
    with IndexStore(index) as st:
        assert st.live_keys() == truth, "final state differs from ground truth"
        assert st.stats()["live_docs"] == len(truth)
    rep = fsck(index)
    assert rep.ok and not rep.warnings, (rep.errors, rep.warnings)
