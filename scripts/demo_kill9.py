r"""LIVE DEMO: kill the writer without warning, then show instant recovery with nothing lost.

    python scripts\demo_kill9.py                 # default: 3 rounds, random kill moments
    python scripts\demo_kill9.py --rounds 5 --docs 4000

Each round: (1) a child process ingests papers into the index and writes down every paper whose commit() RETURNED
(an "acknowledged" paper = a promise not to lose it); (2) we kill it at a random moment with SIGKILL (Windows:
TerminateProcess, equally abrupt); (3) we reopen the index, time the recovery, check that every acknowledged paper
is present exactly once, run fsck, and answer a query IMMEDIATELY. A child that is restarted resends only what was
not acknowledged, so after the last round the index holds exactly the ground truth.

NOTE: this shows process death (what a crash, kill -9 or OOM kill looks like). It is NOT a power-loss test: pulling
the plug needs real hardware, and we do not claim it.
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from paperlens.store.fsck import fsck            # noqa: E402
from paperlens.store.index import IndexStore     # noqa: E402
from paperlens.synth import generate_papers      # noqa: E402


def acked(path):
    if not os.path.exists(path):
        return []
    lines = open(path, "rb").read().decode("utf-8").split("\n")[:-1]
    return [x for x in lines if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--docs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--index")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = a.index or tempfile.mkdtemp(prefix="paperlens-demo-")
    ack = os.path.join(base, "acked.txt")
    ix = os.path.join(base, "ix")
    os.makedirs(base, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    truth = {p.key for p in generate_papers(a.docs, a.seed)}
    print(f"index folder: {ix}\nground truth: {len(truth)} papers; the writer will be killed {a.rounds} time(s).\n")
    for rnd in range(1, a.rounds + 1):
        cmd = [sys.executable, "-m", "paperlens.testing.crash_child", ix, ack, str(a.docs), str(a.seed), "plain",
               "20", "100", "0.02", "skip"]
        child = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        before = len(acked(ack))
        time.sleep(rng.uniform(0.6, 1.8))
        if child.poll() is not None:
            print(f"round {rnd}: the child had already finished; nothing to kill")
            break
        child.kill()                                           # SIGKILL on POSIX, TerminateProcess on Windows
        child.wait()
        promised = set(acked(ack))
        print(f"round {rnd}: killed the writer without warning. It had promised {len(promised)} papers "
              f"({len(promised) - before} new this round).")
        t0 = time.perf_counter()
        st = IndexStore(ix)                                    # recovery happens right here
        took = (time.perf_counter() - t0) * 1000
        rec = st.recovery
        present = st.live_keys()
        missing = promised - present
        print(f"         reopened in {took:.0f} ms: replayed {rec.get('replayed', '?')} diary entries, cut {rec.get('torn_bytes_cut', 0)} torn bytes, "
              f"removed {rec.get('removed_tmp', 0)} half-written folder(s).")
        print(f"         papers present: {len(present)}   promised but missing: {len(missing)}   "
              f"(must be 0)   duplicates: 0 by construction (one key = one paper)")
        hits = st.search("inverted postings segment", k=3)
        print(f"         first query, right after recovery: {[h.arxiv_id for h in hits]}")
        st.close()
        r = fsck(ix)
        print(f"         fsck: {'OK' if r.ok else 'PROBLEMS ' + str(r.errors[:3])}\n")
        if missing or not r.ok:
            print("DEMO FAILED: acknowledged data lost or index damaged")
            sys.exit(1)
    final = subprocess.run([sys.executable, "-m", "paperlens.testing.crash_child", ix, ack, str(a.docs), str(a.seed), "plain",
                            "20", "100", "0", "skip"], env=env, capture_output=True, text=True)
    st = IndexStore(ix)
    ok = st.live_keys() == truth
    print(f"final run to completion (resending only unacknowledged papers): {len(st.live_keys())} papers, "
          f"equals ground truth: {ok}")
    st.close()
    print("DEMO PASSED" if ok else "DEMO FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
