"""Worker processes: readers (tokenize + embed) and the single writer (owns the index).

What they do and why (plain language)
-------------------------------------
READER: gets a PaperJob, does the CPU-heavy work (tokenize, count terms, make a
vector), writes the vector into its shared-memory slot and reports PaperReady.
Many readers run in parallel; that is where the speed-up comes from.

WRITER: the ONLY process that touches the index. It collects up to 64
PaperReady messages, copies each vector out of shared memory and checks its
checksum, puts the papers into the diary, does ONE commit (one fsync) for the
whole batch, and only then acknowledges them. One writer means no write races;
one commit per batch means the fsync cost is shared (group commit).

Both send a Heartbeat regularly from a helper thread. The heartbeat is
deliberately silenced if the main thread has been busy on ONE item longer than
`busy_timeout`, so a worker stuck inside one job stops looking healthy and the
supervisor replaces it. (A heartbeat that always beats would only prove the
process exists, not that it makes progress.)

Everything here is module-level so Windows/macOS 'spawn' can import it.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from ..contracts.paper import PaperRecord
import numpy as np

from ..store.index import IndexStore
from ..text.tokenizer import paper_text, term_freqs
from ..vectors.embedders import make_embedder
from . import shm
from .embed import vector_crc
from .messages import (Ack, AckBatch, Fatal, Heartbeat, JobFailed, PaperJob, PaperReady, Ready,
                       Stop, WriterDone)

POISON_EXIT_CODE = 66          # distinctive exit code for injected worker deaths


@dataclass(frozen=True)
class Faults:
    """Deliberate failures, keyed by arxiv_id, for tests and the chaos demo."""
    crash_on_key: frozenset = frozenset()          # reader dies on EVERY attempt (poison paper)
    crash_once_on_key: frozenset = frozenset()     # reader dies on the first attempt only
    raise_on_key: frozenset = frozenset()          # reader raises an exception (worker survives)
    hang_once_on_key: frozenset = frozenset()      # reader hangs on the first attempt
    corrupt_vector_once_on_key: frozenset = frozenset()  # slot content != checksum, first attempt
    die_at_start_readers: bool = False             # every reader dies before saying Ready
    slow_seconds: float = 0.0                      # sleep per job (gives chaos time to bite)
    burn_seconds: float = 0.0                      # CPU-spin per job: SIMULATED model inference (benchmark only)


class _Chan:
    """Thread-safe sender over a Connection (main thread and heartbeat thread share it)."""

    def __init__(self, conn):
        self.conn, self.lock = conn, threading.Lock()

    def send(self, obj) -> None:
        with self.lock:
            self.conn.send(obj)


class _Beater(threading.Thread):
    def __init__(self, chan, role, wid, inc, interval, busy_timeout):
        super().__init__(daemon=True)
        self.chan, self.msg = chan, Heartbeat(role, wid, inc)
        self.interval, self.busy_timeout = interval, busy_timeout
        self.busy_since = None
        self._halt = threading.Event()

    def run(self):
        while not self._halt.wait(self.interval):
            bs = self.busy_since
            if bs is not None and time.monotonic() - bs > self.busy_timeout:
                continue                                   # stuck on one item: stop looking healthy
            try:
                self.chan.send(self.msg)
            except Exception:                              # supervisor is gone: do not linger as an orphan
                os._exit(0)

    def stop(self):
        self._halt.set()


# ------------------------------------------------------------------------------ reader
def reader_main(wid, inc, arena_name, dim, job_conn, res_conn, faults, limiter,
                hb_interval, busy_timeout, embedder_spec="hash"):
    chan = _Chan(res_conn)
    if faults.die_at_start_readers:
        os._exit(POISON_EXIT_CODE)
    try:
        embedder = make_embedder(f"hash:{dim}" if embedder_spec == "hash" else embedder_spec)
        if embedder.dim != dim:
            raise ValueError(f"embedder {embedder.name} makes {embedder.dim}-number vectors but the "
                             f"pipeline is configured for dim={dim}")
    except Exception as e:                              # configuration problem: tell the supervisor, do not loop
        chan.send(Fatal(f"reader cannot start: {type(e).__name__}: {e}"))
        return
    arena = shm.attach_arena(arena_name)
    beater = _Beater(chan, "reader", wid, inc, hb_interval, busy_timeout)
    beater.start()
    chan.send(Ready("reader", wid, inc, os.getpid()))
    try:
        while True:
            try:
                msg = job_conn.recv()
            except (EOFError, OSError):
                break                                      # supervisor closed our channel: leave
            if isinstance(msg, Stop):
                break
            beater.busy_since = time.monotonic()
            try:
                chan.send(_process(msg, wid, arena, arena_name, dim, faults, limiter, embedder))
            except Exception as e:                         # the worker survives a bad paper
                permanent = isinstance(e, _NoText)
                chan.send(JobFailed(msg.job_id, f"{type(e).__name__}: {e}", permanent))
            finally:
                beater.busy_since = None
    finally:
        beater.stop()
        arena.close()


class _NoText(ValueError):
    pass


def _process(job: PaperJob, wid, arena, arena_name, dim, f: Faults, limiter, embedder) -> PaperReady:
    paper: PaperRecord = job.paper
    first = job.attempt == 0
    aid = paper.arxiv_id
    if aid in f.crash_on_key or (first and aid in f.crash_once_on_key):
        os._exit(POISON_EXIT_CODE)                         # a hard crash, like SIGKILL / segfault
    if first and aid in f.hang_once_on_key:
        time.sleep(10 ** 6)
    if aid in f.raise_on_key:
        raise RuntimeError("injected failure")
    if f.slow_seconds:
        time.sleep(f.slow_seconds)
    if f.burn_seconds:
        end = time.perf_counter() + f.burn_seconds
        while time.perf_counter() < end:
            pass
    if limiter is not None:
        limiter.wait()                                     # stand-in for "fetch from a rate-limited source"
    tf, dl = term_freqs(paper_text(paper.title, paper.abstract))
    if not tf:
        raise _NoText("paper has no indexable text")
    raw = np.ascontiguousarray(embedder.embed_paper(paper, tf), dtype=np.float32).tobytes()
    crc = vector_crc(raw)
    shm.write_slot(arena, job.slot, dim, raw)
    if first and aid in f.corrupt_vector_once_on_key:      # simulate a torn / stale slot
        shm.write_slot(arena, job.slot, dim, bytes([raw[0] ^ 0xFF]) + raw[1:])
    return PaperReady(job.job_id, wid, paper.key, paper, tf, dl, arena_name, job.slot,
                      dim, crc, job.attempt)


# ------------------------------------------------------------------------------ writer
def writer_main(inc, index_path, flush_docs, batch_max, job_conn, res_conn, crash_env,
                hb_interval, busy_timeout, dim=0, store_vectors=True, vec_bits=1, ivf_min_docs=4096,
                embedder_name="", auto_compact=False):
    for k, v in crash_env.items():                         # per-incarnation fault injection
        os.environ[k] = v
    chan = _Chan(res_conn)
    store = IndexStore(index_path, flush_docs=flush_docs, vec_bits=vec_bits,   # runs recovery; may raise IndexLocked
                       ivf_min_docs=ivf_min_docs, embedder_name=embedder_name)
    if store_vectors and store.vec_dim and store.vec_dim != dim:
        chan.send(Fatal(f"index {index_path} stores {store.vec_dim}-number vectors but this run makes "
                        f"{dim}-number vectors; use the same embedder or a new index folder"))
        store.close()
        return
    if auto_compact:
        store.start_background_compaction(interval=0.5)
    beater = _Beater(chan, "writer", 0, inc, hb_interval, busy_timeout)
    beater.start()
    chan.send(Ready("writer", 0, inc, os.getpid(), {"recovery": dict(store.recovery)}))
    arenas = {}
    stats = {"batches": 0, "messages": 0, "commits": 0}
    stopping = False
    try:
        while not stopping:
            try:
                first = job_conn.recv()
            except (EOFError, OSError):
                break
            beater.busy_since = time.monotonic()
            batch = [first]
            while len(batch) < batch_max and job_conn.poll(0):
                try:
                    batch.append(job_conn.recv())
                except (EOFError, OSError):
                    break
            acks = []
            for m in batch:
                if isinstance(m, Stop):
                    stopping = True
                elif isinstance(m, PaperReady):
                    acks.append(_store_one(store, m, arenas, store_vectors))
            if acks:
                store.commit()                             # ONE fsync for the whole batch
                stats["commits"] += 1
                stats["batches"] += 1
                stats["messages"] += len(acks)
                chan.send(AckBatch(tuple(acks)))           # acknowledge only AFTER the commit
            beater.busy_since = None
        if stopping:
            store.commit()
            store.flush()
            chan.send(WriterDone({**stats, **{k: v for k, v in store.stats().items() if k != "recovery"}}))
    finally:
        beater.stop()
        for a in arenas.values():
            a.close()
        store.close()


def _store_one(store: IndexStore, m: PaperReady, arenas: dict, store_vectors: bool = True) -> Ack:
    try:
        arena = arenas.get(m.shm_name)
        if arena is None:
            arena = arenas[m.shm_name] = shm.attach_arena(m.shm_name)
        raw = shm.read_slot(arena, m.shm_slot, m.vec_dim)
    except (FileNotFoundError, ValueError, OSError) as e:
        return Ack(m.job_id, m.key, -2, f"bad_vector: {type(e).__name__}")
    got = vector_crc(raw)
    if got != m.vec_crc:
        return Ack(m.job_id, m.key, -2, "bad_vector: crc mismatch", got)   # stale/torn slot: retry
    try:
        res = store.put(m.paper, precomputed=(m.term_freqs, m.doc_len),
                        vector=np.frombuffer(raw, dtype=np.float32) if store_vectors else None)
    except ValueError as e:
        return Ack(m.job_id, m.key, -2, f"permanent: {e}", got)
    if res.outcome in ("insert", "replace"):
        return Ack(m.job_id, m.key, res.seq, res.outcome, got)
    return Ack(m.job_id, m.key, -1, res.outcome, got)
