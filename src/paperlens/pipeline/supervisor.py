"""Supervisor: starts the workers, routes every message, keeps the books.

What it does and why (plain language)
-------------------------------------
The supervisor is the manager of a small factory.
  * It hands each paper (a "job") to the least-busy reader, never more than
    `max_inflight_per_reader` at once (backpressure).
  * It forwards each finished PaperReady to the single writer.
  * It marks a job DONE only when the writer acknowledges it as durable.
  * If a worker dies or hangs it starts a replacement and re-does the lost work.

Delivery is "at least once" (a job may be sent more than once after a crash) and
the writer is idempotent (same arxiv_id@datestamp twice is ignored), so storage
is exactly-once.

Job life cycle:   pending -> in_reader -> in_writer -> done
                     ^____________|___________|
                  (reader died / failed / writer rejected: back to pending,
                   or to dead-letter after max_attempts)

Blame rule: when a reader dies, its jobs go back to pending, but only the job
it was working on (the oldest unfinished one) gets an attempt counted against
it. A paper that keeps killing workers is parked as a dead letter; everything
else still finishes.

Each worker incarnation gets its OWN two pipes (jobs in, results out). Nothing
is shared between workers, so killing one cannot corrupt another's channel.
A sender thread per worker means the supervisor never blocks on a full pipe
(Windows pipe buffers are small; a blocked supervisor plus a blocked worker
would deadlock).
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import random
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from multiprocessing import connection

from . import shm
from .messages import (AckBatch, Fatal, Heartbeat, JobFailed, PaperJob, PaperReady, Ready, Stop,
                       WriterDone)
from .workers import Faults, reader_main, writer_main


class PipelineError(Exception):
    pass


@dataclass
class PipelineConfig:
    index_path: str
    n_readers: int = 2
    dim: int = 64
    embedder: str = "hash"             # "hash" = hash:<dim>; or hash:N / st:<model> (see vectors/embedders.py)
    store_vectors: bool = True         # persist the verified vectors in the index (Phase 3)
    vec_bits: int = 1                  # quantization bits for new segments
    ivf_min_docs: int = 4096
    max_inflight_per_reader: int = 8
    max_outstanding: int = 128
    writer_batch: int = 64
    flush_docs: int = 1000
    auto_compact: bool = False         # writer merges segments in a background thread (FR4)
    max_attempts: int = 3
    heartbeat_interval: float = 0.25
    job_timeout: float = 10.0          # reader stops heartbeating if ONE job takes longer than this
    writer_job_timeout: float = 60.0
    hang_timeout: float = 5.0          # supervisor kills a worker silent for this long
    startup_timeout: float = 60.0      # worker not Ready after this long (spawn is slow on Windows)
    max_respawns: int = 200
    startup_fail_limit: int = 5        # consecutive deaths before Ready -> give up with a clear error
    faults: Faults = field(default_factory=Faults)
    reader_limiter: object = None      # optional SharedLimiter each reader waits on per job
    chaos_reader_interval: float = 0.0  # >0: kill a random ready reader this often (seconds)
    chaos_writer_kills: int = 0
    chaos_writer_interval: float = 1.0
    chaos_seed: int = 0
    writer_crash_at: str = ""          # crash point injected into the FIRST writer incarnation
    writer_crash_after: int = 1
    timeout: float = 0.0               # >0: overall safety limit in seconds
    log: object = None


@dataclass
class PipelineReport:
    jobs: int = 0
    outcomes: dict = field(default_factory=dict)       # insert / replace / duplicate / stale
    dead_letters: list = field(default_factory=list)   # (key, reason, attempts)
    reader_respawns: int = 0
    writer_respawns: int = 0
    chaos_reader_kills: int = 0
    chaos_writer_kills: int = 0
    hang_kills: int = 0
    rejected_retries: int = 0
    replace_seconds_max: float = 0.0
    replace_seconds: list = field(default_factory=list)
    elapsed: float = 0.0
    writer_stats: dict = field(default_factory=dict)
    limiter_min_gap: object = None


_CLOSE = object()


class _Job:
    __slots__ = ("id", "paper", "attempt", "state", "reader", "inc", "slot", "ready")

    def __init__(self, jid, paper):
        self.id, self.paper, self.attempt = jid, paper, 0
        self.state, self.reader, self.inc, self.slot, self.ready = "pending", None, None, None, None


class _Worker:
    def __init__(self, role, wid, inc, proc, send_conn, recv_conn, died_at_prev):
        self.role, self.wid, self.inc, self.proc = role, wid, inc, proc
        self.send_conn, self.recv_conn = send_conn, recv_conn
        self.outbox = queue.Queue()
        self.ready = False
        self.alive = True
        self.killed_by_us = False
        self.kill_time = None
        self.inflight = deque()                    # reader: job ids in dispatch order
        self.started = self.last_seen = time.monotonic()
        self.replaces_died_at = died_at_prev
        self.sender = threading.Thread(target=self._pump, daemon=True)
        self.sender.start()

    def _pump(self):
        while True:
            item = self.outbox.get()
            if item is _CLOSE:
                return
            try:
                self.send_conn.send(item)
            except (OSError, ValueError, EOFError):
                return                              # worker is gone; the supervisor notices separately

    def send(self, msg):
        self.outbox.put(msg)

    def close(self):
        self.outbox.put(_CLOSE)
        for c in (self.send_conn, self.recv_conn):
            try:
                c.close()
            except Exception:
                pass


class Supervisor:
    def __init__(self, papers, cfg: PipelineConfig):
        self.cfg = cfg
        self.ctx = mp.get_context("spawn")                      # same behaviour on every OS
        self.src = iter(papers)
        self.src_done = False
        self.jobs: dict = {}
        self.next_job = 0
        self.pending = deque()
        self.finished = 0
        self.report = PipelineReport()
        self.outcomes = Counter()
        self.readers: dict = {}
        self.writer = None
        self.arenas: dict = {}
        self.free_slots: dict = {}
        self.next_inc: dict = {}
        self.startup_fails: Counter = Counter()
        self.respawns = 0
        self.stopping = False
        self.writer_done = False
        self.rng = random.Random(cfg.chaos_seed)
        self.last_died_at: dict = {}
        self.log = cfg.log or (lambda *_: None)

    # ------------------------------------------------------------------ process management
    def _spawn(self, role, wid):
        cfg = self.cfg
        key = (role, wid)
        inc = self.next_inc.get(key, 0)
        self.next_inc[key] = inc + 1
        job_r, job_w = self.ctx.Pipe(duplex=False)
        res_r, res_w = self.ctx.Pipe(duplex=False)
        if role == "reader":
            args = (wid, inc, self.arenas[wid].name, cfg.dim, job_r, res_w, cfg.faults,
                    cfg.reader_limiter, cfg.heartbeat_interval, cfg.job_timeout, cfg.embedder)
            proc = self.ctx.Process(target=reader_main, args=args, daemon=True,
                                    name=f"paperlens-reader-{wid}.{inc}")
        else:
            crash_env = {}
            if inc == 0 and cfg.writer_crash_at:
                crash_env = {"PAPERLENS_CRASH_AT": cfg.writer_crash_at,
                             "PAPERLENS_CRASH_AFTER": str(cfg.writer_crash_after)}
            args = (inc, str(cfg.index_path), cfg.flush_docs, cfg.writer_batch, job_r, res_w,
                    crash_env, cfg.heartbeat_interval, cfg.writer_job_timeout, cfg.dim,
                    cfg.store_vectors, cfg.vec_bits, cfg.ivf_min_docs,
                    f"hash:{cfg.dim}" if cfg.embedder == "hash" else cfg.embedder,
                    cfg.auto_compact)
            proc = self.ctx.Process(target=writer_main, args=args, daemon=True,
                                    name=f"paperlens-writer.{inc}")
        proc.start()
        job_r.close()                                         # the child owns these ends now; closing ours
        res_w.close()                                         # lets EOF reach whoever is left on each pipe
        w = _Worker(role, wid, inc, proc, job_w, res_r, self.last_died_at.pop(key, None))
        if role == "reader":
            self.readers[wid] = w
        else:
            self.writer = w
            for jid in sorted(j.id for j in self.jobs.values() if j.state == "in_writer"):
                w.send(self.jobs[jid].ready)                  # re-send everything not yet acknowledged
            if self.stopping:
                w.send(Stop())
        return w

    def _kill(self, w, reason):
        if not w.alive or w.killed_by_us:
            return
        w.killed_by_us = True
        w.kill_time = time.monotonic()
        self.log(f"killing {w.role} {w.wid}.{w.inc}: {reason}")
        try:
            w.proc.kill()
        except Exception:
            pass

    def _on_death(self, w):
        if not w.alive:
            return
        # Read whatever the dead worker managed to send before dying (results are still valid).
        try:
            while w.recv_conn.poll(0):
                self._handle(w, w.recv_conn.recv())
        except (EOFError, OSError):
            pass
        w.alive = False
        try:
            w.proc.join(timeout=2)
        except Exception:
            pass
        code = w.proc.exitcode
        now = time.monotonic()
        died_at = w.kill_time or now
        self.last_died_at[(w.role, w.wid)] = died_at
        w.close()
        self.log(f"{w.role} {w.wid}.{w.inc} died (exit code {code})")
        if not w.ready and not w.killed_by_us:
            self.startup_fails[w.role] += 1
            if self.startup_fails[w.role] >= self.cfg.startup_fail_limit:
                raise PipelineError(
                    f"{w.role} workers keep dying before they are ready (last exit code {code}); "
                    "this is a start-up problem, not a bad paper. Run the worker import by hand "
                    "to see the error.")
        if w.role == "reader":
            self._requeue_reader_jobs(w)
        self.respawns += 1
        if self.respawns > self.cfg.max_respawns:
            raise PipelineError(f"respawn limit ({self.cfg.max_respawns}) exceeded")
        if w.role == "reader":
            self.report.reader_respawns += 1
            self._spawn("reader", w.wid)
        else:
            self.report.writer_respawns += 1
            self._spawn("writer", 0)

    def _requeue_reader_jobs(self, w):
        victims = [self.jobs[j] for j in w.inflight
                   if self.jobs[j].state == "in_reader" and self.jobs[j].reader == w.wid
                   and self.jobs[j].inc == w.inc]
        w.inflight.clear()
        for n, job in enumerate(victims):
            self._release_slot(job)
            if n == 0:                                         # only the job at the head is blamed
                job.attempt += 1
                if job.attempt >= self.cfg.max_attempts:
                    self._dead_letter(job, f"worker died {job.attempt} times on it")
                    continue
        # put survivors back at the FRONT, in original order
        for job in reversed([j for j in victims if j.state != "dead"]):
            job.state = "pending"
            self.pending.appendleft(job.id)

    # ------------------------------------------------------------------ bookkeeping helpers
    def _release_slot(self, job):
        if job.slot is not None and job.reader is not None:
            self.free_slots[job.reader].append(job.slot)
        job.slot = job.reader = job.inc = None

    def _dead_letter(self, job, reason):
        self._release_slot(job)
        job.state = "dead"
        job.ready = None
        self.finished += 1
        self.report.dead_letters.append((job.paper.key, reason, job.attempt))
        self.log(f"dead letter {job.paper.key}: {reason}")

    def _done(self, job, outcome):
        self._release_slot(job)
        job.state = "done"
        job.ready = None
        job.paper = job.paper                                  # key stays available for the report
        self.finished += 1
        self.outcomes[outcome] += 1

    def _retry(self, job, reason, permanent=False):
        self._release_slot(job)
        job.ready = None
        job.attempt += 1
        if permanent or job.attempt >= self.cfg.max_attempts:
            self._dead_letter(job, reason)
        else:
            job.state = "pending"
            self.pending.appendleft(job.id)

    # ------------------------------------------------------------------ message handling
    def _handle(self, w, msg):
        w.last_seen = time.monotonic()
        if isinstance(msg, Heartbeat):
            return
        if isinstance(msg, Fatal):
            raise PipelineError(msg.message)
        if isinstance(msg, Ready):
            w.ready = True
            self.startup_fails[w.role] = 0
            if w.replaces_died_at is not None:
                dt = time.monotonic() - w.replaces_died_at
                self.report.replace_seconds.append(dt)
                self.report.replace_seconds_max = max(self.report.replace_seconds_max, dt)
            return
        if w.role == "reader":
            job = self.jobs.get(getattr(msg, "job_id", -1))
            if job is None or job.state != "in_reader" or job.reader != w.wid or job.inc != w.inc:
                return                                         # stale message from a replaced incarnation
            if w.inflight and job.id in w.inflight:
                w.inflight.remove(job.id)
            if isinstance(msg, PaperReady):
                job.state = "in_writer"
                job.ready = msg
                if self.writer is not None and self.writer.alive:
                    self.writer.send(msg)
            elif isinstance(msg, JobFailed):
                self._retry(job, msg.error, msg.permanent)
        else:
            if isinstance(msg, AckBatch):
                for a in msg.acks:
                    job = self.jobs.get(a.job_id)
                    if job is None or job.state != "in_writer":
                        continue                               # duplicate ack after a resend
                    if a.seq >= 1 or a.seq == -1:
                        self._done(job, a.outcome)
                    else:
                        self.report.rejected_retries += 1
                        self._retry(job, a.outcome, permanent=a.outcome.startswith("permanent"))
            elif isinstance(msg, WriterDone):
                self.writer_done = True
                self.report.writer_stats = msg.stats

    # ------------------------------------------------------------------ scheduling
    def _outstanding(self):
        return self.next_job - self.finished

    def _refill(self):
        cap = self.cfg.n_readers * self.cfg.max_inflight_per_reader
        while (not self.src_done and len(self.pending) < cap
               and self._outstanding() < self.cfg.max_outstanding):
            try:
                paper = next(self.src)
            except StopIteration:
                self.src_done = True
                break
            job = _Job(self.next_job, paper)
            self.next_job += 1
            self.jobs[job.id] = job
            self.pending.append(job.id)

    def _dispatch(self):
        while self.pending:
            best = None
            for w in self.readers.values():
                if (w.alive and w.ready and len(w.inflight) < self.cfg.max_inflight_per_reader
                        and self.free_slots[w.wid]):
                    if best is None or len(w.inflight) < len(best.inflight):
                        best = w                               # least loaded first
            if best is None:
                return
            job = self.jobs[self.pending.popleft()]
            job.state, job.reader, job.inc = "in_reader", best.wid, best.inc
            job.slot = self.free_slots[best.wid].pop()
            best.inflight.append(job.id)
            best.send(PaperJob(job.id, job.paper, job.slot, job.attempt))

    # ------------------------------------------------------------------ main loop
    def run(self) -> PipelineReport:
        cfg = self.cfg
        t0 = time.monotonic()
        try:
            slots = cfg.max_outstanding + cfg.n_readers
            for wid in range(cfg.n_readers):
                self.arenas[wid] = shm.create_arena(slots, cfg.dim)
                self.free_slots[wid] = list(range(slots))
            self._spawn("writer", 0)
            for wid in range(cfg.n_readers):
                self._spawn("reader", wid)
            next_chaos = t0 + cfg.chaos_reader_interval
            writer_kills = 0
            next_wkill = None
            while not self.writer_done:
                now = time.monotonic()
                if cfg.timeout and now - t0 > cfg.timeout:
                    raise PipelineError(f"overall timeout after {cfg.timeout} s")
                self._refill()
                self._dispatch()
                if (self.src_done and not self.pending and self.finished == self.next_job
                        and not self.stopping):
                    self.stopping = True
                    if self.writer.alive:
                        self.writer.send(Stop())
                workers = [w for w in [*self.readers.values(), self.writer] if w and w.alive]
                waitables = {}
                for w in workers:
                    waitables[w.recv_conn] = w
                    waitables[w.proc.sentinel] = w
                try:
                    ready = connection.wait(list(waitables), timeout=0.05)
                except (OSError, ValueError):
                    ready = []
                dead = []
                for obj in ready:
                    w = waitables[obj]
                    if obj is w.proc.sentinel:
                        dead.append(w)
                        continue
                    try:
                        while w.alive and w.recv_conn.poll(0):
                            self._handle(w, w.recv_conn.recv())
                    except (EOFError, OSError):
                        dead.append(w)
                for w in workers:                              # belt and braces: is_alive can disagree
                    if w.alive and not w.proc.is_alive():
                        dead.append(w)
                for w in dict.fromkeys(dead):
                    if w.alive and not self.writer_done:
                        self._on_death(w)
                now = time.monotonic()
                for w in [*self.readers.values(), self.writer]:
                    if not w.alive or w.killed_by_us:
                        continue
                    if w.ready and now - w.last_seen > cfg.hang_timeout:
                        self.report.hang_kills += 1
                        self._kill(w, f"no heartbeat for {cfg.hang_timeout} s")
                    elif not w.ready and now - w.started > cfg.startup_timeout:
                        self._kill(w, "did not become ready in time")
                if cfg.chaos_reader_interval > 0 and not self.stopping and now >= next_chaos:
                    victims = [w for w in self.readers.values() if w.alive and w.ready and not w.killed_by_us]
                    if victims:
                        v = self.rng.choice(victims)
                        self.report.chaos_reader_kills += 1
                        self._kill(v, "chaos")
                    next_chaos = now + cfg.chaos_reader_interval
                if cfg.chaos_writer_kills and writer_kills < cfg.chaos_writer_kills and not self.stopping:
                    if self.writer.alive and self.writer.ready and not self.writer.killed_by_us:
                        if next_wkill is None:
                            next_wkill = now + cfg.chaos_writer_interval
                        elif now >= next_wkill:
                            writer_kills += 1
                            self.report.chaos_writer_kills += 1
                            next_wkill = None
                            self._kill(self.writer, "chaos")
            self.report.jobs = self.next_job
            self.report.outcomes = dict(self.outcomes)
            self.report.elapsed = time.monotonic() - t0
            if cfg.reader_limiter is not None:
                self.report.limiter_min_gap = cfg.reader_limiter.min_gap_seen()
            return self.report
        finally:
            self._shutdown()

    def _shutdown(self):
        workers = [w for w in [*self.readers.values(), self.writer] if w]
        for w in workers:
            if w.alive and w.role == "reader":
                w.send(Stop())
        deadline = time.monotonic() + 5
        for w in workers:
            try:
                w.proc.join(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                pass
            if w.proc.is_alive():
                w.proc.kill()
                w.proc.join(timeout=5)
            w.alive = False
            w.close()
        for a in self.arenas.values():
            try:
                a.close()
                a.unlink()
            except Exception:
                pass


def run_pipeline(papers, cfg: PipelineConfig) -> PipelineReport:
    return Supervisor(papers, cfg).run()
