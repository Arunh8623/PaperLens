"""Messages exchanged between the supervisor and its worker processes.

Plain frozen dataclasses so they pickle the same way on Windows (spawn) and
Linux. Shapes follow docs/CONTRACTS.md section 7; fields added beyond the brief
(attempt, outcome, vec_crc on Ack) are marked.

Flow:   supervisor --PaperJob--> reader --PaperReady--> supervisor --PaperReady--> writer
        writer --AckBatch--> supervisor
Every worker also sends Ready (once, after start-up) and Heartbeat (regularly).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts.paper import PaperRecord


@dataclass(frozen=True)
class PaperJob:
    job_id: int
    paper: PaperRecord
    slot: int                 # shared-memory slot CHOSEN BY THE SUPERVISOR for this job's vector
    attempt: int = 0          # how many times this job was tried before (extra)


@dataclass(frozen=True)
class PaperReady:
    job_id: int
    reader_id: int
    key: str
    paper: PaperRecord
    term_freqs: dict          # computed by the reader so the single writer never tokenizes
    doc_len: int
    shm_name: str
    shm_slot: int
    vec_dim: int
    vec_crc: int              # crc32 of the vector bytes the reader wrote into the slot
    attempt: int = 0


@dataclass(frozen=True)
class Ack:
    job_id: int
    key: str
    seq: int                  # >= 1 stored, -1 duplicate/stale (already known), -2 rejected (retry)
    outcome: str = ""         # extra: insert | replace | duplicate | stale | bad_vector | error:...
    vec_crc: int = 0          # extra: crc of the vector bytes the writer actually read


@dataclass(frozen=True)
class AckBatch:
    acks: tuple


@dataclass(frozen=True)
class Heartbeat:
    role: str
    worker_id: int
    incarnation: int


@dataclass(frozen=True)
class Ready:
    role: str
    worker_id: int
    incarnation: int
    pid: int
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class JobFailed:
    job_id: int
    error: str
    permanent: bool = False   # True: retrying cannot help (e.g. paper has no indexable text)


@dataclass(frozen=True)
class WriterDone:
    stats: dict


@dataclass(frozen=True)
class Fatal:
    """A worker found a configuration problem no retry can fix (e.g. wrong vector dimension)."""
    message: str


class Stop:
    """Tell a worker to finish and exit (writer: commit + flush first)."""

    def __repr__(self):
        return "Stop()"

    def __eq__(self, other):
        return isinstance(other, Stop)

    def __hash__(self):
        return 1
