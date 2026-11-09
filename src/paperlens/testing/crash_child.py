"""A child process that ingests papers and records what it ACKNOWLEDGED.

    python -m paperlens.testing.crash_child INDEX ACKFILE N SEED KIND BATCH FLUSH_EVERY [SLEEP]

KIND "plain": N unique synthetic papers.   KIND "messy": the same plus duplicates,
newer versions and stale versions, shuffled.

It puts papers in batches of BATCH, calls commit(), and ONLY THEN appends the
keys of that batch to ACKFILE (one per line, flushed). So ACKFILE holds exactly
the papers whose commit() returned: the ones the system promised not to lose.
Every FLUSH_EVERY papers it calls flush(). It may be killed at any moment
(crash point env vars, or SIGKILL from the parent).
"""
import sys
import time

from ..store.index import IndexStore
from ..synth import generate_papers, messy_stream
from ..vectors.embedders import HashEmbedder


def make_stream(n: int, seed: int, kind: str):
    papers = generate_papers(n, seed)
    return papers if kind == "plain" else messy_stream(papers, seed)


def main(argv) -> int:
    index, ackfile, n, seed, kind, batch, flush_every = argv[:7]
    n, seed, batch, flush_every = int(n), int(seed), int(batch), int(flush_every)
    pause = float(argv[7]) if len(argv) > 7 else 0.0
    flags = set(argv[8:])
    skip_acked = "skip" in flags
    compact = "compact" in flags                        # merge neighbouring segments after every flush
    with_vec = "vec" in flags                           # also store a (hash) vector with every paper
    emb = HashEmbedder(16) if with_vec else None
    stream = make_stream(n, seed, kind)
    if skip_acked:
        # Like a supervisor: only resend what was never acknowledged.
        try:
            done = set(open(ackfile, "rb").read().decode("utf-8").split("\n")[:-1])
        except FileNotFoundError:
            done = set()
        stream = [p for p in stream if p.key not in done]
    st = IndexStore(index, flush_docs=10**9, ivf_min_docs=30 if with_vec else 0)   # we flush explicitly, to control timing
    with open(ackfile, "a", encoding="utf-8", newline="\n") as ack:
        since_flush = 0
        for i in range(0, len(stream), batch):
            chunk = stream[i:i + batch]
            for p in chunk:
                st.put(p, vector=emb.embed_paper(p) if with_vec else None)
            st.commit()
            for p in chunk:                               # promised durable only now
                ack.write(p.key + "\n")
            ack.flush()
            since_flush += len(chunk)
            if since_flush >= flush_every:
                st.flush()
                if compact:
                    st.compact(fan_in=2, ratio=100.0)
                since_flush = 0
            if pause:
                time.sleep(pause)
    st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
