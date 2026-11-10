"""Speed-up benchmark for the ingestion pipeline: 1/2/4/8 readers.

    python bench/pipeline_speedup.py --papers 4000 --burn-ms 5 --out docs/evidence/speedup-<machine>.json

IMPORTANT, read before quoting numbers:
* Run it on the machine you want to report. A sandbox with 1-2 cores cannot show 4-reader scaling.
* With --burn-ms 0 (default) only the REAL stand-in work runs (tokenise + hash-embed, well under a
  millisecond per paper): the single writer/fsync and process start-up dominate, so scaling will look
  poor. That is a true statement about this light workload, not a bug.
* With --burn-ms N each reader additionally spins the CPU for N ms per paper. That SIMULATES model
  inference (Phase 3 will use a real embedding model). The JSON records this, so nobody mistakes it
  for a measurement of a real model.
* efficiency(N) = T(1) / (N * T(N)); the brief's target is >= 0.70 at 4 readers on a >= 4-core machine.
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import tempfile
import time


def main(argv=None):
    from paperlens.pipeline.supervisor import PipelineConfig, run_pipeline
    from paperlens.pipeline.workers import Faults
    from paperlens.synth import generate_papers
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=4000)
    ap.add_argument("--readers", default="1,2,4,8")
    ap.add_argument("--burn-ms", type=float, default=0.0)
    ap.add_argument("--repeat", type=int, default=2, help="runs per setting; the best is reported")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    papers = generate_papers(a.papers, 1)
    rows = []
    for n in [int(x) for x in a.readers.split(",")]:
        times = []
        for _ in range(a.repeat):
            d = tempfile.mkdtemp(prefix="paperlens-bench-")
            try:
                rep = run_pipeline(papers, PipelineConfig(
                    index_path=d, n_readers=n, dim=64, flush_docs=2000,
                    faults=Faults(burn_seconds=a.burn_ms / 1000.0), timeout=3600))
                assert not rep.dead_letters and rep.outcomes == {"insert": a.papers}, rep
                times.append(rep.elapsed)
            finally:
                shutil.rmtree(d, ignore_errors=True)
        rows.append({"readers": n, "seconds": min(times), "all_runs": times,
                     "papers_per_second": a.papers / min(times)})
        print(rows[-1], flush=True)
    t1 = next((r["seconds"] for r in rows if r["readers"] == 1), None)
    for r in rows:
        if t1:
            r["speedup"] = t1 / r["seconds"]
            r["efficiency"] = t1 / (r["readers"] * r["seconds"])
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "papers": a.papers, "burn_ms_simulated": a.burn_ms,
           "cpu_count": os.cpu_count(), "platform": platform.platform(), "python": sys.version.split()[0],
           "rows": rows,
           "note": "burn_ms_simulated > 0 means reader CPU work is SIMULATED, not a real model"}
    print(json.dumps(out, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
