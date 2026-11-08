"""Recall vs quantization bits (NFR7): how much do 1-4 bit codes lose, and what does rescoring win back?

    python bench/recall_vs_bits.py                                   # synthetic clustered vectors (default)
    python bench/recall_vs_bits.py --index data/ix --queries 300     # REAL vectors stored in one of your indexes
    python bench/recall_vs_bits.py --out docs/evidence/recall-bits-<what>.json

What is measured, per bit width b:
  bytes_per_vec / ratio_vs_float32   size of the code + its 3 correction numbers (12 bytes)
  recall10_codes_only                fraction of the EXACT top-10 found in the top 10 by quantized estimate
  recall10_rescored                  same, but take the top `pool` by estimate and re-score them exactly
                                     (this is what IndexStore.search_vector does)
  nfr7_pass                          ratio <= 1/8 AND recall10_rescored >= 0.95 (the brief's NFR7)
"Exact" = float32 cosine between unit vectors. Queries are perturbed copies of dataset vectors (synthetic)
or the stored vectors of random papers with the paper itself removed (--index).

HONESTY: the synthetic default shows the mechanism and costs; it says NOTHING about how well a real
embedding model's vectors quantize. Run it with --index on real vectors (after indexing with an
st: embedder) for the numbers that go in the report. The JSON records which data was used.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from paperlens.vectors import quant


def synthetic(n, dim, seed):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(40, dim))
    x = c[rng.integers(0, 40, n)] + 0.7 * rng.normal(size=(n, dim))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    q = x[rng.choice(n, 200, replace=False)] + 0.15 * rng.normal(size=(200, dim)).astype(np.float32)
    return x, (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32), "synthetic clustered gaussian"


def from_index(path, n_queries, seed):
    from paperlens.store.index import IndexStore
    with IndexStore(path) as st:
        ids = [a for a in sorted(st.live) if st.get_vector(a) is not None]
        if len(ids) < 200:
            raise SystemExit(f"{path}: only {len(ids)} papers have vectors; need >= 200")
        x = np.stack([st.get_vector(a) for a in ids]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    qi = rng.choice(len(x), min(n_queries, len(x)), replace=False)
    return x, qi, f"real vectors from {os.path.basename(os.path.abspath(path))} ({len(x)} papers)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--index")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.index:
        x, qi, label = from_index(a.index, a.queries, a.seed)
        queries = [(x[i], i) for i in qi]
    else:
        x, qs, label = synthetic(a.n, a.dim, a.seed)
        queries = [(q, None) for q in qs[:a.queries]]
    n, dim = x.shape
    c = x.mean(axis=0)
    rot = quant.rotation_matrix(dim)
    rows = []
    for bits in (1, 2, 3, 4):
        t0 = time.perf_counter()
        e = quant.encode(x, c, bits, rot)
        enc_s = time.perf_counter() - t0
        r_codes, r_res = [], []
        t1 = time.perf_counter()
        for q, self_i in queries:
            exact = x @ q
            if self_i is not None:
                exact[self_i] = -np.inf                           # a paper is not its own neighbour
            truth = set(np.argsort(-exact)[:10])
            est = quant.estimate(e["codes"], e["d_o"], e["inv"], e["oc"], quant.QueryPrep(q, c, rot), dim, bits)
            if self_i is not None:
                est[self_i] = -np.inf
            r_codes.append(len(truth & set(np.argsort(-est)[:10])) / 10)
            cand = np.argsort(-est)[:a.pool]
            r_res.append(len(truth & set(cand[np.argsort(-exact[cand])[:10]])) / 10)
        scan_ms = (time.perf_counter() - t1) / len(queries) * 1000
        per_vec = quant.code_bytes(dim, bits) + 12
        row = {"bits": bits, "bytes_per_vec": per_vec, "ratio_vs_float32": round(per_vec / (dim * 4), 4),
               "recall10_codes_only": round(float(np.mean(r_codes)), 4),
               "recall10_rescored": round(float(np.mean(r_res)), 4), "pool": a.pool,
               "encode_seconds": round(enc_s, 2), "ms_per_query_incl_exact_ground_truth": round(scan_ms, 1)}
        row["nfr7_pass"] = row["ratio_vs_float32"] <= 0.125 and row["recall10_rescored"] >= 0.95
        rows.append(row)
        print(row, flush=True)
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "data": label, "n": n, "dim": dim,
           "queries": len(queries), "rows": rows, "python": sys.version.split()[0],
           "platform": platform.platform(), "numpy": np.__version__,
           "float16_copy_bytes_per_vec": dim * 2,
           "disk_note": "disk per vector = code bytes + float16 copy (the copy is kept for rescoring); "
                        "codes save memory and scan time, not disk",
           "note": "synthetic data shows the mechanism only" if not a.index else "real vectors from an index"}
    print(json.dumps(out, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
