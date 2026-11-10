# Phase 6: late-interaction rerank, "More like this", "Key papers"

What it does and why: after the first ranking, a second look compares query words with
paper words one by one (MaxSim, the idea of ColBERT) to reorder the top ~100.
"More like this" finds papers near a given paper; "Key papers" lists papers with the highest PageRank.

Built: `vectors/late.py` (hashed-trigram token encoder, int8 codes, MaxSim), `discover.py`,
CLI `similar`, `key-papers`, `search --late`, `bench/late_ablation.py`.
Tested (Linux): `tests/test_discover.py`, `tests/test_cli_graph.py`; mutation script `scripts/discover_mutation_check.sh`.
NOT tested: real ColBERT/PLAID-style model (the encoder is a stand-in), gain on real data.
Verdict by our own rule: on synthetic data the baseline is already at the ceiling, so the stage stays OFF by default.
