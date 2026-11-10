#!/usr/bin/env bash
# Deliberately breaks the Phase 4 code one way at a time; each mutant must make the named tests FAIL.
# POSIX shell. Needs a CLEAN git tree: files are restored with `git checkout -- src`.  ONLY=G3 filters.
cd "$(git rev-parse --show-toplevel)" || exit 1
[ -z "$(git status --porcelain src)" ] || { echo "src/ has uncommitted changes; commit first"; exit 1; }
run() { # name file old new tests
  [ -n "$ONLY" ] && [[ "$1" != $ONLY* ]] && return
  python - "$2" "$3" "$4" <<'PY'
import sys
f,o,n=sys.argv[1:4]
s=open(f).read()
assert s.count(o)==1,(f,o,s.count(o))
open(f,'w').write(s.replace(o,n))
PY
  if [ $? -ne 0 ]; then echo "BADMUTANT $1"; git checkout -- src; return; fi
  if timeout 400 python -m pytest $5 -q -x >/dev/null 2>&1; then echo "SURVIVED $1"; else echo "KILLED   $1"; fi
  git checkout -- src
}
R=src/paperlens/graph/rank.py; O=src/paperlens/graph/openalex.py; M=src/paperlens/graph/minhash.py; I=src/paperlens/store/index.py
run G1_dangling_mass_dropped $R '(spread + r[dangling].sum() * tele)' '(spread)' "tests/test_graph.py -k pagerank"
run G2_ppr_ignores_seed_weights $R '            w[g.index[a]] += x' '            w[g.index[a]] += 1.0' "tests/test_graph.py -k personalized"
run G3_label_prop_never_keeps_own_label $R '            if lab[i] in best:' '            if False:' "tests/test_graph.py -k label"
run G4_budget_not_enforced $O 'if self.budget is not None and not self.budget.try_spend(self.cost):' 'if False:' "tests/test_graph.py -k budget"
run G5_resume_asks_again $O 'todo = [a for a in arxiv_ids if a not in done]' 'todo = list(arxiv_ids)' "tests/test_graph.py -k resume"
run G6_unknown_papers_not_remembered $O 'rec = found.get(a) or {"arxiv_id": a, "openalex_id": None, "refs": []}' 'rec = found.get(a) or {"arxiv_id": a, "openalex_id": "https://openalex.org/W0", "refs": []}' "tests/test_graph.py -k resume"
run G7_lsh_candidates_not_verified $M 'if jv >= threshold:' 'if True:' "tests/test_minhash.py"
run G8_dedupe_keeps_worst $M 'best = min(g, key=ranked_keys.index)' 'best = max(g, key=ranked_keys.index)' "tests/test_minhash.py -k clusters_merge"
run G9_shingles_case_sensitive $M '    words = _WORD.findall(text.lower())' '    words = _WORD.findall(text)' "tests/test_minhash.py -k shingles"
run G10_rerank_ignores_graph src/paperlens/graph/rerank.py '            s += weight / (rrf_k + graph_rank[a])' '            s += 0.0' "tests/test_minhash.py -k lifts"
run G11_graph_crc_unchecked src/paperlens/store/graphfile.py 'if zlib.crc32(body) & 0xFFFFFFFF != crc:' 'if False:' "tests/test_graph_publish.py -k corrupt"
run G12_old_graph_files_kept src/paperlens/store/graphfile.py '        if name != keep:' '        if False:' "tests/test_graph_publish.py"
run G13_recovery_skips_graph_cleanup $I 'stats["removed_graph_files"] = clean_graph_dir(self.path, self.manifest.graph_version)' 'stats["removed_graph_files"] = 0' "tests/test_graph_publish.py -k crash"
run G14_fsck_skips_graph src/paperlens/store/fsck.py '    _check_graph(index_dir, m.graph_version, rep)' '    pass' "tests/test_graph_publish.py -k corrupt"
run G15_manifest_before_file_is_visible src/paperlens/store/graphfile.py '    crashpoint("graph_after_file")
    write_manifest(index_dir, m.next(graph_version=version))' '    write_manifest(index_dir, m.next(graph_version=version))
    crashpoint("graph_after_file")' "tests/test_graph_publish.py -k crash"
