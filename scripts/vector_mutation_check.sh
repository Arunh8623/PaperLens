#!/usr/bin/env bash
# Deliberately breaks the Phase 3 code one way at a time; each mutant must make the named tests FAIL.
# POSIX shell. Needs a CLEAN git tree: files are restored with `git checkout -- src`.
cd "$(git rev-parse --show-toplevel)" || exit 1
[ -z "$(git status --porcelain src)" ] || { echo "src/ has uncommitted changes; commit first"; exit 1; }
run() { # name file old new tests
  [ -n "$ONLY" ] && [[ "$1" != $ONLY* ]] && return   # e.g. ONLY=V13 bash scripts/vector_mutation_check.sh
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
I=src/paperlens/store/index.py; S=src/paperlens/store/segment.py; Q=src/paperlens/vectors/quant.py
run V1_vector_not_in_diary $I '            entry["vec"] = base64.b64encode(vec).decode("ascii")' '            pass' "tests/test_vectors.py -k reopen_from_segments"
run V2_dead_docs_in_vector_search $I '                ok[list(self.dead[si])] = False' '                pass' "tests/test_vectors.py -k replacing_and_deleting"
run V3_filter_ignored_in_vector_search $I '                ok &= fm' '                pass' "tests/test_vectors.py -k filters_equal_brute_force"
run V4_estimator_without_correction $Q '        est = inv[a:b] * (u @ prep.q_rot)' '        est = (u @ prep.q_rot) / np.sqrt(dim)' "tests/test_vectors.py -k nearly_unbiased"
run V5_rrf_off_by_one src/paperlens/vectors/fusion.py '            score[key] = score.get(key, 0.0) + 1.0 / (k + pos)' '            score[key] = score.get(key, 0.0) + 1.0 / (k + pos + 1)' "tests/test_vectors.py -k rrf_hand"
run V6_fsck_vector_check_removed src/paperlens/store/fsck.py '        if seg.has_vectors:
            _check_vectors(seg, err)' '        pass' "tests/test_vectors.py -k cross_checks"
run V7_diary_vector_dropped_on_replay $I '                    self._apply_put(PaperRecord.from_dict(e["paper"]), vec=vec)' '                    self._apply_put(PaperRecord.from_dict(e["paper"]), vec=None)' "tests/test_vectors.py -k reopen_from_segments"
run V8_flush_drops_vectors $I '        if self.vec_dim and any(d.vec is not None for d in self.buf.values()):' '        if False:' "tests/test_vectors.py -k reopen_from_segments"
run V10_writer_forgets_vector src/paperlens/pipeline/workers.py 'vector=np.frombuffer(raw, dtype=np.float32) if store_vectors else None)' 'vector=None)' "tests/test_pipeline.py -k plain_run"
run V12_ivf_list_truncated src/paperlens/vectors/ivf.py 'self.order[self.offsets[i]:self.offsets[i + 1]]' 'self.order[self.offsets[i]:self.offsets[i + 1] - 1]' "tests/test_vectors.py -k ivf_probing"
run V13_zone_map_off_by_one src/paperlens/store/filters.py 'zone["year_max"] < self.year_from' 'zone["year_max"] <= self.year_from' "tests/test_vectors.py -k zone_map"
run V14_bm25_filter_ignored_in_segments $I 'if local in dead or (mask is not None and not mask[local]):' 'if local in dead:' "tests/test_vectors.py -k filters_equal_brute_force"
run V15_dimension_check_removed $I '        if self.vec_dim and v.size != self.vec_dim:' '        if False:' "tests/test_vectors.py -k bad_vectors"
