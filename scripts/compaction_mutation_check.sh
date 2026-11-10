#!/usr/bin/env bash
# Deliberately breaks the Phase 5 code one way at a time; each mutant must make the named tests FAIL.
# POSIX shell. Needs a CLEAN git tree: files are restored with `git checkout -- src`.  ONLY=C3 filters.
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
CP=src/paperlens/store/compaction.py; I=src/paperlens/store/index.py; PN=src/paperlens/store/pins.py
RD=src/paperlens/store/reader.py; QS=src/paperlens/server/query_server.py
run C1_dead_docs_kept_in_merge $CP '            if local in dead:' '            if False:' "tests/test_compaction.py -k 'oracle or merge_without or dead_ratio or everything'"
run C2_all_tombstones_dropped $CP '    tomb &= older_keys' '    tomb = set()' "tests/test_compaction.py -k tombstones"
run C3_all_tombstones_kept $CP '    tomb &= older_keys' '    pass' "tests/test_compaction.py -k tombstones"
run C4_gc_ignores_pins $I '            if name.startswith("seg-") and name not in listed and name not in pinned:' '            if name.startswith("seg-") and name not in listed:' "tests/test_compaction.py -k 'pins or pinned'"
run C5_recovery_ignores_pins $I '            elif name.startswith("seg-") and name not in listed and pinned is not None and name not in pinned:' '            elif name.startswith("seg-") and name not in listed:' "tests/test_compaction.py -k restart"
run C6_locate_forgets_pending_deletes $I '        for key in self.pending_tomb:                     # killed since the last flush (tombstone not written yet)' '        for key in ():' "tests/test_compaction.py -k 'pending or arrives'"
run C7_segment_not_published $I '                publish_segment(self.seg_root, tmp, name)
                new_name = name' '                new_name = name' "tests/test_compaction.py -k 'crash or merge_without'"
run C8_stale_pins_count_as_alive $PN '        probe.acquire()                        # we could lock it -> the owner is gone' '        raise IndexLocked("x")' "tests/test_compaction.py -k dead_readers"
run C9_vectors_dropped_in_merge $CP '    if vec_dim and any(d[4] is not None for d in docs):' '    if False:' "tests/test_compaction.py -k vectors"
run C10_policy_merges_anything $CP '        if max(w) <= ratio * max(1, min(w)):' '        if True:' "tests/test_compaction.py -k plan_run"
run C11_server_does_not_replace_dead_worker $QS '                if attempt == 0:' '                if False:  ' "tests/test_query_server.py -k killed"
run C12_reader_never_refreshes $RD '        if cur.generation == self.manifest.generation:' '        if True:' "tests/test_compaction.py -k frozen"
run C13_manifest_crashpoint_missing $I '                crashpoint("compact_after_manifest")' '                pass' "tests/test_compaction.py -k crash_during"
run C14_reader_serves_writes $RD '    put = delete = commit = flush = compact = publish_graph = _refuse' '    pass' "tests/test_compaction.py -k frozen"
