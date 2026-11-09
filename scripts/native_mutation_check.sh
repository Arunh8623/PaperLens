#!/usr/bin/env bash
# Breaks the C++ source one way at a time, REBUILDS, and runs the differential tests: each mutant must FAIL them.
# Restores the source with `git checkout -- native` and rebuilds at the end. Commit first. Takes ~1 minute per mutant.
cd "$(git rev-parse --show-toplevel)" || exit 1
[ -z "$(git status --porcelain native)" ] || { echo "native/ has uncommitted changes; commit first"; exit 1; }
F=native/src/paperlens_native.cpp
run() { # name old new
  python - "$F" "$2" "$3" <<'PY'
import sys
f,o,n=sys.argv[1:4]
s=open(f).read()
assert s.count(o)==1,(o,s.count(o))
open(f,'w').write(s.replace(o,n))
PY
  if [ $? -ne 0 ]; then echo "BADMUTANT $1"; git checkout -- native; return; fi
  python scripts/build_native.py >/dev/null 2>&1 || { echo "BUILDFAIL $1"; git checkout -- native; return; }
  if timeout 300 python -m pytest tests/test_native.py -q -x >/dev/null 2>&1; then echo "SURVIVED $1"; else echo "KILLED   $1"; fi
  git checkout -- native
}
run N1_bm25_operation_order 'k1 * (1.0 - b + b * ddl / avgdl)' 'k1 * (1.0 - b + (b / avgdl) * ddl)'
run N2_dead_docs_scored 'if ((dead && dead[prev]) || (mask && !mask[prev])) continue;' 'if ((mask && !mask[prev])) continue;'
run N3_filter_ignored 'if ((dead && dead[prev]) || (mask && !mask[prev])) continue;' 'if ((dead && dead[prev])) continue;'
run N4_gap_off_by_one 'prev += static_cast<int64_t>(gap);
        if (prev >= n)' 'prev += static_cast<int64_t>(gap) + (i == 5);
        if (prev >= n)'
run N5_level_bit_order 'c |= ((row[bitpos >> 3] >> (7 - (bitpos & 7))) & 1) << j;' 'c |= ((row[bitpos >> 3] >> (7 - (bitpos & 7))) & 1) << (bits - 1 - j);'
run N6_varint_too_long_accepted 'if (shift == 63) break;  // 10th byte still says "more": too long (Python also stops after 10 groups)' 'if (shift == 63) { pos += 10; return result; }'
run N7_zero_tf_accepted 'if (gap < 1 || tf < 1) throw std::runtime_error("corrupt postings (gap or tf is zero)");
        prev += static_cast<int64_t>(gap);
        if (prev >= n)' 'prev += static_cast<int64_t>(gap);
        if (prev >= n)'
python scripts/build_native.py >/dev/null 2>&1 && echo "extension restored"
