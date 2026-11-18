#!/usr/bin/env bash
# Deliberately breaks the Phase 6 code one way at a time; each mutant must make the named tests FAIL.
# POSIX shell. Needs a CLEAN git tree: files are restored with `git checkout -- src`.  ONLY=D3 filters.
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
DS=src/paperlens/discover.py; LT=src/paperlens/vectors/late.py
run D1_mlt_returns_itself $DS 'return [(h, how) for h in hits if h.arxiv_id != arxiv_id][:k]' 'return [(h, how) for h in hits][:k]' "tests/test_discover.py -k more_like_this"
run D2_key_papers_ascending $DS 'key=lambda a: (-float(pr[g.index[a]]), a)' 'key=lambda a: (float(pr[g.index[a]]), a)' "tests/test_discover.py -k key_papers"
run D3_quantizer_scale_wrong $LT 'np.abs(mat).max(axis=1), 1e-12) / 127.0' 'np.abs(mat).max(axis=1), 1e-12) / 60.0' "tests/test_discover.py -k 8bit"
run D4_maxsim_uses_max_not_sum $LT '(query_tokens @ doc_tokens.T).max(axis=1).sum()' '(query_tokens @ doc_tokens.T).max(axis=1).max()' "tests/test_discover.py -k maxsim"
run D5_late_tail_reordered $LT 'for j, h in enumerate(tail)]' 'for j, h in enumerate(tail[::-1])]' "tests/test_discover.py -k late_rerank"
run D6_late_ignores_encoder_dedupe $LT 'toks = list(dict.fromkeys(tokenize(text)))[: self.max_tokens]' 'toks = list(tokenize(text))[: self.max_tokens]' "tests/test_discover.py -k token_encoder"
