cd "$(git rev-parse --show-toplevel)"
run() { # name file old new tests
  python - "$2" "$3" "$4" <<'PY'
import sys
f,o,n=sys.argv[1:4]
s=open(f).read()
assert s.count(o)==1,(f,o,s.count(o))
open(f,'w').write(s.replace(o,n))
PY
  if timeout 300 python -m pytest $5 -q -x >/dev/null 2>&1; then echo "SURVIVED $1"; else echo "KILLED   $1"; fi
  git checkout -- src
}
S=src/paperlens/pipeline
run P1_ack_before_commit $S/workers.py "                store.commit()                             # ONE fsync for the whole batch" "                pass" "tests/test_pipeline.py -k crash_point"
run P2_skip_crc_check $S/workers.py "if got != m.vec_crc:" "if False:" "tests/test_pipeline.py -k corrupt"
run P3_blame_everyone $S/supervisor.py "            if n == 0:                                         # only the job at the head is blamed" "            if True:" "tests/test_pipeline.py -k neighbours"
run P4_no_resend_to_new_writer $S/supervisor.py "                w.send(self.jobs[jid].ready)                  # re-send everything not yet acknowledged" "                pass" "tests/test_pipeline.py -k crash_point"
run P5_no_dead_letter $S/supervisor.py "if permanent or job.attempt >= self.cfg.max_attempts:" "if permanent:" "tests/test_pipeline.py -k exception"
run P6_limiter_sleep_ignored $S/shared_limiter.py "            self._sleep(delay)                                         # outside the lock, on purpose" "            pass" "tests/test_shared_limiter.py"
run P7_budget_overspend $S/shared_limiter.py "if self._spent.value + n > self.total:" "if False:" "tests/test_shared_limiter.py"
run P8_slot_never_freed $S/supervisor.py "            self.free_slots[job.reader].append(job.slot)" "            pass" "tests/test_pipeline.py -k plain_run"
