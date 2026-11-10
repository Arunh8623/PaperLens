"""Vectors must survive crashes TOGETHER with their paper (same diary entry, same segment, same flush)."""
import pytest

from paperlens.store.crashpoints import CRASH_EXIT_CODE
from tests import crashlib as cl
from tests.test_crash import CASES


@pytest.mark.parametrize("point,after", CASES)
@pytest.mark.parametrize("kind", ["plain", "messy"])
def test_crash_with_vectors_loses_no_paper_and_no_vector(tmp_path, point, after, kind):
    ix, ack = tmp_path / "ix", tmp_path / "ack.txt"
    n, seed = 150, 3
    r = cl.run_child(ix, ack, n, seed, kind, crash_at=point, crash_after=after, vec=True)
    assert r.returncode == CRASH_EXIT_CODE, f"{point} never fired (exit {r.returncode}): {r.stderr[-500:]}"
    cl.check_after_crash(ix, ack, kind, vec=True, n=n, seed=seed)
    cl.finish_and_verify(ix, ack, n, seed, kind, vec=True)
