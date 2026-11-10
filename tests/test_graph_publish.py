import os
import subprocess
import sys

import pytest

from paperlens.graph.csr import Graph
from paperlens.store.crashpoints import CRASH_EXIT_CODE, GRAPH_POINTS
from paperlens.store.fsck import fsck
from paperlens.store.graphfile import GraphFileError, graph_path, load_graph, read_graph_file
from paperlens.store.index import IndexStore
from paperlens.store.manifest import read_manifest
from paperlens.synth import generate_papers

from tests.crashlib import child_env


def g(n, shift=0):
    ids = [f"p{i}" for i in range(n)]
    return Graph.from_edges(ids, [(ids[i], ids[(i + 1 + shift) % n]) for i in range(n)])


def test_publish_load_and_reopen(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        assert st.graph is None
        assert st.publish_graph(g(5)) == 1
        assert st.publish_graph(g(7)) == 2
        assert st.manifest.graph_version == 2
    with IndexStore(tmp_path / "ix") as st:
        assert st.graph.n == 7 and st.manifest.graph_version == 2
    names = sorted(os.listdir(tmp_path / "ix" / "graphs"))
    assert names == ["graph-000002.bin"]                       # old versions are deleted
    assert fsck(tmp_path / "ix").ok


def test_publishing_a_graph_does_not_disturb_papers(tmp_path):
    papers = generate_papers(30, 1)
    with IndexStore(tmp_path / "ix", flush_docs=10) as st:
        for p in papers:
            st.put(p)
        st.commit()
        st.flush()
        before = st.stats()
        st.publish_graph(g(4))
        assert st.stats()["generation"] == before["generation"] + 1
        assert st.stats()["last_flushed_seq"] == before["last_flushed_seq"]
    with IndexStore(tmp_path / "ix") as st:
        assert len(st.live_keys()) == 30


def test_corrupt_graph_file_is_detected_by_fsck_and_by_open(tmp_path):
    with IndexStore(tmp_path / "ix") as st:
        st.publish_graph(g(6))
    path = graph_path(str(tmp_path / "ix"), 1)
    data = bytearray(open(path, "rb").read())
    data[data.rindex(b'p')] ^= 0x01                            # 'p' -> 'q' inside the ids text: still parses, only the crc notices
    open(path, "wb").write(bytes(data))
    rep = fsck(tmp_path / "ix")
    assert not rep.ok and any("graph v1" in e for e in rep.errors)
    with pytest.raises(GraphFileError):
        read_graph_file(path)
    with pytest.raises(Exception):
        IndexStore(tmp_path / "ix")


CHILD = ("import sys\n"
         "from paperlens.graph.csr import Graph\n"
         "from paperlens.store.index import IndexStore\n"
         "ids=[f'p{i}' for i in range(8)]\n"
         "st=IndexStore(sys.argv[1])\n"
         "st.publish_graph(Graph.from_edges(ids[:5], [('p0','p1')]))\n"          # version 1, no crash env yet matters (hit 1)
         "st.publish_graph(Graph.from_edges(ids, [('p0','p1'),('p1','p2')]))\n"  # version 2: crash on hit 2
         "st.close()\n")


@pytest.mark.parametrize("point,live_version,live_nodes", [("graph_after_file", 1, 5), ("graph_after_manifest", 2, 8)])
def test_crash_while_publishing_a_graph_leaves_exactly_one_consistent_version(tmp_path, point, live_version, live_nodes):
    assert point in GRAPH_POINTS
    ix = str(tmp_path / "ix")
    env = child_env(PAPERLENS_CRASH_AT=point, PAPERLENS_CRASH_AFTER=2)
    r = subprocess.run([sys.executable, "-c", CHILD, ix], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == CRASH_EXIT_CODE, r.stderr
    rep = fsck(ix)                                             # BEFORE recovery: leftovers are only warnings
    assert rep.ok, rep.errors
    assert read_manifest(ix).graph_version == live_version
    with IndexStore(ix) as st:                                 # recovery removes leftovers
        assert st.graph.n == live_nodes
        assert st.recovery["removed_graph_files"] == 1
    assert fsck(ix).ok and not fsck(ix).warnings
    assert sorted(os.listdir(os.path.join(ix, "graphs"))) == [f"graph-{live_version:06d}.bin"]
    assert load_graph(ix).n == live_nodes
