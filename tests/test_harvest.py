import json
import shutil
from datetime import datetime, timezone

import pytest

from paperlens.contracts.paper import PaperRecord, read_jsonl
from paperlens.harvest.arxiv_oai import (
    OAIClient, HarvestError, harvest, parse_response, parse_retry_after,
    summarize_jsonl, ARXIV_MIN_INTERVAL, ARXIV_RULE_SECONDS,
)
from paperlens.harvest.fake_oai_server import FakeOAIServer, record_xml, OAI_NS
from paperlens.harvest.limiter import MinIntervalLimiter
from paperlens.synth import generate_papers
from tests.fakes import FakeTime

CONTACT = "team@example.org"
N = 230  # 5 pages of 50 (last page has 30)


def strip_synth(p):
    return PaperRecord(**{**p.to_dict(), "synthetic": False})


def setup(tmp_path, server, ft, **kw):
    lim = MinIntervalLimiter(ARXIV_MIN_INTERVAL, ft.clock, ft.sleep)
    client = OAIClient(CONTACT, base_url=server.url, limiter=lim, **kw)
    return client, str(tmp_path / "out.jsonl"), str(tmp_path / "state.json")


def keys_in(path):
    return {p.key for p in read_jsonl(path)}


@pytest.fixture
def papers():
    return generate_papers(N, seed=11)


# ---------------------------------------------------------------- happy path
def test_full_harvest_pages_and_spacing(tmp_path, papers):
    ft = FakeTime()
    with FakeOAIServer(papers, page_size=50) as srv:
        client, out, st = setup(tmp_path, srv, ft)
        state = harvest(client, set_spec="cs", out_path=out, state_path=st, log=lambda *_: None)
    assert state["done"] and state["pages"] == 5 and client.requests_made == 5
    assert keys_in(out) == {p.key for p in papers}
    assert summarize_jsonl(out)["lines"] == N
    assert client.limiter.min_gap_seen() >= ARXIV_RULE_SECONDS      # NFR8 evidence (virtual clock)
    assert state["out_of_order"] == 0
    got = {p.key: p for p in read_jsonl(out)}
    assert all(got[p.key] == strip_synth(p) for p in papers)         # content survives the trip


def test_user_agent_carries_contact(tmp_path, papers):
    ft = FakeTime()
    with FakeOAIServer(papers[:10], page_size=50) as srv:
        client, out, st = setup(tmp_path, srv, ft)
        harvest(client, out_path=out, state_path=st, log=lambda *_: None)
        assert all(CONTACT in ua for ua in srv.user_agents) and len(srv.user_agents) == 1


def test_contact_is_required():
    for bad in ("", None, "no-at-sign"):
        with pytest.raises(ValueError):
            OAIClient(bad)


def test_rule_margin():
    assert ARXIV_MIN_INTERVAL >= ARXIV_RULE_SECONDS


# ------------------------------------------------ Retry-After / rate limiting
def test_429_retry_after_is_honoured(tmp_path, papers):
    ft = FakeTime()
    with FakeOAIServer(papers, page_size=50, faults={2: ("http", 429, "20")}) as srv:
        client, out, st = setup(tmp_path, srv, ft)
        state = harvest(client, out_path=out, state_path=st, log=lambda *_: None)
    h = client.limiter.history
    assert h[2] - h[1] >= 20                      # request 3 waited out the 20 s
    assert client.requests_made == 6              # 5 pages + the refused one
    assert keys_in(out) == {p.key for p in papers} and state["done"]
    assert client.limiter.min_gap_seen() >= ARXIV_RULE_SECONDS


def test_503_without_header_uses_backoff(tmp_path, papers):
    ft = FakeTime()
    with FakeOAIServer(papers, page_size=50, faults={1: ("http", 503, None)}) as srv:
        client, out, st = setup(tmp_path, srv, ft, backoff_base=5.0)
        harvest(client, out_path=out, state_path=st, log=lambda *_: None)
    h = client.limiter.history
    assert h[1] - h[0] >= 5.0
    assert keys_in(out) == {p.key for p in papers}


def test_retry_gives_up_after_max_retries(tmp_path, papers):
    ft = FakeTime()
    faults = {i: ("http", 503, None) for i in range(1, 30)}
    with FakeOAIServer(papers, page_size=50, faults=faults) as srv:
        client, out, st = setup(tmp_path, srv, ft, max_retries=3)
        with pytest.raises(HarvestError, match="giving up"):
            harvest(client, out_path=out, state_path=st, log=lambda *_: None)
    assert client.requests_made == 4              # 1 try + 3 retries


def test_huge_retry_after_stops_politely(tmp_path, papers):
    ft = FakeTime()
    with FakeOAIServer(papers, faults={1: ("http", 429, "999999")}) as srv:
        client, out, st = setup(tmp_path, srv, ft)
        with pytest.raises(HarvestError, match="try again later"):
            harvest(client, out_path=out, state_path=st, log=lambda *_: None)
    assert client.requests_made == 1


def test_non_retryable_error_fails_fast(tmp_path, papers):
    ft = FakeTime()
    with FakeOAIServer(papers, faults={1: ("http", 404, None)}) as srv:
        client, out, st = setup(tmp_path, srv, ft)
        with pytest.raises(HarvestError, match="404"):
            harvest(client, out_path=out, state_path=st, log=lambda *_: None)
    assert client.requests_made == 1


def test_parse_retry_after():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_retry_after("5") == 5.0 and parse_retry_after("0") == 0.0
    assert parse_retry_after("2.5") == 2.5
    for bad in (None, "", "  ", "-1", "abc", "Smarch 99"):
        assert parse_retry_after(bad, now) is None
    assert parse_retry_after("Thu, 01 Jan 2026 00:00:30 GMT", now) == 30.0
    assert parse_retry_after("Wed, 31 Dec 2025 23:00:00 GMT", now) == 0.0


# ------------------------------------------------------- flaky network / XML
@pytest.mark.parametrize("fault", [("drop",), ("garbage",)])
def test_dropped_connection_and_truncated_xml_are_retried(tmp_path, papers, fault):
    ft = FakeTime()
    with FakeOAIServer(papers, page_size=50, faults={2: fault}) as srv:
        client, out, st = setup(tmp_path, srv, ft)
        state = harvest(client, out_path=out, state_path=st, log=lambda *_: None)
    assert state["done"] and keys_in(out) == {p.key for p in papers}
    assert client.requests_made == 6


# ---------------------------------------------------------- expired tokens
def test_expired_token_restarts_from_newest_datestamp(tmp_path, papers):
    ft = FakeTime()
    with FakeOAIServer(papers, page_size=50, faults={3: ("expire",)}) as srv:
        client, out, st = setup(tmp_path, srv, ft)
        state = harvest(client, out_path=out, state_path=st, log=lambda *_: None)
        restart_q = srv.log[3][1]                 # the request AFTER the expiry
    assert state["restarts"] == 1 and state["done"]
    assert "resumptionToken" not in restart_q and "from" in restart_q
    assert restart_q["from"] == sorted(p.datestamp for p in papers)[99]  # newest of first 100
    assert keys_in(out) == {p.key for p in papers}                       # nothing lost
    assert summarize_jsonl(out)["lines"] > N                             # overlap day re-fetched


def test_token_that_keeps_expiring_gives_up(tmp_path, papers):
    ft = FakeTime()
    faults = {i: ("expire",) for i in range(2, 40)}
    with FakeOAIServer(papers, page_size=50, faults=faults) as srv:
        client, out, st = setup(tmp_path, srv, ft)
        with pytest.raises(HarvestError, match="keeps expiring"):
            harvest(client, out_path=out, state_path=st, max_restarts=3, log=lambda *_: None)


# --------------------------------------------------------- resume / crash
def test_resume_uses_saved_token_and_has_no_duplicates(tmp_path, papers):
    with FakeOAIServer(papers, page_size=50) as srv:
        c1, out, st = setup(tmp_path, srv, FakeTime())
        s1 = harvest(c1, out_path=out, state_path=st, max_records=100, log=lambda *_: None)
        assert not s1["done"] and s1["token"] and s1["lines_written"] == 100
        c2, _, _ = setup(tmp_path, srv, FakeTime())          # brand-new client = "restarted program"
        s2 = harvest(c2, out_path=out, state_path=st, log=lambda *_: None)
        first_q_after = srv.log[2][1]
    assert "resumptionToken" in first_q_after                # it resumed, not restarted
    assert s2["done"] and summarize_jsonl(out) == {
        **summarize_jsonl(out), "lines": N, "unique_keys": N}


def test_crash_before_state_save_refetches_page_but_loses_nothing(tmp_path, papers):
    """At-least-once: records are fsynced before the state file moves on."""
    with FakeOAIServer(papers, page_size=50) as srv:
        c1, out, st = setup(tmp_path, srv, FakeTime())
        harvest(c1, out_path=out, state_path=st, max_records=100, log=lambda *_: None)
        saved = str(tmp_path / "state_after_2_pages.json")
        shutil.copy(st, saved)
        harvest(setup(tmp_path, srv, FakeTime())[0], out_path=out, state_path=st,
                max_records=150, log=lambda *_: None)          # page 3 written + state saved
        shutil.copy(saved, st)                                  # "crash": state save for page 3 lost
        harvest(setup(tmp_path, srv, FakeTime())[0], out_path=out, state_path=st, log=lambda *_: None)
    s = summarize_jsonl(out)
    assert s["unique_keys"] == N                                # nothing missing
    assert s["lines"] == 150 + 130                              # page 3 fetched twice -> duplicates only


def test_state_never_claims_more_than_the_file_holds(tmp_path, papers, monkeypatch):
    """Ordering rule: records first, state second. Checked at EVERY state save."""
    import paperlens.harvest.arxiv_oai as mod
    real_save = mod._save_state
    out = str(tmp_path / "out.jsonl")
    seen = []

    def checking_save(path, state):
        try:
            with open(out, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
        except FileNotFoundError:
            lines = 0
        seen.append((state["lines_written"], lines))
        assert lines >= state["lines_written"], "state is ahead of the data file!"
        real_save(path, state)

    monkeypatch.setattr(mod, "_save_state", checking_save)
    with FakeOAIServer(papers, page_size=50) as srv:
        client, _, st = setup(tmp_path, srv, FakeTime())
        harvest(client, out_path=out, state_path=st, log=lambda *_: None)
    assert len(seen) == 5 and seen[-1] == (N, N)


def test_state_for_other_set_is_refused(tmp_path, papers):
    with FakeOAIServer(papers) as srv:
        client, out, st = setup(tmp_path, srv, FakeTime())
        harvest(client, set_spec="cs", out_path=out, state_path=st, max_records=1, log=lambda *_: None)
        with pytest.raises(HarvestError, match="state file is for set"):
            harvest(client, set_spec="physics", out_path=out, state_path=st, log=lambda *_: None)


def test_finished_harvest_is_not_repeated(tmp_path, papers):
    with FakeOAIServer(papers[:20]) as srv:
        client, out, st = setup(tmp_path, srv, FakeTime())
        harvest(client, out_path=out, state_path=st, log=lambda *_: None)
        n = client.requests_made
        harvest(client, out_path=out, state_path=st, log=lambda *_: None)
    assert client.requests_made == n


# ------------------------------------------------- content edge cases
def test_deleted_records_and_unknown_set(tmp_path, papers):
    with FakeOAIServer(papers[:30], deleted=[("old.0001", "2019-01-01")]) as srv:
        client, out, st = setup(tmp_path, srv, FakeTime())
        s = harvest(client, out_path=out, state_path=st, log=lambda *_: None)
        assert s["deleted_seen"] == 1 and summarize_jsonl(out)["lines"] == 30
        client2, out2, st2 = setup(tmp_path / "x" if False else tmp_path, srv, FakeTime())
        s2 = harvest(client2, set_spec="physics", out_path=str(tmp_path / "o2.jsonl"),
                     state_path=str(tmp_path / "s2.json"), log=lambda *_: None)
    assert s2["done"] and s2["lines_written"] == 0


def _wrap(records_xml):
    return (f'<OAI-PMH xmlns="{OAI_NS}"><ListRecords>{records_xml}'
            f"<resumptionToken/></ListRecords></OAI-PMH>").encode()


def test_roundtrip_all_records_through_xml(papers):
    page = parse_response(_wrap("".join(record_xml(p) for p in papers)))
    assert page.records == [strip_synth(p) for p in papers] and page.token is None


def test_parser_cleans_whitespace_and_skips_bad_records():
    good = (
        '<record><header><identifier>oai:arXiv.org:hep-th/9901001</identifier>'
        '<datestamp>2024-01-05T00:00:00Z</datestamp></header><metadata>'
        '<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>hep-th/9901001</id>'
        '<created>2024-01-01</created><authors><author><keyname>Rao</keyname>'
        '<forenames>A.  B.</forenames><suffix>Jr.</suffix></author></authors>'
        "<title>A  title\n  across lines</title><categories>cs.IR   cs.LG</categories>"
        "<abstract>\n  Some\n   text.\n</abstract><doi>10.1/x</doi></arXiv></metadata></record>")
    no_title = good.replace("<title>A  title\n  across lines</title>", "<title></title>") \
        .replace("hep-th/9901001", "bad.1")
    page = parse_response(_wrap(good + no_title))
    assert page.skipped_bad == 1 and "bad.1" in page.bad_examples[0]
    (p,) = page.records
    assert p.arxiv_id == "hep-th/9901001" and p.datestamp == "2024-01-05"
    assert p.title == "A title across lines" and p.abstract == "Some text."
    assert p.authors == ("A. B. Rao Jr.",) and p.categories == ("cs.IR", "cs.LG")
    assert p.doi == "10.1/x" and p.created == "2024-01-01" and p.updated == ""


def test_resumption_token_text_is_extracted():
    xml = f'<OAI-PMH xmlns="{OAI_NS}"><ListRecords><resumptionToken>abc|123</resumptionToken></ListRecords></OAI-PMH>'
    assert parse_response(xml.encode()).token == "abc|123"


# ----------------------------------------------- real clock, real spacing
def test_real_time_spacing_measured_by_the_server(tmp_path):
    papers = generate_papers(150, seed=2)
    interval = 0.2
    with FakeOAIServer(papers, page_size=50) as srv:
        client = OAIClient(CONTACT, base_url=srv.url, limiter=MinIntervalLimiter(interval))
        harvest(client, out_path=str(tmp_path / "o.jsonl"), state_path=str(tmp_path / "s.json"),
                log=lambda *_: None)
        times = [t for t, _ in srv.log]
    assert len(times) == 3
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) >= interval - 0.06, gaps       # measured at the SERVER; 60 ms slack for Windows' ~15 ms clock tick
