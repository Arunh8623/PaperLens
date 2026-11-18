"""Quick read (Gemini) against a LOCAL fake server. The request shape comes from Google's developer guide (read
2026-11-17); a green test here does NOT prove the real API accepts it -- that needs a real key (docs/PHASE8.md)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from paperlens.enrich import quickread as qrmod
from paperlens.enrich.quickread import QuickRead, QuickReadError, download_pdf
from paperlens.harvest.limiter import MinIntervalLimiter

GOOD = json.dumps({"tldr": "T", "problem": "P", "method": "M", "results": "R", "limitations": "L"})


class FakeGemini:
    """script: list of ("ok", text) | ("http", code, message) | ("raw", dict); the last item repeats."""

    def __init__(self, script, delay=0.0):
        self.script, self.calls, self.delay = list(script), [], delay
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                n = len(outer.calls)
                outer.calls.append({"path": self.path, "key": self.headers.get("x-goog-api-key"), "body": body})
                time.sleep(outer.delay)
                step = outer.script[min(n, len(outer.script) - 1)]
                if step[0] == "http":
                    out, code = json.dumps({"error": {"message": step[2]}}).encode(), step[1]
                elif step[0] == "raw":
                    out, code = json.dumps(step[1]).encode(), 200
                else:
                    out, code = json.dumps({"candidates": [{"content": {"parts": [{"text": step[1]}]}}]}).encode(), 200
                self.send_response(code); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown(); self.server.server_close()


@pytest.fixture
def gem():
    made = []

    def make(script, **kw):
        f = FakeGemini(script, **kw); made.append(f); return f
    yield make
    for f in made:
        f.close()


def _qr(tmp_path, fake, pdf=b"%PDF-1.5 fake", key="SECRET-KEY", **kw):
    return QuickRead(tmp_path / "qr", api_key=key, base_url=fake.url, pdf_fetch=lambda i: pdf, **kw)


def test_happy_path_sends_pdf_and_key_in_header_then_serves_from_cache(tmp_path, gem):
    f = gem([("ok", GOOD)])
    qr = _qr(tmp_path, f)
    r = qr.get("2101.00001", "A title", "An abstract")
    assert r["source"] == "pdf" and r["summary"]["tldr"] == "T" and not r["cached"] and r["model"] == "gemini-3-flash-preview"
    c = f.calls[0]
    assert c["path"] == "/v1beta/models/gemini-3-flash-preview:generateContent"
    assert c["key"] == "SECRET-KEY" and "SECRET-KEY" not in c["path"]            # header, never the URL
    parts = c["body"]["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "application/pdf" and parts[0]["inline_data"]["data"]
    assert "untrusted" in parts[1]["text"] and "A title" in parts[1]["text"]
    assert c["body"]["generationConfig"]["responseMimeType"] == "application/json"
    again = qr.get("2101.00001", "A title", "An abstract")
    assert again["cached"] and len(f.calls) == 1                                  # second click costs nothing
    assert QuickRead(tmp_path / "qr", api_key=None).cached("2101.00001")["summary"]["method"] == "M"   # survives restart


def test_falls_back_to_the_abstract_and_says_so(tmp_path, gem):
    f = gem([("ok", GOOD)])
    r = _qr(tmp_path, f, pdf=None).get("2101.00002", "T", "Abs")
    assert r["source"] == "abstract"
    parts = f.calls[0]["body"]["contents"][0]["parts"]
    assert len(parts) == 1 and "abstract only" in parts[0]["text"]


def test_a_400_about_a_setting_steps_down_to_simpler_settings(tmp_path, gem):
    f = gem([("http", 400, "thinking not supported"), ("ok", GOOD)])
    assert _qr(tmp_path, f).get("2101.00003", "T", "A")["summary"]["tldr"] == "T"
    assert "thinkingConfig" in f.calls[0]["body"]["generationConfig"]
    assert "thinkingConfig" not in f.calls[1]["body"]["generationConfig"]


def test_errors_become_clear_messages_and_nothing_is_cached(tmp_path, gem):
    for code, needle, status in ((429, "free-tier limit", 429), (404, "--gemini-model", 502), (403, "API key", 502)):
        f = gem([("http", code, "x")])
        qr = QuickRead(tmp_path / f"qr{code}", api_key="SECRET-KEY-123", base_url=f.url, pdf_fetch=lambda i: None)
        with pytest.raises(QuickReadError) as e:
            qr.get("2101.00004", "T", "A")
        assert needle in str(e.value) and e.value.status == status and "SECRET-KEY-123" not in str(e.value)
        assert qr.cached("2101.00004") is None


def test_no_key_means_503_and_no_call(tmp_path, gem, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    f = gem([("ok", GOOD)])
    qr = QuickRead(tmp_path / "qr", api_key=None, base_url=f.url, pdf_fetch=lambda i: None)
    assert not qr.available
    with pytest.raises(QuickReadError) as e:
        qr.get("2101.00005", "T", "A")
    assert e.value.status == 503 and not f.calls


def test_non_json_answer_is_kept_as_the_takeaway(tmp_path, gem):
    f = gem([("ok", "Sorry, here is prose, not JSON.")])
    r = _qr(tmp_path, f).get("2101.00006", "T", "A")
    assert r["summary"]["tldr"].startswith("Sorry") and r["summary"]["method"] == ""
    f2 = gem([("ok", "```json\n" + GOOD + "\n```")])                                # code fences are tolerated
    assert _qr(tmp_path / "b", f2).get("2101.00007", "T", "A")["summary"]["results"] == "R"


def test_blocked_or_empty_answers_are_reported(tmp_path, gem):
    f = gem([("raw", {"promptFeedback": {"blockReason": "SAFETY"}})])
    with pytest.raises(QuickReadError, match="declined"):
        _qr(tmp_path, f).get("2101.00008", "T", "A")
    f2 = gem([("raw", {"candidates": []})])
    with pytest.raises(QuickReadError, match="no answer"):
        _qr(tmp_path / "b", f2).get("2101.00008", "T", "A")


def test_two_simultaneous_clicks_make_one_api_call(tmp_path, gem):
    f = gem([("ok", GOOD)], delay=0.4)
    qr = _qr(tmp_path, f)
    out = []
    ts = [threading.Thread(target=lambda: out.append(qr.get("2101.00009", "T", "A"))) for _ in range(3)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(out) == 3 and len(f.calls) == 1 and sorted(o["cached"] for o in out) == [False, True, True]


def test_download_pdf_checks_what_it_got():
    class R:
        def __init__(self, b): self.b = b
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n): return self.b[:n]
    lim = MinIntervalLimiter(0.0)
    ok = lambda req, timeout: R(b"%PDF-1.7 ...")
    assert download_pdf("2101.00001", limiter=lim, opener=ok) == b"%PDF-1.7 ..."
    assert download_pdf("2101.00001", limiter=lim, opener=lambda r, timeout: R(b"<html>nope")) is None
    assert download_pdf("2101.00001", limiter=lim, opener=lambda r, timeout: R(b"%PDF" + b"x" * (qrmod.MAX_PDF_BYTES + 5))) is None
    def boom(req, timeout): raise OSError("offline")
    assert download_pdf("2101.00001", limiter=lim, opener=boom) is None
