import pytest


@pytest.fixture(autouse=True)
def _no_proxy_for_localhost(monkeypatch):
    # Some environments route HTTP through a proxy; local test servers must not.
    for k in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(k, "127.0.0.1,localhost")
    for k in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
