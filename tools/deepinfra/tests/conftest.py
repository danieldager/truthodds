import os
import threading

import pytest


@pytest.fixture(autouse=True)
def _key(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key-not-real")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPINFRA_BASE_URL", raising=False)
    # never let a test append to the machine's real observations file
    monkeypatch.setenv("DEEPINFRA_RUNNER_OBS", str(tmp_path / "obs.jsonl"))
    # nor share the machine's real slot pool: the pool stays ON (so the default path is
    # what the tests exercise) but in a directory of its own.
    monkeypatch.setenv("DEEPINFRA_POOL_DIR", str(tmp_path / "pool"))
    monkeypatch.delenv("DEEPINFRA_POOL", raising=False)
    monkeypatch.delenv("DEEPINFRA_POOL_CEILING", raising=False)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeTransport:
    """Records every POST and answers from a scripted handler.

    handler(body, n_calls_for_item) -> FakeResponse; default: a cheap successful chat reply."""

    def __init__(self, handler=None):
        self.handler = handler or (lambda body, n: ok_reply(body))
        self.calls = []
        self.per_item = {}
        self.lock = threading.Lock()

    def post(self, url, headers, body, timeout):
        marker = body["messages"][-1]["content"] if "messages" in body else str(body.get("input"))
        with self.lock:
            self.calls.append((url, marker, body))
            self.per_item[marker] = self.per_item.get(marker, 0) + 1
            n = self.per_item[marker]
        r = self.handler(body, n)
        if isinstance(r, Exception):
            raise r
        return r

    @property
    def n(self):
        return len(self.calls)


def ok_reply(body, content="yes", cost=None, prompt_tokens=10, completion_tokens=1):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if cost is not None:
        usage["estimated_cost"] = cost
    return FakeResponse(200, {"choices": [{"message": {"content": content}}], "usage": usage})


def busy():
    return FakeResponse(429, {}, text="Model busy")
