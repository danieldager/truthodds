"""Runner behaviour, offline, against a fake transport."""
import json

import pytest
from conftest import FakeResponse, FakeTransport, busy, ok_reply

from deepinfra_runner import MODEL_WORKERS, cache_key, resolve_workers, run

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
NOSLEEP = lambda s: None            # noqa: E731
QUIET = lambda *a, **k: None        # noqa: E731


def items(n=5):
    return [{"id": f"i{k}", "text": f"item {k}"} for k in range(n)]


def req(it):
    return {"messages": [{"role": "user", "content": it["text"]}], "max_tokens": 8}


def go(its, tr, **kw):
    kw.setdefault("workers", 4)
    kw.setdefault("gate_target", 4)
    kw.setdefault("gate_floor", 2)
    return run(its, req, model=MODEL, transport=tr, log=QUIET, sleep=NOSLEEP, **kw)


# --- cache ---------------------------------------------------------------

def test_cache_hit_skips_the_call(tmp_path):
    cache = tmp_path / "c.jsonl"
    tr = FakeTransport()
    r1 = go(items(5), tr, cache_path=cache)
    assert len(r1.results) == 5 and tr.n == 5 and r1.cached == 0

    tr2 = FakeTransport()
    r2 = go(items(5), tr2, cache_path=cache)
    assert tr2.n == 0                          # every item served from cache
    assert r2.cached == 5 and r2.spent == 0.0
    assert r2.results == r1.results
    assert len(cache.read_text().strip().splitlines()) == 5


def test_cache_key_changes_with_prompt_and_model():
    a = cache_key("i0", {"messages": [{"role": "user", "content": "x"}]}, MODEL)
    b = cache_key("i0", {"messages": [{"role": "user", "content": "y"}]}, MODEL)
    c = cache_key("i0", {"messages": [{"role": "user", "content": "x"}]}, "other/model")
    d = cache_key("i0", {"max_tokens": 8, "messages": [{"role": "user", "content": "x"}]}, MODEL)
    e = cache_key("i0", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8}, MODEL)
    assert a != b and a != c and a != d
    assert d == e                              # key order does not matter


def test_failed_items_are_not_cached(tmp_path):
    cache = tmp_path / "c.jsonl"
    tr = FakeTransport(lambda body, n: FakeResponse(500, text="boom"))
    r = go(items(2), tr, cache_path=cache, attempts=2, sweeps=1)
    assert len(r.errors) == 2
    assert not cache.exists() or cache.read_text().strip() == ""


# --- retries -------------------------------------------------------------

def test_retry_on_429_then_success():
    tr = FakeTransport(lambda body, n: busy() if n <= 2 else ok_reply(body))
    r = go(items(3), tr, attempts=6)
    assert len(r.results) == 3 and not r.errors
    assert tr.n == 9                           # 3 items x (2 busy + 1 ok)
    assert r.gate.n_429_total == 6             # one gate event per ATTEMPT: 3 items x 2 busy
    assert r.gate.n_cuts == 0                  # too few samples to breach the window


def test_retry_on_transport_exception():
    tr = FakeTransport(lambda body, n: ConnectionError("reset") if n == 1 else ok_reply(body))
    r = go(items(2), tr, attempts=3)
    assert len(r.results) == 2 and not r.errors and tr.n == 4


def test_gives_up_after_bounded_attempts():
    tr = FakeTransport(lambda body, n: busy())
    r = go(items(1), tr, attempts=3, sweeps=0)
    assert len(r.errors) == 1 and tr.n == 3    # bounded: exactly `attempts` calls


# --- spend cap -----------------------------------------------------------

def test_spend_cap_stops_the_run():
    tr = FakeTransport(lambda body, n: ok_reply(body, cost=1.0))
    r = go(items(10), tr, workers=1, gate_target=1, gate_floor=1, max_spend=2.0)
    assert r.cap_hit
    assert tr.n == 2                           # cap bites after the 2nd $1 call
    assert len(r.results) == 2
    assert sum(1 for e in r.errors.values() if e == "spend-cap") == 8
    assert r.spent == 2.0


def test_cap_skipped_items_are_not_swept():
    tr = FakeTransport(lambda body, n: ok_reply(body, cost=1.0))
    r = go(items(6), tr, workers=1, gate_target=1, gate_floor=1, max_spend=1.0, sweeps=2)
    assert tr.n == 1 and r.sweeps == 0         # no sweep re-spends past the cap


def test_cost_from_price_table_when_provider_gives_none():
    tr = FakeTransport(lambda body, n: ok_reply(body, prompt_tokens=1_000_000,
                                                completion_tokens=1_000_000))
    r = go(items(1), tr)
    assert r.spent == pytest.approx(0.09 + 0.18)   # DeepSeek-V4-Flash list price


def test_price_override_hook():
    tr = FakeTransport(lambda body, n: ok_reply(body, prompt_tokens=1_000_000, completion_tokens=0))
    r = go(items(1), tr, prices={MODEL: (1.0, 0.0)})
    assert r.spent == pytest.approx(1.0)


# --- sweep ---------------------------------------------------------------

def test_failed_item_sweep_recovers():
    # item 1 fails every attempt of the first pass (6), succeeds on the sweep
    def handler(body, n):
        if body["messages"][-1]["content"] == "item 1" and n <= 6:
            return busy()
        return ok_reply(body)

    tr = FakeTransport(handler)
    r = go(items(3), tr, attempts=6, sweeps=2)
    assert r.sweeps == 1
    assert len(r.results) == 3 and not r.errors


def test_sweep_gives_up_and_reports():
    tr = FakeTransport(lambda body, n: busy())
    r = go(items(2), tr, attempts=2, sweeps=2)
    assert r.sweeps == 2 and len(r.errors) == 2
    assert tr.n == 2 * 2 * 3                   # 2 items x 2 attempts x (1 pass + 2 sweeps)


# --- per-model workers ---------------------------------------------------

def test_per_model_worker_override():
    assert resolve_workers("deepseek-ai/DeepSeek-V4-Flash") == 200
    assert resolve_workers("Qwen/Qwen3-235B-A22B-Instruct-2507") == 8
    assert MODEL_WORKERS["Qwen3-235B"] == 8
    assert resolve_workers("Qwen/Qwen3-235B-A22B-Instruct-2507", 64) == 64   # explicit wins


def test_heavy_model_gate_target_follows_workers():
    tr = FakeTransport()
    r = run(items(3), req, model="Qwen/Qwen3-235B-A22B-Instruct-2507", transport=tr,
            log=QUIET, sleep=NOSLEEP)
    assert r.gate.ceiling == 200
    assert r.gate.target <= 8 and r.gate.peak <= 8      # never more than 8 in flight
    assert r.gate.floor == 8


# --- observations --------------------------------------------------------

def test_observation_line_written(tmp_path):
    obs = tmp_path / "obs.jsonl"
    tr = FakeTransport()
    go(items(3), tr, obs_path=obs)
    rows = [json.loads(x) for x in obs.read_text().splitlines()]
    assert rows and set(rows[0]) >= {"ts", "model", "target", "inflight", "completed",
                                     "reads_s", "n429_window", "rate_429", "p50_latency",
                                     "p95_latency", "failed", "cuts"}
    assert rows[-1]["model"] == MODEL and rows[-1]["completed"] == 3


# --- parse / ids ---------------------------------------------------------

def test_parse_hook_and_parse_failure_is_an_item_failure():
    tr = FakeTransport()
    r = go(items(2), tr, parse=lambda j, it: j["choices"][0]["message"]["content"].upper())
    assert set(r.results.values()) == {"YES"}

    tr2 = FakeTransport()
    r2 = go(items(2), tr2, parse=lambda j, it: j["nope"], sweeps=0)
    assert len(r2.errors) == 2 and all("parse" in e for e in r2.errors.values())
