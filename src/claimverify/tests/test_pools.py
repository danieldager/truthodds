"""Fake-clock unit tests for the retry/timeout state machine in claimverify/pools.py
(adapted from tests/test_pools.py; every class it covers survived the trim).

No real sleeping, no network: RetryState/CircuitBreaker are pure (injected rng/now);
ResourcePool gets an injected sleep that advances a fake clock instantly. The only
near-real-time tests are the stream-stall ones, which use 30ms timeouts.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from claimverify.pools import (
    AttemptFailure,
    CallFatal,
    CircuitBreaker,
    LLMPool,
    OrchestrationConfig,
    Outcome,
    ResourcePool,
    RetryState,
    ScrapePool,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, s: float):
        self.sleeps.append(s)
        self.t += s
        await asyncio.sleep(0)  # yield to the loop without real waiting


def cfg(**over) -> OrchestrationConfig:
    return OrchestrationConfig(**over)


# =============================================================================
# RetryState (pure)
# =============================================================================

def test_transient_backoff_doubles_and_caps():
    st = RetryState(cfg(backoff_base=1.0, backoff_cap=60.0), rng=lambda: 0.0)
    sleeps = [st.decide(Outcome.CONNECT).sleep_s for _ in range(8)]
    assert sleeps == [1, 2, 4, 8, 16, 32, 60, 60]
    assert all(st.decide(Outcome.HTTP_5XX).action == "retry" for _ in range(5))  # unbounded


def test_jitter_is_multiplicative_and_bounded():
    st = RetryState(cfg(backoff_base=1.0, backoff_jitter=0.5), rng=lambda: 1.0)
    assert st.decide(Outcome.CONNECT).sleep_s == pytest.approx(1.5)  # 1 * (1 + 0.5*1.0)


def test_429_honors_retry_after():
    st = RetryState(cfg(backoff_base=1.0), rng=lambda: 0.0)
    d = st.decide(Outcome.HTTP_429, retry_after=17.0)
    assert d.action == "retry" and d.sleep_s == 17.0
    # ...but never sleeps LESS than the computed backoff
    st2 = RetryState(cfg(backoff_base=1.0), rng=lambda: 0.0)
    for _ in range(6):
        st2.decide(Outcome.CONNECT)
    assert st2.decide(Outcome.HTTP_429, retry_after=2.0).sleep_s == 60.0


def test_parse_fail_gets_exactly_one_reask():
    st = RetryState(cfg(parse_reasks=1))
    first = st.decide(Outcome.PARSE_FAIL)
    assert first.action == "retry" and first.sleep_s == 0.0
    assert st.decide(Outcome.PARSE_FAIL).action == "raise"


def test_length_cap_and_fatal_never_retry():
    st = RetryState(cfg())
    assert st.decide(Outcome.LENGTH_CAP).action == "raise"
    assert RetryState(cfg()).decide(Outcome.FATAL).action == "raise"


def test_attempt_log_records_every_attempt():
    st = RetryState(cfg())
    st.decide(Outcome.CONNECT, detail="boom")
    st.decide(Outcome.HTTP_429, detail="slow down")
    assert st.attempts == [(Outcome.CONNECT, "boom"), (Outcome.HTTP_429, "slow down")]


# =============================================================================
# CircuitBreaker (pure, fake now)
# =============================================================================

def make_breaker(clock, **over):
    c = cfg(breaker_window=10, breaker_min_events=4, breaker_err_rate=0.5,
            breaker_cooldown=30.0, **over)
    return CircuitBreaker(c, now=clock.now)


def test_breaker_stays_closed_below_threshold():
    clock = FakeClock()
    br = make_breaker(clock)
    for ok in [True, True, True, False, True, True, False, True]:
        br.record(ok)
    assert br.state == "closed" and br.acquire_wait() == 0.0


def test_breaker_trips_pauses_then_probe_recovers():
    clock = FakeClock()
    br = make_breaker(clock)
    for _ in range(4):
        br.record(False)
    assert br.state == "open" and br.trips == 1
    assert br.acquire_wait() == pytest.approx(30.0)      # full cooldown remaining
    clock.t = 29.0
    assert br.acquire_wait() == pytest.approx(1.0)
    clock.t = 31.0
    assert br.acquire_wait() == 0.0                      # the half-open probe passes
    assert br.state == "half-open"
    assert br.acquire_wait() == pytest.approx(0.5)       # everyone else polls
    br.record(True)                                      # probe succeeded
    assert br.state == "closed" and br.acquire_wait() == 0.0
    br.record(False)                                     # window was cleared: one error is fine
    assert br.state == "closed"


def test_breaker_probe_failure_reopens():
    clock = FakeClock()
    br = make_breaker(clock)
    for _ in range(4):
        br.record(False)
    clock.t = 31.0
    assert br.acquire_wait() == 0.0
    br.record(False)                                     # probe failed
    assert br.state == "open" and br.trips == 2
    assert br.acquire_wait() == pytest.approx(30.0)      # fresh cooldown from t=31


def test_breaker_ignores_stragglers_while_open():
    clock = FakeClock()
    br = make_breaker(clock)
    for _ in range(4):
        br.record(False)
    br.record(True)                                      # in-flight call from before the trip
    assert br.state == "open"


# =============================================================================
# ResourcePool.run (fake clock end-to-end)
# =============================================================================

def make_pool(clock, size=4, **over) -> ResourcePool:
    kw = dict(backoff_base=1.0, backoff_jitter=0.0, breaker_window=10,
              breaker_min_events=4, breaker_err_rate=0.9, breaker_cooldown=30.0)
    kw.update(over)
    c = cfg(**kw)
    return ResourcePool("test", size, c, sleep=clock.sleep, now=clock.now, rng=lambda: 0.0)


def failing_then_ok(n_failures: int, outcome=Outcome.CONNECT, result="done"):
    calls = {"n": 0}

    async def attempt():
        calls["n"] += 1
        if calls["n"] <= n_failures:
            raise AttemptFailure(outcome, f"fail {calls['n']}")
        return result
    return attempt, calls


@pytest.mark.asyncio
async def test_pool_retries_until_success_no_real_time():
    clock = FakeClock()
    pool = make_pool(clock)
    attempt, calls = failing_then_ok(3)
    assert await pool.run(attempt, label="x") == "done"
    assert calls["n"] == 4
    assert clock.sleeps == [1.0, 2.0, 4.0]               # exp backoff, zero real time
    assert pool.metrics.ok == 1
    assert pool.metrics.retries[Outcome.CONNECT] == 3
    assert pool.metrics.in_flight == 0


@pytest.mark.asyncio
async def test_pool_guaranteed_eventual_success_is_unbounded():
    clock = FakeClock()
    pool = make_pool(clock, breaker_min_events=10_000)   # breaker out of the way
    attempt, calls = failing_then_ok(50, outcome=Outcome.HTTP_5XX)
    assert await pool.run(attempt) == "done"
    assert calls["n"] == 51
    assert max(clock.sleeps) == 60.0                     # capped backoff


@pytest.mark.asyncio
async def test_pool_raises_callfatal_on_data_error():
    clock = FakeClock()
    pool = make_pool(clock)

    async def attempt():
        raise AttemptFailure(Outcome.LENGTH_CAP, "max_tokens=100 hit")
    with pytest.raises(CallFatal) as ei:
        await pool.run(attempt)
    assert ei.value.outcome is Outcome.LENGTH_CAP
    assert len(ei.value.attempts) == 1
    assert pool.metrics.fatal == 1


@pytest.mark.asyncio
async def test_pool_trips_breaker_then_recovers_via_probe():
    clock = FakeClock()
    pool = make_pool(clock, breaker_min_events=4, breaker_err_rate=0.9)
    attempt, calls = failing_then_ok(4)                  # 4 failures trip it; 5th succeeds
    assert await pool.run(attempt) == "done"
    assert pool.breaker.trips == 1
    assert pool.breaker.state == "closed"                # probe (the success) closed it
    assert any(s >= 5.0 for s in clock.sleeps)           # gate waited out the cooldown


@pytest.mark.asyncio
async def test_pool_semaphore_caps_concurrency():
    clock = FakeClock()
    pool = make_pool(clock, size=2)
    state = {"now": 0, "peak": 0}

    async def attempt():
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01)                        # real, tiny: forces overlap
        state["now"] -= 1
        return "ok"
    await asyncio.gather(*(pool.run(attempt) for _ in range(6)))
    assert state["peak"] <= 2
    assert pool.metrics.peak_in_flight <= 2


# =============================================================================
# LLM stream: TTFT/idle stalls, length cap, parse re-ask (fake client, 30ms timeouts)
# =============================================================================

def chunk(content=None, finish=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)],
        usage=None)


class FakeStream:
    def __init__(self, chunks, hang_at: int | None = None):
        self.chunks = list(chunks)
        self.hang_at = hang_at
        self.i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.hang_at is not None and self.i == self.hang_at:
            await asyncio.sleep(999)
        if not self.chunks:
            raise StopAsyncIteration
        self.i += 1
        return self.chunks.pop(0)

    async def close(self):
        pass


def make_llm(clock, streams) -> LLMPool:
    c = cfg(llm_ttft_timeout=0.03, llm_idle_timeout=0.03, backoff_jitter=0.0,
            breaker_min_events=10_000)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=make_create(streams))))
    # use_cache=False: identical (empty) messages across pools would otherwise replay
    # an earlier test's cached answer instead of exercising the stream path
    return LLMPool(c, client=fake, use_cache=False, sleep=clock.sleep, now=clock.now,
                   rng=lambda: 0.0)


def make_create(streams):
    it = iter(streams)

    async def create(**kw):
        return next(it)
    return create


@pytest.mark.asyncio
async def test_stream_ttft_stall_retried_then_succeeds():
    clock = FakeClock()
    good = FakeStream([chunk('{"a"'), chunk(": 1}"), chunk(finish="stop")])
    llm = make_llm(clock, [FakeStream([], hang_at=0), good])
    assert await llm.chat_json([], max_tokens=100) == {"a": 1}
    assert llm.metrics.retries[Outcome.STALL_TTFT] == 1
    assert llm.metrics.hangs == 1


@pytest.mark.asyncio
async def test_stream_idle_stall_mid_generation():
    clock = FakeClock()
    good = FakeStream([chunk('{"ok": true}'), chunk(finish="stop")])
    llm = make_llm(clock, [FakeStream([chunk('{"ok"')], hang_at=1), good])
    assert await llm.chat_json([], max_tokens=100) == {"ok": True}
    assert llm.metrics.retries[Outcome.STALL_IDLE] == 1


@pytest.mark.asyncio
async def test_finish_reason_length_is_fatal_data_error():
    clock = FakeClock()
    llm = make_llm(clock, [FakeStream([chunk('{"truncat'), chunk(finish="length")])])
    with pytest.raises(CallFatal) as ei:
        await llm.chat_json([], max_tokens=100)
    assert ei.value.outcome is Outcome.LENGTH_CAP


@pytest.mark.asyncio
async def test_parse_failure_gets_one_reask_then_raises():
    clock = FakeClock()
    bad1 = FakeStream([chunk("not json at all"), chunk(finish="stop")])
    good = FakeStream([chunk('prose then {"v": 2} trailing'), chunk(finish="stop")])
    llm = make_llm(clock, [bad1, good])
    assert await llm.chat_json([], max_tokens=100) == {"v": 2}   # brace-matcher + re-ask
    assert llm.metrics.retries[Outcome.PARSE_FAIL] == 1

    clock2 = FakeClock()
    llm2 = make_llm(clock2, [FakeStream([chunk("junk"), chunk(finish="stop")]),
                             FakeStream([chunk("junk"), chunk(finish="stop")])])
    with pytest.raises(CallFatal) as ei:
        await llm2.chat_json([], max_tokens=100)
    assert ei.value.outcome is Outcome.PARSE_FAIL


@pytest.mark.asyncio
async def test_bare_array_output_violates_dict_contract_and_is_reasked():
    """Seen live in the ramp: READ answered with the stances LIST unwrapped."""
    clock = FakeClock()
    bad = FakeStream([chunk('[{"claim": 0, "stance": "supports"}]'), chunk(finish="stop")])
    good = FakeStream([chunk('{"stances": []}'), chunk(finish="stop")])
    llm = make_llm(clock, [bad, good])
    assert await llm.chat_json([], max_tokens=100) == {"stances": []}
    assert llm.metrics.retries[Outcome.PARSE_FAIL] == 1


@pytest.mark.asyncio
async def test_slow_but_streaming_call_is_never_killed():
    """A call that keeps producing chunks under the idle timeout is healthy — no stall."""
    clock = FakeClock()

    class Trickle(FakeStream):
        async def __anext__(self):
            await asyncio.sleep(0.01)                    # < idle timeout, forever "slow"
            return await super().__anext__()

    many = [chunk('{"n"')] + [chunk(" ")] * 10 + [chunk(': 1}'), chunk(finish="stop")]
    llm = make_llm(clock, [Trickle(many)])
    assert await llm.chat_json([], max_tokens=100) == {"n": 1}
    assert llm.metrics.hangs == 0


# =============================================================================
# ScrapePool: per-domain politeness + global cap
# =============================================================================

@pytest.mark.asyncio
async def test_scrape_per_domain_cap():
    import threading
    peak = {"n": 0, "max": 0}
    lock = threading.Lock()

    def slow_scrape(url):
        with lock:
            peak["n"] += 1
            peak["max"] = max(peak["max"], peak["n"])
        import time as _t
        _t.sleep(0.05)
        with lock:
            peak["n"] -= 1
        return "text " * 30

    c = cfg(scrape_per_domain=2, scrape_pool_size=32)
    pool = ScrapePool(c, scrape_fn=slow_scrape)
    urls = [f"https://example.com/a{i}" for i in range(6)]
    out = await asyncio.gather(*(pool.scrape(u) for u in urls))
    assert all(out)
    assert peak["max"] <= 2                              # politeness held
    assert pool.metrics.ok == 6


@pytest.mark.asyncio
async def test_scrape_wall_timeout_frees_slot_counts_hang():
    def wedge(url):
        import time as _t
        _t.sleep(5)
        return "late"

    c = cfg(scrape_wall_timeout=0.05)
    pool = ScrapePool(c, scrape_fn=wedge)
    assert await pool.scrape("https://example.com/x") is None
    assert pool.metrics.hangs == 1
    assert pool.metrics.in_flight == 0
