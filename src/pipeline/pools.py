"""Pooled async clients for the post-verify loop — concurrency lives HERE, not in pipeline code.

Architecture (design: docs/verify_tweet_claims_loop_design.md, measurements: clog/100726.md):
each external resource gets ONE pooled client owning its concurrency budget (semaphore),
timeout/retry policy, circuit breaker, and metrics:

    llm     DeepInfra chat (streaming; TTFT + inter-chunk idle timeouts kill hangs)
    serper  Google SERP search
    exa     neural search — NEVER live-tested (1k free req/mo); sized from docs only
    scrape  pipeline/search.py scrape() in threads, per-domain politeness + global cap

The per-post pipeline is a plain sequential async function that awaits these clients; the
harness (pipeline/harness.py) runs K posts concurrently and pool saturation emerges on its
own. Add/remove/reorder pipeline steps freely — concurrency never needs re-tuning.

Retry contract (guaranteed eventual success on transient failures):
  retried, unbounded, exp backoff + jitter : connect errors, 5xx, 429 (honors Retry-After),
                                             TTFT stall, inter-chunk idle stall
  retried once ("one re-ask")              : LLM JSON parse failure
  never retried (deterministic data error) : finish_reason=length (truncated JSON — raise),
                                             non-429 4xx (config error)
  NEVER retried for being slow             : a call that is still streaming is healthy.

Circuit breaker: if a pool's recent error rate spikes, the whole pool pauses (loud log),
then half-opens with a single probe. The retry/backoff/breaker state machines are pure
(injectable clock/rng/sleep) — see tests/test_pools.py for the fake-clock tests.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from config import (
    EXA_API_KEY,
    EXA_ENDPOINT,
    SERPER_ENDPOINT,
    VERIFICATION_API_KEY,
    VERIFICATION_BASE_URL,
    VERIFICATION_MODEL,
)
from pipeline import disk_cache, search

log = logging.getLogger("pools")


# =============================================================================
# THE config block — every orchestration knob lives here and nowhere else.
# =============================================================================

@dataclass
class OrchestrationConfig:
    # --- LLM pool (DeepInfra, DeepSeek-V4-Flash) -------------------------------------------
    # Documented limit 200 concurrent/model; measured 2026-07-10: latency flat to 192 workers,
    # zero 429s, ~1% hang tail at 96+. 150 = validated headroom.
    llm_pool_size: int = 150
    llm_model: str = VERIFICATION_MODEL
    llm_ttft_timeout: float = 30.0        # s to first streamed chunk (measured TTFT 1-2s)
    llm_idle_timeout: float = 30.0        # s between chunks (hung calls sit ~180s; this kills them fast)
    llm_connect_timeout: float = 15.0

    # --- Serper pool ------------------------------------------------------------------------
    # Documented (2026-07-10 research): QPS-based limits by tier — Starter 50 / Standard 100 /
    # Scale 200 / Ultimate 300 QPS. 8 concurrent ≈ 5-10 QPS effective at ~1-2s/query, an order
    # of magnitude under even Starter. 429 semantics UNDOCUMENTED (no known Retry-After) →
    # blind exponential backoff; don't raise past ~20 without a ramp.
    serper_pool_size: int = 8
    serper_timeout: float = 20.0          # non-streaming; whole-request wall

    # --- Exa pool — docs-sized ONLY, never live-tested (1k free req/mo, house frugality) ----
    # Official docs: flat 10 QPS on /search (all tiers); 429 body has no Retry-After header.
    # 3 concurrent ≈ 1-3 QPS effective — safely under.
    exa_pool_size: int = 3
    exa_timeout: float = 30.0

    # --- Scrape pool --------------------------------------------------------------------
    scrape_pool_size: int = 32            # global concurrent scrapes
    scrape_per_domain: int = 2            # politeness: concurrent scrapes per domain
    scrape_wall_timeout: float = 120.0    # abandon a wedged scrape thread (slot freed; counted as hang)

    # --- Retry policy (shared by all pools) --------------------------------------------------
    backoff_base: float = 1.0             # first retry sleep (doubles per transient attempt)
    backoff_cap: float = 60.0
    backoff_jitter: float = 0.5           # sleep *= 1 + U(0, jitter)
    parse_reasks: int = 1                 # JSON parse failure: one re-ask, then raise

    # --- Circuit breaker (per pool) ----------------------------------------------------------
    breaker_window: int = 30              # sliding window of attempt outcomes
    breaker_min_events: int = 10          # don't judge before this many outcomes
    breaker_err_rate: float = 0.5         # error fraction that trips the pool OPEN
    breaker_cooldown: float = 30.0        # s paused before the half-open probe

    # --- Harness ------------------------------------------------------------------------------
    k_posts: int = 300                    # posts in flight (posts mostly wait on pools; K is cheap)
    progress_every: float = 10.0          # s between flushed status lines
    stall_alarm_s: float = 300.0          # loud alarm if no post completes for this long
    n_shards: int = 16                    # checkpoint jsonl shards


# =============================================================================
# Outcomes + pure retry state machine
# =============================================================================

class Outcome(str, Enum):
    OK = "ok"
    CONNECT = "connect"          # DNS/TCP/TLS/read errors, mid-stream drops
    HTTP_5XX = "5xx"
    HTTP_429 = "429"
    STALL_TTFT = "stall_ttft"    # no first chunk within ttft_timeout
    STALL_IDLE = "stall_idle"    # no next chunk within idle_timeout
    PARSE_FAIL = "parse_fail"    # LLM output not valid JSON
    LENGTH_CAP = "length_cap"    # finish_reason=length — truncated output, data error
    FATAL = "fatal"              # non-429 4xx etc — config error, retrying can't help


TRANSIENT = {Outcome.CONNECT, Outcome.HTTP_5XX, Outcome.HTTP_429,
             Outcome.STALL_TTFT, Outcome.STALL_IDLE}


class AttemptFailure(Exception):
    """One attempt failed. Raised inside an attempt fn; classified for the retry machine."""

    def __init__(self, outcome: Outcome, detail: str = "", retry_after: float | None = None):
        super().__init__(f"{outcome.value}: {detail}")
        self.outcome = outcome
        self.detail = detail
        self.retry_after = retry_after


class CallFatal(Exception):
    """A pooled call gave up: deterministic data error (length cap, repeated parse failure)
    or config error (4xx). Transient failures never surface as this — they retry forever."""

    def __init__(self, outcome: Outcome, detail: str, attempts: list):
        super().__init__(f"{outcome.value} after {len(attempts)} attempt(s): {detail}")
        self.outcome = outcome
        self.detail = detail
        self.attempts = attempts


@dataclass
class Decision:
    action: str            # "retry" | "raise"
    sleep_s: float = 0.0


class RetryState:
    """Pure per-call retry state machine — no IO, no clock; rng injectable for tests.

    Transient outcomes retry forever (exp backoff + jitter, 429 honors Retry-After);
    PARSE_FAIL retries `parse_reasks` times; LENGTH_CAP/FATAL raise immediately.
    """

    def __init__(self, cfg: OrchestrationConfig, rng: Callable[[], float] = random.random):
        self.cfg = cfg
        self.rng = rng
        self.attempts: list[tuple[Outcome, str]] = []   # per-call attempt log
        self._transient_n = 0
        self._parse_n = 0

    def decide(self, outcome: Outcome, *, retry_after: float | None = None,
               detail: str = "") -> Decision:
        self.attempts.append((outcome, detail))
        if outcome in TRANSIENT:
            self._transient_n += 1
            sleep = min(self.cfg.backoff_cap,
                        self.cfg.backoff_base * 2 ** (self._transient_n - 1))
            sleep *= 1 + self.cfg.backoff_jitter * self.rng()
            if outcome is Outcome.HTTP_429 and retry_after:
                sleep = max(sleep, retry_after)
            return Decision("retry", sleep)
        if outcome is Outcome.PARSE_FAIL:
            self._parse_n += 1
            if self._parse_n <= self.cfg.parse_reasks:
                return Decision("retry", 0.0)   # immediate re-ask
            return Decision("raise")
        return Decision("raise")                # LENGTH_CAP, FATAL


class CircuitBreaker:
    """Pool-level breaker, pure logic (injectable now()).

    CLOSED: everything passes; outcomes fill a sliding window. If error fraction >= err_rate
    over >= min_events outcomes -> OPEN (trip): acquire_wait() returns the remaining cooldown.
    After cooldown -> HALF-OPEN: exactly one probe passes (others poll); the next recorded
    outcome closes (success, window cleared) or re-opens (failure, fresh cooldown).
    """

    def __init__(self, cfg: OrchestrationConfig, now: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self._now = now
        self._events: deque[bool] = deque(maxlen=cfg.breaker_window)
        self.state = "closed"                  # closed | open | half-open
        self._opened_at = 0.0
        self._probe_out = False
        self.trips = 0

    def acquire_wait(self) -> float:
        """0.0 -> caller may proceed; >0 -> wait that long and call again."""
        if self.state == "closed":
            return 0.0
        remaining = self.cfg.breaker_cooldown - (self._now() - self._opened_at)
        if remaining > 0:
            return remaining
        if not self._probe_out:                # cooldown over: release ONE probe
            self._probe_out = True
            self.state = "half-open"
            return 0.0
        return 0.5                             # probe in flight; poll shortly

    def record(self, ok: bool) -> None:
        if self.state == "open":
            return                             # stragglers from before the trip: ignore
        if self.state == "half-open":          # first outcome decides (probe, in practice)
            self._probe_out = False
            if ok:
                self.state = "closed"
                self._events.clear()
            else:
                self.state = "open"
                self._opened_at = self._now()
                self.trips += 1
            return
        self._events.append(ok)
        if len(self._events) >= self.cfg.breaker_min_events:
            err = self._events.count(False) / len(self._events)
            if err >= self.cfg.breaker_err_rate:
                self.state = "open"
                self._opened_at = self._now()
                self.trips += 1


# =============================================================================
# Metrics
# =============================================================================

class PoolMetrics:
    def __init__(self):
        self.in_flight = 0
        self.peak_in_flight = 0
        self.ok = 0
        self.fatal = 0
        self.retries: Counter = Counter()      # Outcome -> count (each = one failed attempt)
        self.latencies: deque[float] = deque(maxlen=500)
        self.cache_hits = 0
        self.misses = 0                        # scrape: None results (legitimate outcome)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_ratelimit: dict = {}         # last seen X-RateLimit-* / Retry-After headers

    @property
    def hangs(self) -> int:
        return self.retries[Outcome.STALL_TTFT] + self.retries[Outcome.STALL_IDLE]

    def pct(self, q: float) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[min(len(s) - 1, int(q * len(s)))]

    def line(self, name: str, size: int) -> str:
        parts = [f"{name} {self.in_flight}/{size}", f"ok {self.ok}", f"p50 {self.pct(0.5):.1f}s"]
        n_retry = sum(self.retries.values())
        if n_retry:
            parts.append(f"rty {n_retry}")
        if self.hangs:
            parts.append(f"hang {self.hangs}")
        if self.fatal:
            parts.append(f"FATAL {self.fatal}")
        if self.cache_hits:
            parts.append(f"cache {self.cache_hits}")
        return " ".join(parts)


# =============================================================================
# Pool base
# =============================================================================

class ResourcePool:
    """Semaphore + retry loop + breaker + metrics around single-attempt coroutines.

    Subclasses expose typed methods that build an `attempt` closure (raising AttemptFailure
    on any classified failure) and pass it to run(). A call holds its pool slot across its
    retries — a retrying call is still demand on the resource.
    """

    def __init__(self, name: str, size: int, cfg: OrchestrationConfig, *,
                 sleep: Callable[[float], Awaitable] = asyncio.sleep,
                 now: Callable[[], float] = time.monotonic,
                 rng: Callable[[], float] = random.random):
        self.name = name
        self.size = size
        self.cfg = cfg
        self._sem = asyncio.Semaphore(size)
        self._sleep = sleep
        self._now = now
        self._rng = rng
        self.breaker = CircuitBreaker(cfg, now=now)
        self.metrics = PoolMetrics()

    async def _breaker_gate(self, label: str) -> None:
        warned = False
        while (w := self.breaker.acquire_wait()) > 0:
            if not warned:
                log.error("[%s] CIRCUIT OPEN (trip #%d) — pool paused, %s waiting %.0fs",
                          self.name, self.breaker.trips, label, w)
                warned = True
            await self._sleep(min(w, 5.0))

    async def run(self, attempt_fn: Callable[[], Awaitable[Any]], *, label: str = "") -> Any:
        async with self._sem:
            state = RetryState(self.cfg, rng=self._rng)
            while True:
                await self._breaker_gate(label)
                self.metrics.in_flight += 1
                self.metrics.peak_in_flight = max(self.metrics.peak_in_flight,
                                                  self.metrics.in_flight)
                t0 = self._now()
                try:
                    result = await attempt_fn()
                except AttemptFailure as f:
                    self.breaker.record(False)
                    self.metrics.retries[f.outcome] += 1
                    d = state.decide(f.outcome, retry_after=f.retry_after, detail=f.detail)
                    if d.action == "raise":
                        self.metrics.fatal += 1
                        raise CallFatal(f.outcome, f.detail, state.attempts) from f
                    log.warning("[%s] %s attempt %d failed (%s: %.120s) — retry in %.1fs",
                                self.name, label, len(state.attempts), f.outcome.value,
                                f.detail, d.sleep_s)
                    await self._sleep(d.sleep_s)
                    continue
                finally:
                    self.metrics.in_flight -= 1
                self.breaker.record(True)
                self.metrics.ok += 1
                self.metrics.latencies.append(self._now() - t0)
                return result


# =============================================================================
# LLM pool (DeepInfra, streaming)
# =============================================================================

def _extract_json(text: str) -> dict:
    """Parse the model's JSON object; tolerate prose wrapping by brace-matching the first {…}."""
    text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        return obj
    if obj is not None:
        # parsed fine but a bare array/scalar violates the chat_json dict contract
        # -> PARSE_FAIL (one re-ask). Seen live: READ returning the stances list unwrapped.
        raise ValueError(f"top-level JSON is {type(obj).__name__}, not object")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in output head={text[:80]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in output head={text[:80]!r}")


class LLMPool(ResourcePool):
    """Streaming chat against DeepInfra. json_object mode, non-thinking (matches verify.py).

    Timeouts: TTFT and inter-chunk idle — a call that is still streaming is healthy and is
    NEVER killed for total duration. finish_reason=length raises CallFatal (data error:
    raise the caller's max_tokens, don't re-bill a doomed generation).
    """

    def __init__(self, cfg: OrchestrationConfig, *, client=None, **inj):
        super().__init__("llm", cfg.llm_pool_size, cfg, **inj)
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=VERIFICATION_BASE_URL, api_key=VERIFICATION_API_KEY,
                max_retries=0,   # retries are OUR job — the SDK's would fight the policy
                timeout=httpx.Timeout(connect=cfg.llm_connect_timeout,
                                      read=cfg.llm_idle_timeout * 3, write=30.0, pool=60.0),
            )
        self._client = client

    async def chat_json(self, messages: list[dict], *, max_tokens: int,
                        temperature: float = 0.0, label: str = "llm") -> dict:
        """One chat call -> parsed JSON dict. Retries per the pool contract.

        The parse-failure re-ask is CORRECTIVE: the invalid output is fed back with a fix
        instruction (an identical re-send at temperature 0 would just reproduce it)."""
        repair: list[dict] = []

        async def attempt():
            text = await self._stream_once(messages + repair, max_tokens, temperature)
            try:
                return _extract_json(text)
            except Exception as e:
                repair[:] = [
                    {"role": "assistant", "content": text[-1500:]},
                    {"role": "user", "content":
                     f"That output was invalid ({e}). Respond again with ONLY a single "
                     "valid JSON OBJECT ({...}) matching the requested schema."}]
                raise AttemptFailure(Outcome.PARSE_FAIL, str(e))
        return await self.run(attempt, label=label)

    async def _stream_once(self, messages: list[dict], max_tokens: int,
                           temperature: float) -> str:
        import openai
        try:
            stream = await self._client.chat.completions.create(
                model=self.cfg.llm_model, messages=messages, stream=True,
                temperature=temperature, max_tokens=max_tokens,
                response_format={"type": "json_object"},
                stream_options={"include_usage": True},
            )
        except openai.RateLimitError as e:
            raise AttemptFailure(Outcome.HTTP_429, str(e), retry_after=_retry_after(e))
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise AttemptFailure(Outcome.CONNECT, str(e))
        except openai.APIStatusError as e:
            out = Outcome.HTTP_5XX if e.status_code >= 500 else Outcome.FATAL
            raise AttemptFailure(out, f"HTTP {e.status_code}: {str(e)[:200]}")

        parts: list[str] = []
        finish = None
        usage = None
        first = True
        try:
            it = stream.__aiter__()
            while True:
                tmo = self.cfg.llm_ttft_timeout if first else self.cfg.llm_idle_timeout
                try:
                    chunk = await asyncio.wait_for(it.__anext__(), tmo)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    out = Outcome.STALL_TTFT if first else Outcome.STALL_IDLE
                    raise AttemptFailure(out, f"stalled after {len(parts)} chunks")
                except (openai.APIError, httpx.HTTPError) as e:
                    raise AttemptFailure(Outcome.CONNECT, f"mid-stream: {e}")
                first = False
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if chunk.choices:
                    ch = chunk.choices[0]
                    if ch.delta and ch.delta.content:
                        parts.append(ch.delta.content)
                    if ch.finish_reason:
                        finish = ch.finish_reason
        finally:
            try:
                await stream.close()
            except Exception:
                pass
        if usage is not None:
            self.metrics.prompt_tokens += usage.prompt_tokens or 0
            self.metrics.completion_tokens += usage.completion_tokens or 0
        if finish == "length":
            raise AttemptFailure(Outcome.LENGTH_CAP, f"max_tokens={max_tokens} hit")
        return "".join(parts)


def _retry_after(exc) -> float | None:
    try:
        v = exc.response.headers.get("retry-after")
        return float(v) if v else None
    except Exception:
        return None


# =============================================================================
# Search pools (Serper / Exa) — native async httpx, disk_cache-compatible with search.py
# =============================================================================

def _classify_status(status: int, body_head: str) -> AttemptFailure:
    if status == 429:
        return AttemptFailure(Outcome.HTTP_429, body_head)
    if status >= 500:
        return AttemptFailure(Outcome.HTTP_5XX, f"HTTP {status}: {body_head}")
    return AttemptFailure(Outcome.FATAL, f"HTTP {status}: {body_head}")


class _HttpSearchPool(ResourcePool):
    def __init__(self, name: str, size: int, timeout: float, cfg: OrchestrationConfig, *,
                 http: httpx.AsyncClient | None = None, **inj):
        super().__init__(name, size, cfg, **inj)
        self._http = http or httpx.AsyncClient(timeout=timeout)

    async def _post_json(self, url: str, payload: dict, headers: dict) -> dict:
        """One POST attempt -> parsed JSON. Classifies failures; records rate-limit headers."""
        try:
            r = await self._http.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            raise AttemptFailure(Outcome.CONNECT, f"{type(e).__name__}: {e}")
        rl = {k: v for k, v in r.headers.items()
              if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"}
        if rl:
            self.metrics.last_ratelimit = rl
        if r.status_code != 200:
            f = _classify_status(r.status_code, r.text[:200])
            if f.outcome is Outcome.HTTP_429:
                f.retry_after = _header_retry_after(r)
            raise f
        try:
            return r.json()
        except Exception as e:
            raise AttemptFailure(Outcome.CONNECT, f"bad JSON body: {e}")

    async def close(self):
        await self._http.aclose()


def _header_retry_after(r: httpx.Response) -> float | None:
    try:
        v = r.headers.get("retry-after")
        return float(v) if v else None
    except Exception:
        return None


class SerperPool(_HttpSearchPool):
    """Pooled Serper search. Owns ONLY concurrency, retry, breaker and metrics — the
    request shape, result parsing, blocklist drop, backfill rule and cache key all come
    from pipeline/search.py, which is the single source of truth for what a Serper call
    is. (Before 2026-07-28 this class re-implemented all of that and had drifted: it
    dropped PDFs, never applied SCRAPE_BLOCKLIST, and shared a cache key with the
    search.py path despite the different policy — so whichever path ran a query first
    silently decided the other's results.)"""

    def __init__(self, cfg: OrchestrationConfig, **kw):
        super().__init__("serper", cfg.serper_pool_size, cfg.serper_timeout, cfg, **kw)

    async def search_(self, query: str, top_k: int, *, date_ceiling: str | None = None,
                      exclude_domains: list[str] | None = None, min_results: int = 0,
                      label: str = "serper") -> list[dict]:
        xd = sorted(exclude_domains or [])
        key = search.cache_key("serper", query, top_k, date_ceiling, xd, min_results)
        cached = await asyncio.to_thread(disk_cache.get, "serper", key)
        if cached is not None:
            self.metrics.cache_hits += 1
            for r in cached:
                r.setdefault("provider", "serper")
            return cached

        headers = search.serper_headers()

        async def page(n: int) -> list[dict]:
            payload = search.serper_payload(query, top_k, date_ceiling, xd, page=n)
            data = await self.run(lambda: self._post_json(SERPER_ENDPOINT, payload, headers),
                                  label=label)
            return search.serper_finalize(search.serper_parse(data, top_k), xd)

        results = await page(1)
        if min_results and len(results) < min_results:
            results = search.serper_merge(results, await page(2))
        results = results[: max(top_k, min_results)]

        await asyncio.to_thread(disk_cache.set_, "serper", key, results)
        for r in results:
            r.setdefault("provider", "serper")
        return results


class EXA_NEVER_LIVE_TESTED:  # marker class: grep-friendly reminder
    """Exa has 1,000 free requests/month — the pool is sized from documentation only and is
    excluded from ALL smoke/ramp validation. Do not exercise it without an explicit go."""


class ExaPool(_HttpSearchPool):
    def __init__(self, cfg: OrchestrationConfig, **kw):
        super().__init__("exa", cfg.exa_pool_size, cfg.exa_timeout, cfg, **kw)

    async def search_(self, query: str, top_k: int, *, date_ceiling: str | None = None,
                      exclude_domains: list[str] | None = None,
                      label: str = "exa") -> list[dict]:
        # KNOWN, DELIBERATE divergence (2026-07-28): this key is built here rather than via
        # search.cache_key() because the two formats differ on the xd branch, and unifying
        # would orphan ~1.7k cached Exa entries — refetching them would blow the 1k/mo free
        # tier (house rule: never live-test Exa). Exa's result policy is unchanged, so the
        # entries are still valid. Unify the day Exa is on a paid plan or the cache is cold.
        xd = sorted(exclude_domains or [])
        extra = (["xd-" + hashlib.sha1(",".join(xd).encode()).hexdigest()[:8], "min0"]
                 if xd else [])
        key = disk_cache.make_key(query, str(top_k), date_ceiling or "",
                                  search._BLOCKLIST_TAG, *extra)
        cached = await asyncio.to_thread(disk_cache.get, "exa", key)
        if cached is not None:
            self.metrics.cache_hits += 1
            for r in cached:
                r.setdefault("provider", "exa")
            return cached

        payload = {"query": query, "type": "auto", "numResults": top_k,
                   "excludeDomains": search._SOCIAL_EXCLUDES + xd,
                   "contents": {"highlights": True, "text": {"maxCharacters": 12000}}}
        if date_ceiling:
            payload["endPublishedDate"] = date_ceiling
        headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}

        async def attempt():
            data = await self._post_json(EXA_ENDPOINT, payload, headers)
            out = []
            for r in data.get("results", []):
                link = r.get("url", "")
                if not link or link.lower().endswith(".pdf"):
                    continue
                highlights = [h.strip() for h in (r.get("highlights") or []) if h.strip()]
                text = (r.get("text") or "").strip()
                out.append({"url": link,
                            "snippet": (" … ".join(highlights) or text[:300]).strip(),
                            "date": r.get("publishedDate") or None,
                            "content": text or None})
                if len(out) >= top_k:
                    break
            return out

        results = await self.run(attempt, label=label)
        await asyncio.to_thread(disk_cache.set_, "exa", key, results)
        for r in results:
            r.setdefault("provider", "exa")
        return results


# =============================================================================
# Scrape pool — wraps sync search.scrape() in threads
# =============================================================================

class ScrapePool:
    """Global cap + per-domain politeness around pipeline/search.py scrape().

    None is a LEGITIMATE outcome (paywall/404/thin page — the pipeline falls back to the
    snippet), so there is no retry loop here; scrape() already owns its Jina fallback and
    disk cache. A scrape exceeding the wall timeout is abandoned (its thread finishes in
    the background; the pool slot is freed) and counted as a hang.
    """

    def __init__(self, cfg: OrchestrationConfig, *,
                 scrape_fn: Callable[[str], str | None] = search.scrape,
                 now: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.size = cfg.scrape_pool_size
        self._scrape_fn = scrape_fn
        self._now = now
        self._sem = asyncio.Semaphore(cfg.scrape_pool_size)
        self._domains: dict[str, asyncio.Semaphore] = {}
        self.metrics = PoolMetrics()

    async def scrape(self, url: str) -> str | None:
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        dsem = self._domains.setdefault(domain, asyncio.Semaphore(self.cfg.scrape_per_domain))
        # domain slot FIRST, then global — never hold a global slot while queueing on one host
        async with dsem, self._sem:
            self.metrics.in_flight += 1
            self.metrics.peak_in_flight = max(self.metrics.peak_in_flight,
                                              self.metrics.in_flight)
            t0 = self._now()
            try:
                text = await asyncio.wait_for(asyncio.to_thread(self._scrape_fn, url),
                                              self.cfg.scrape_wall_timeout)
            except asyncio.TimeoutError:
                self.metrics.retries[Outcome.STALL_IDLE] += 1
                log.warning("[scrape] wall timeout (%.0fs) on %s — abandoned",
                            self.cfg.scrape_wall_timeout, url)
                return None
            except Exception as e:
                self.metrics.fatal += 1
                log.warning("[scrape] unexpected error on %s: %s", url, e)
                return None
            finally:
                self.metrics.in_flight -= 1
            self.metrics.ok += 1
            self.metrics.latencies.append(self._now() - t0)
            if text is None:
                self.metrics.misses += 1
            return text


# =============================================================================
# Container
# =============================================================================

@dataclass
class Pools:
    cfg: OrchestrationConfig
    llm: LLMPool
    serper: SerperPool
    exa: ExaPool
    scrape: ScrapePool

    def status_line(self) -> str:
        segs = [self.llm.metrics.line("llm", self.llm.size),
                self.serper.metrics.line("serper", self.serper.size),
                self.scrape.metrics.line("scrape", self.scrape.size)]
        if self.exa.metrics.ok or self.exa.metrics.in_flight:
            segs.append(self.exa.metrics.line("exa", self.exa.size))
        return " | ".join(segs)

    async def close(self):
        try:
            await self.llm._client.close()
        except Exception:
            pass
        await self.serper.close()
        await self.exa.close()


def make_pools(cfg: OrchestrationConfig | None = None) -> Pools:
    cfg = cfg or OrchestrationConfig()
    return Pools(cfg=cfg, llm=LLMPool(cfg), serper=SerperPool(cfg),
                 exa=ExaPool(cfg), scrape=ScrapePool(cfg))
