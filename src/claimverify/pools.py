"""Pooled async clients (copied from pipeline/pools.py, Tavily/Brave/SearXNG paths out) with
full per-call tracing and an exact-match LLM response cache.

Each external resource has ONE pooled client owning its semaphore, timeout/retry policy,
circuit breaker and metrics: llm (DeepInfra, streaming), serper, exa, scrape. The retry
contract is unchanged: transient failures (connect, 5xx, 429 w/ Retry-After, TTFT/idle
stalls) retry forever with capped exponential backoff; a JSON parse failure gets one
corrective re-ask; finish_reason=length and non-429 4xx raise CallFatal.

Tracing: every chat_json / search_ / scrape call accepts `trace=` (claimverify.trace.Trace)
and appends a record — messages, raw output, parse status, finish reason, usage, latency,
attempt outcomes, cache hit (LLM); exact payload incl. q and tbs, raw JSON (search);
url, status, source, chars, reason (scrape).

LLM cache: key = sha256(model | temperature | max_tokens | json(messages)); a hit returns
the cached raw content with no network (cache_hit=True). Only successfully parsed outputs
are cached, under the ORIGINAL messages (the corrective re-ask never enters the key).
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from claimverify import disk_cache, retrieval
from claimverify.config import (
    EXA_API_KEY, EXA_ENDPOINT, PRICE_PER_M, SERPER_ENDPOINT, VERIFICATION_API_KEY,
    VERIFICATION_BASE_URL, OrchestrationConfig,
)

log = logging.getLogger("claimverify.pools")


# =============================================================================
# Outcomes + pure retry state machine (unchanged)
# =============================================================================

class Outcome(str, Enum):
    OK = "ok"
    CONNECT = "connect"
    HTTP_5XX = "5xx"
    HTTP_429 = "429"
    STALL_TTFT = "stall_ttft"
    STALL_IDLE = "stall_idle"
    PARSE_FAIL = "parse_fail"
    LENGTH_CAP = "length_cap"
    FATAL = "fatal"


TRANSIENT = {Outcome.CONNECT, Outcome.HTTP_5XX, Outcome.HTTP_429,
             Outcome.STALL_TTFT, Outcome.STALL_IDLE}


class AttemptFailure(Exception):
    def __init__(self, outcome: Outcome, detail: str = "", retry_after: float | None = None):
        super().__init__(f"{outcome.value}: {detail}")
        self.outcome = outcome
        self.detail = detail
        self.retry_after = retry_after


class CallFatal(Exception):
    def __init__(self, outcome: Outcome, detail: str, attempts: list):
        super().__init__(f"{outcome.value} after {len(attempts)} attempt(s): {detail}")
        self.outcome = outcome
        self.detail = detail
        self.attempts = attempts


@dataclass
class Decision:
    action: str
    sleep_s: float = 0.0


class RetryState:
    def __init__(self, cfg: OrchestrationConfig, rng: Callable[[], float] = random.random):
        self.cfg = cfg
        self.rng = rng
        self.attempts: list[tuple[Outcome, str]] = []
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
                return Decision("retry", 0.0)
            return Decision("raise")
        return Decision("raise")


class CircuitBreaker:
    def __init__(self, cfg: OrchestrationConfig, now: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self._now = now
        self._events: deque[bool] = deque(maxlen=cfg.breaker_window)
        self.state = "closed"
        self._opened_at = 0.0
        self._probe_out = False
        self.trips = 0

    def acquire_wait(self) -> float:
        if self.state == "closed":
            return 0.0
        remaining = self.cfg.breaker_cooldown - (self._now() - self._opened_at)
        if remaining > 0:
            return remaining
        if not self._probe_out:
            self._probe_out = True
            self.state = "half-open"
            return 0.0
        return 0.5

    def record(self, ok: bool) -> None:
        if self.state == "open":
            return
        if self.state == "half-open":
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


class PoolMetrics:
    def __init__(self):
        self.in_flight = 0
        self.peak_in_flight = 0
        self.ok = 0
        self.fatal = 0
        self.retries: Counter = Counter()
        self.latencies: deque[float] = deque(maxlen=500)
        self.cache_hits = 0
        self.misses = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.last_ratelimit: dict = {}

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


class ResourcePool:
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

    async def run(self, attempt_fn: Callable[[], Awaitable[Any]], *, label: str = "",
                  attempts_out: list | None = None) -> Any:
        async with self._sem:
            state = RetryState(self.cfg, rng=self._rng)
            if attempts_out is not None:
                attempts_out.clear()
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
                    if attempts_out is not None:
                        attempts_out.append({"outcome": f.outcome.value, "detail": f.detail[:200]})
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
                if attempts_out is not None:
                    attempts_out.append({"outcome": "ok"})
                return result


# =============================================================================
# LLM pool
# =============================================================================

def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        return obj
    if obj is not None:
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


def _usage_dict(usage) -> dict:
    """Normalise the SDK usage object: prompt / cached / completion tokens + DeepInfra's
    estimated_cost when present (falls back to PRICE_PER_M, flagged as `cost_source`)."""
    if usage is None:
        return {"prompt": 0, "cached": 0, "completion": 0, "cost_usd": 0.0, "cost_source": "none"}
    extra = getattr(usage, "model_extra", None) or {}
    details = getattr(usage, "prompt_tokens_details", None)
    cached = 0
    if details is not None:
        cached = getattr(details, "cached_tokens", None) or \
            (details.get("cached_tokens") if isinstance(details, dict) else 0) or 0
    p = getattr(usage, "prompt_tokens", 0) or 0
    c = getattr(usage, "completion_tokens", 0) or 0
    est = extra.get("estimated_cost") if isinstance(extra, dict) else None
    if est is None:
        est = getattr(usage, "estimated_cost", None)
    if est is not None:
        return {"prompt": p, "cached": cached, "completion": c, "cost_usd": float(est),
                "cost_source": "deepinfra.estimated_cost"}
    cost = ((p - cached) * PRICE_PER_M["prompt"] + cached * PRICE_PER_M["cached"]
            + c * PRICE_PER_M["completion"]) / 1e6
    return {"prompt": p, "cached": cached, "completion": c, "cost_usd": cost,
            "cost_source": "list-price-fallback"}


def llm_cache_key(model: str, temperature: float, max_tokens: int, messages: list[dict]) -> str:
    raw = "|".join([model, repr(float(temperature)), str(max_tokens),
                    json.dumps(messages, sort_keys=True, ensure_ascii=False)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMPool(ResourcePool):
    """Streaming chat against DeepInfra. json_object mode, non-thinking, temperature 0."""

    def __init__(self, cfg: OrchestrationConfig, *, client=None, use_cache: bool = True, **inj):
        super().__init__("llm", cfg.llm_pool_size, cfg, **inj)
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=VERIFICATION_BASE_URL, api_key=VERIFICATION_API_KEY,
                max_retries=0,
                timeout=httpx.Timeout(connect=cfg.llm_connect_timeout,
                                      read=cfg.llm_idle_timeout * 3, write=30.0, pool=60.0),
            )
        self._client = client
        self.use_cache = use_cache

    async def chat_json(self, messages: list[dict], *, max_tokens: int,
                        temperature: float = 0.0, label: str = "llm",
                        trace=None) -> dict:
        model = self.cfg.llm_model
        rec = {"label": label, "model": model,
               "params": {"temperature": temperature, "max_tokens": max_tokens,
                          "response_format": "json_object", "stream": True},
               "messages": messages, "raw_content": None, "parsed_ok": False,
               "finish_reason": None, "usage": None, "latency_s": None,
               "attempts": [], "cache_hit": False}
        key = llm_cache_key(model, temperature, max_tokens, messages)
        t0 = self._now()
        if self.use_cache:
            hit = await asyncio.to_thread(disk_cache.get, "llm", key)
            if hit is not None and isinstance(hit.get("content"), str):
                try:
                    obj = _extract_json(hit["content"])
                except Exception:
                    obj = None
                if obj is not None:
                    self.metrics.cache_hits += 1
                    rec.update(raw_content=hit["content"], parsed_ok=True,
                               finish_reason=hit.get("finish_reason"),
                               usage={**(hit.get("usage") or {}), "cost_usd": 0.0,
                                      "cost_source": "cache"},
                               latency_s=round(self._now() - t0, 3), cache_hit=True)
                    if trace is not None:
                        trace.llm_calls.append(rec)
                    return obj

        repair: list[dict] = []
        last: dict = {}

        async def attempt():
            text, finish, usage = await self._stream_once(messages + repair, max_tokens, temperature)
            last.update(text=text, finish=finish, usage=usage)
            try:
                return _extract_json(text)
            except Exception as e:
                repair[:] = [
                    {"role": "assistant", "content": text[-1500:]},
                    {"role": "user", "content":
                     f"That output was invalid ({e}). Respond again with ONLY a single "
                     "valid JSON OBJECT ({...}) matching the requested schema."}]
                raise AttemptFailure(Outcome.PARSE_FAIL, str(e))

        try:
            obj = await self.run(attempt, label=label, attempts_out=rec["attempts"])
        except CallFatal:
            rec.update(raw_content=last.get("text"), finish_reason=last.get("finish"),
                       usage=_usage_dict(last.get("usage")) if last.get("usage") else None,
                       latency_s=round(self._now() - t0, 3))
            if trace is not None:
                trace.llm_calls.append(rec)
            raise
        u = _usage_dict(last.get("usage"))
        self.metrics.prompt_tokens += u["prompt"]
        self.metrics.cached_tokens += u["cached"]
        self.metrics.completion_tokens += u["completion"]
        self.metrics.cost_usd += u["cost_usd"]
        rec.update(raw_content=last.get("text"), parsed_ok=True, finish_reason=last.get("finish"),
                   usage=u, latency_s=round(self._now() - t0, 3))
        if trace is not None:
            trace.llm_calls.append(rec)
        if self.use_cache:
            await asyncio.to_thread(disk_cache.set_, "llm", key, {
                "content": last.get("text"), "finish_reason": last.get("finish"),
                "usage": {k: u[k] for k in ("prompt", "cached", "completion")},
                "model": model, "label": label})
        return obj

    async def _stream_once(self, messages: list[dict], max_tokens: int,
                           temperature: float) -> tuple[str, str | None, Any]:
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
        if finish == "length":
            raise AttemptFailure(Outcome.LENGTH_CAP, f"max_tokens={max_tokens} hit")
        return "".join(parts), finish, usage


def _retry_after(exc) -> float | None:
    try:
        v = exc.response.headers.get("retry-after")
        return float(v) if v else None
    except Exception:
        return None


# =============================================================================
# Search pools
# =============================================================================

def _classify_status(status: int, body_head: str) -> AttemptFailure:
    if status == 429:
        return AttemptFailure(Outcome.HTTP_429, body_head)
    if status >= 500:
        return AttemptFailure(Outcome.HTTP_5XX, f"HTTP {status}: {body_head}")
    return AttemptFailure(Outcome.FATAL, f"HTTP {status}: {body_head}")


def _header_retry_after(r: httpx.Response) -> float | None:
    try:
        v = r.headers.get("retry-after")
        return float(v) if v else None
    except Exception:
        return None


class _HttpSearchPool(ResourcePool):
    def __init__(self, name: str, size: int, timeout: float, cfg: OrchestrationConfig, *,
                 http: httpx.AsyncClient | None = None, **inj):
        super().__init__(name, size, cfg, **inj)
        self._http = http or httpx.AsyncClient(timeout=timeout)

    async def _post_json(self, url: str, payload: dict, headers: dict) -> dict:
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


class SerperPool(_HttpSearchPool):
    """Pooled Serper search; request shape / parse / policy from claimverify.retrieval
    (identical to pipeline/search.py, so cache entries are shared). The raw JSON is
    cached in namespace `serper_raw` under the same key and returned in the trace."""

    def __init__(self, cfg: OrchestrationConfig, *, blocklist: bool = True, **kw):
        super().__init__("serper", cfg.serper_pool_size, cfg.serper_timeout, cfg, **kw)
        self.credits = 0
        self.blocklist = blocklist   # False: no -site: suffix, no client-side drop

    async def search_(self, query: str, top_k: int, *, date_ceiling: str | None = None,
                      exclude_domains: list[str] | None = None, min_results: int = 0,
                      label: str = "serper", trace=None) -> list[dict]:
        xd = sorted(exclude_domains or [])
        key = retrieval.cache_key("serper", query, top_k, date_ceiling, xd, min_results,
                                  blocklist=self.blocklist)
        payload = retrieval.serper_payload(query, top_k, date_ceiling, xd,
                                           blocklist=self.blocklist)
        rec = {"provider": "serper", "query": query, "payload": payload, "date_ceiling": date_ceiling,
               "exclude_domains": xd, "blocklist": self.blocklist, "raw_json": None, "n_results": 0, "latency_s": None,
               "cache_hit": False, "post_ceiling_hits": 0, "undated_hits": 0, "attempts": []}
        t0 = self._now()
        cached = await asyncio.to_thread(disk_cache.get, "serper", key)
        if cached is not None:
            self.metrics.cache_hits += 1
            for r in cached:
                r.setdefault("provider", "serper")
            post, undated = retrieval.post_ceiling_counts(cached, date_ceiling)
            rec.update(raw_json=await asyncio.to_thread(disk_cache.get, "serper_raw", key),
                       n_results=len(cached), latency_s=round(self._now() - t0, 3), cache_hit=True,
                       post_ceiling_hits=post, undated_hits=undated)
            if trace is not None:
                trace.searches.append(rec)
            return cached

        headers = retrieval.serper_headers()
        raw_pages = []

        async def page(n: int) -> list[dict]:
            pl = retrieval.serper_payload(query, top_k, date_ceiling, xd, page=n,
                                          blocklist=self.blocklist)
            data = await self.run(lambda: self._post_json(SERPER_ENDPOINT, pl, headers),
                                  label=label, attempts_out=rec["attempts"])
            self.credits += int((data.get("credits") or 1) if isinstance(data, dict) else 1)
            raw_pages.append(data)
            return retrieval.serper_finalize(retrieval.serper_parse(data, top_k), xd,
                                             blocklist=self.blocklist)

        results = await page(1)
        if min_results and len(results) < min_results:
            results = retrieval.serper_merge(results, await page(2))
        results = results[: max(top_k, min_results)]
        raw = raw_pages[0] if len(raw_pages) == 1 else {"pages": raw_pages}
        await asyncio.to_thread(disk_cache.set_, "serper", key, results)
        await asyncio.to_thread(disk_cache.set_, "serper_raw", key, raw)
        for r in results:
            r.setdefault("provider", "serper")
        post, undated = retrieval.post_ceiling_counts(results, date_ceiling)
        rec.update(raw_json=raw, n_results=len(results), latency_s=round(self._now() - t0, 3),
                   post_ceiling_hits=post, undated_hits=undated)
        if trace is not None:
            trace.searches.append(rec)
        return results


class ExaPool(_HttpSearchPool):
    """Exa neural search — 1k free requests/month; never exercised without an explicit go.
    Cache key format deliberately matches pipeline/pools.ExaPool so cached entries are reused."""

    def __init__(self, cfg: OrchestrationConfig, *, blocklist: bool = True, **kw):
        super().__init__("exa", cfg.exa_pool_size, cfg.exa_timeout, cfg, **kw)
        self.calls = 0
        self.blocklist = blocklist

    async def search_(self, query: str, top_k: int, *, date_ceiling: str | None = None,
                      exclude_domains: list[str] | None = None,
                      label: str = "exa", trace=None) -> list[dict]:
        xd = sorted(exclude_domains or [])
        extra = (["xd-" + hashlib.sha1(",".join(xd).encode()).hexdigest()[:8], "min0"]
                 if xd else [])
        key = disk_cache.make_key(query, str(top_k), date_ceiling or "",
                                  retrieval.blocklist_tag(self.blocklist), *extra)
        payload = {"query": query, "type": "auto", "numResults": top_k,
                   "excludeDomains": (retrieval._SOCIAL_EXCLUDES if self.blocklist else []) + xd,
                   "contents": {"highlights": True, "text": {"maxCharacters": 12000}}}
        if date_ceiling:
            payload["endPublishedDate"] = date_ceiling
        rec = {"provider": "exa", "query": query, "payload": payload, "date_ceiling": date_ceiling,
               "exclude_domains": xd, "raw_json": None, "n_results": 0, "latency_s": None,
               "cache_hit": False, "attempts": []}
        t0 = self._now()
        cached = await asyncio.to_thread(disk_cache.get, "exa", key)
        if cached is not None:
            self.metrics.cache_hits += 1
            for r in cached:
                r.setdefault("provider", "exa")
            rec.update(raw_json=await asyncio.to_thread(disk_cache.get, "exa_raw", key),
                       n_results=len(cached), latency_s=round(self._now() - t0, 3), cache_hit=True)
            if trace is not None:
                trace.searches.append(rec)
            return cached
        headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
        raw_box: dict = {}

        async def attempt():
            data = await self._post_json(EXA_ENDPOINT, payload, headers)
            raw_box["data"] = data
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

        results = await self.run(attempt, label=label, attempts_out=rec["attempts"])
        self.calls += 1
        await asyncio.to_thread(disk_cache.set_, "exa", key, results)
        await asyncio.to_thread(disk_cache.set_, "exa_raw", key, raw_box.get("data"))
        for r in results:
            r.setdefault("provider", "exa")
        rec.update(raw_json=raw_box.get("data"), n_results=len(results),
                   latency_s=round(self._now() - t0, 3))
        if trace is not None:
            trace.searches.append(rec)
        return results


# =============================================================================
# Scrape pool
# =============================================================================

class ScrapePool:
    def __init__(self, cfg: OrchestrationConfig, *,
                 scrape_fn: Callable[[str], dict] = retrieval.scrape_detail,
                 now: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.size = cfg.scrape_pool_size
        self._scrape_fn = scrape_fn
        self._now = now
        self._sem = asyncio.Semaphore(cfg.scrape_pool_size)
        self._domains: dict[str, asyncio.Semaphore] = {}
        self.metrics = PoolMetrics()

    async def scrape(self, url: str, trace=None) -> str | None:
        return (await self.scrape_detail(url, trace=trace))["text"]

    async def scrape_detail(self, url: str, trace=None) -> dict:
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        dsem = self._domains.setdefault(domain, asyncio.Semaphore(self.cfg.scrape_per_domain))
        rec = {"url": url, "status": None, "source": None, "chars": 0, "latency_s": None,
               "reason": None}
        async with dsem, self._sem:
            self.metrics.in_flight += 1
            self.metrics.peak_in_flight = max(self.metrics.peak_in_flight, self.metrics.in_flight)
            t0 = self._now()
            try:
                d = await asyncio.wait_for(asyncio.to_thread(self._scrape_fn, url),
                                           self.cfg.scrape_wall_timeout)
                if isinstance(d, str) or d is None:      # plain scrape_fn (tests)
                    d = {"text": d, "source": "fn", "reason": None if d else "miss"}
            except asyncio.TimeoutError:
                self.metrics.retries[Outcome.STALL_IDLE] += 1
                log.warning("[scrape] wall timeout (%.0fs) on %s — abandoned",
                            self.cfg.scrape_wall_timeout, url)
                d = {"text": None, "source": "timeout", "reason": "wall-timeout"}
            except Exception as e:
                self.metrics.fatal += 1
                log.warning("[scrape] unexpected error on %s: %s", url, e)
                d = {"text": None, "source": "error", "reason": f"{type(e).__name__}: {e}"[:200]}
            finally:
                self.metrics.in_flight -= 1
            self.metrics.ok += 1
            self.metrics.latencies.append(self._now() - t0)
            if d.get("text") is None:
                self.metrics.misses += 1
            rec.update(status=d.get("status"), source=d.get("source"),
                       chars=len(d.get("text") or ""), latency_s=round(self._now() - t0, 3),
                       reason=d.get("reason"))
            if trace is not None:
                trace.scrapes.append(rec)
            return d


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


def make_pools(cfg: OrchestrationConfig | None = None, *, llm_cache: bool = True,
               blocklist: bool = True) -> Pools:
    """`blocklist=False` (ClaimVerifyConfig.ugc_blocklist) drops the UGC blocklist from the
    query string, the client-side filter and the scrape gate alike."""
    cfg = cfg or OrchestrationConfig()
    scrape_fn = retrieval.scrape_detail if blocklist else \
        functools.partial(retrieval.scrape_detail, blocklist=False)
    return Pools(cfg=cfg, llm=LLMPool(cfg, use_cache=llm_cache),
                 serper=SerperPool(cfg, blocklist=blocklist),
                 exa=ExaPool(cfg, blocklist=blocklist),
                 scrape=ScrapePool(cfg, scrape_fn=scrape_fn))
