"""The runner: one gated, cached, retried, swept pass over a list of items.

This is THE way to drive DeepInfra from any project. Do not write another ThreadPool
loop. `run()` takes your items and a callable that builds one request body per item,
and handles everything around it: the RateGate, per-request retries with jittered
backoff, a per-item JSONL cache keyed by a stable hash, a progress line with ETA and
cost, per-minute observation lines, a hard spend cap, and an end-of-run sweep so no run
finishes with silently missing items.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import prices as price_mod
from .gate import THROTTLE_CODES, Observer, RateGate
from .pool import default_pool
from .transport import RequestsTransport, api_key, base_url

DEFAULT_WORKERS = 200

# Per-model worker override. Substring match, longest first. Big MoE models have a
# per-model THROUGHPUT budget rather than a request-rate one: 12 concurrent Qwen3-235B
# extraction calls on long prompts ran at a 148s median and timed out 3-4 of 12 at a
# 240s timeout, where 4 concurrent had none (policytrace, 2026-09-12).
MODEL_WORKERS = {
    "Qwen3-235B": 8,
    "GLM-5": 16,
    "Thinking": 16,
}


class PreflightAbort(RuntimeError):
    pass


def resolve_workers(model: str, explicit: int | None = None) -> int:
    """Worker count for this model: explicit wins, else the per-model override, else 200."""
    if explicit:
        return explicit
    for k in sorted(MODEL_WORKERS, key=len, reverse=True):
        if k in (model or ""):
            return MODEL_WORKERS[k]
    return DEFAULT_WORKERS


def cache_key(item_id, body: dict, model: str) -> str:
    """Stable hash of (item id, request body, model). The body is canonicalised so key
    order never changes the key; changing the prompt changes the key, as it must."""
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{item_id}|{model}|{blob}".encode()).hexdigest()[:32]


class Cache:
    """Append-only per-item JSONL cache: one {key, item_id, result} line per item.
    The last line for a key wins, so a rerun is free and a partial run resumes.
    Terminal failures are NOT cached — the sweep and the next run retry them."""

    def __init__(self, path):
        self.path = Path(path).expanduser() if path else None
        self.data: dict[str, object] = {}
        self.lock = threading.Lock()
        self.hits = 0
        if self.path and self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if "key" in rec:
                        self.data[rec["key"]] = rec.get("result")

    def get(self, key):
        with self.lock:
            v = self.data.get(key, _MISS)
            if v is not _MISS:
                self.hits += 1
            return v

    def put(self, key, item_id, result):
        with self.lock:
            self.data[key] = result
            if not self.path:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "item_id": item_id, "result": result},
                                   ensure_ascii=False, default=str) + "\n")


class _Miss:
    def __repr__(self):
        return "<miss>"


_MISS = _Miss()


@dataclass
class RunResult:
    results: dict = field(default_factory=dict)     # item_id -> parsed result
    errors: dict = field(default_factory=dict)      # item_id -> error string
    spent: float = 0.0
    elapsed: float = 0.0
    cached: int = 0
    cap_hit: bool = False
    sweeps: int = 0
    gate: RateGate | None = None

    @property
    def ok(self):
        return not self.errors

    def summary(self):
        return (f"{len(self.results)} ok, {len(self.errors)} failed, {self.cached} cached, "
                f"${self.spent:.4f} in {self.elapsed:.0f}s"
                + (f"  [{self.gate.summary()}]" if self.gate else "")
                + ("  SPEND CAP HIT" if self.cap_hit else ""))


def run(items, make_request, *, model, cache_path=None, workers=None,
        gate=None, gate_target=160, gate_ceiling=200, gate_floor=32,
        pool=None, obs_path=None, obs_every=60.0, max_spend=None, attempts=6, timeout=300,
        sweeps=2, parse=None, item_id=None, transport=None, prices=None,
        preflight=False, preflight_n=200, base=None, key=None, endpoint=None,
        log=print, sleep=time.sleep, progress_every=50) -> RunResult:
    """Run `make_request(item)` -> request body over every item, through the RateGate.

    items         any sequence; each item is passed to make_request and item_id
    make_request  item -> dict request body (chat/completions or embeddings)
    model         model id; injected into the body when absent; picks the worker default
    cache_path    JSONL cache file (per-item, keyed by item id + body + model)
    workers       thread pool size; None = per-model default (200, 8 for Qwen3-235B)
    pool          machine-wide slot pool shared with every other process using this
                  runner; None = the process default (ceiling 190, $DEEPINFRA_POOL=0 off)
    max_spend     hard cap in $: once reached, no further calls are made
    parse         (response_json, item) -> stored result; default = the raw response json.
                  NOTE the cache stores the PARSED result and the key does not cover `parse`,
                  so a changed parse wants a new cache_path.
    transport     object with .post(url, headers, body, timeout); default requests
    """
    items = list(items)
    item_id = item_id or _default_item_id
    parse = parse or (lambda j, item: j)
    workers = resolve_workers(model, workers)
    gate = gate or RateGate(target=min(gate_target, workers), ceiling=gate_ceiling,
                            floor=min(gate_floor, workers), log=log)
    pool = pool if pool is not None else default_pool()
    obs = Observer(gate, obs_path, log=log, pool=pool)
    transport = transport or RequestsTransport(pool=max(64, workers + 16))
    url_base = base_url(base)
    auth = {"Authorization": f"Bearer {api_key(key)}"}
    cache = Cache(cache_path)

    res = RunResult(gate=gate)
    lock = threading.Lock()
    t_start = time.time()
    state = {"done": 0, "failed": 0, "t0": time.time(), "last_log": 0.0,
             "last_obs": 0.0, "last_obs_done": 0, "total": len(items)}
    state["last_log"] = state["last_obs"] = state["t0"]

    def gated_post(body, ep):
        """POST through the gate, retrying 429 'model busy' / 5xx / transport failure on the
        SAME request with jittered exponential backoff (1,2,4,8,16, capped at 30s) before
        giving up. Raises after `attempts` so the caller marks the item failed for the sweep."""
        # Every ATTEMPT is one gate event, so the windowed rate is the per-request 429 rate.
        # (Recording once per item with a sticky flag over-counted: a 2 % per-request rate read
        # as 1-(1-p)^k ~ 5-10 % and breached the threshold -- observed 2026-09-23.)
        last = None
        for attempt in range(attempts):
            with pool.slot(gate):          # one machine-wide slot + one gate seat
                try:
                    r = transport.post(f"{url_base}/{ep}", auth, body, timeout)
                except Exception as e:  # noqa: BLE001  reset/timeout are pressure signals too
                    last, r = e, None
            if r is not None and r.status_code not in THROTTLE_CODES:
                r.raise_for_status()
                gate.record(False)
                return r
            gate.record(r is not None and r.status_code == 429)
            if r is not None:
                last = Exception(f"http {r.status_code}: {str(getattr(r, 'text', ''))[:120]}")
            if attempt < attempts - 1:
                sleep(min(30.0, 2.0 ** attempt) + random.random())
        raise last if last is not None else RuntimeError("request failed")

    def work(item):
        iid = item_id(item)
        body = dict(make_request(item))
        body.setdefault("model", model)
        ep = endpoint or ("embeddings" if "input" in body else "chat/completions")
        ck = cache_key(iid, body, body["model"])
        hit = cache.get(ck)
        if hit is not _MISS:
            with lock:
                res.results[iid] = hit
                res.errors.pop(iid, None)
                res.cached += 1
                _tick(iid, 0.0, None)
            return
        with lock:
            if max_spend is not None and res.spent >= max_spend:
                res.cap_hit = True
                res.errors[iid] = "spend-cap"
                state["done"] += 1
                return
        t0 = time.time()
        try:
            r = gated_post(body, ep)
            j = r.json()
        except Exception as e:  # noqa: BLE001  exhausted its retries
            with lock:
                res.errors[iid] = repr(e)
                state["failed"] += 1
                _tick(iid, 0.0, None)
            return
        cost = price_mod.cost_of(body["model"], j.get("usage"), prices)
        try:
            out = parse(j, item)
        except Exception as e:  # noqa: BLE001  a malformed body is a failed item, not a crash
            with lock:
                res.spent += cost
                res.errors[iid] = f"parse: {e!r}"
                state["failed"] += 1
                _tick(iid, cost, time.time() - t0)
            return
        cache.put(ck, iid, out)
        with lock:
            res.results[iid] = out
            res.errors.pop(iid, None)
            res.spent += cost
            _tick(iid, cost, time.time() - t0)

    def _tick(iid, cost, latency):
        """Caller holds `lock`. Progress line + per-minute observation."""
        state["done"] += 1
        done, total = state["done"], state["total"]
        now = time.time()
        if latency is not None:
            obs.latency(now, latency)
        if done % progress_every == 0 or done == total or now - state["last_log"] >= 60:
            state["last_log"] = now
            el = max(1e-9, now - state["t0"])
            eta = el / done * (total - done) / 60 if done else 0.0
            log(f"  {done}/{total} ({done / max(1, total):.0%})  ${res.spent:.4f}  {el:.0f}s  "
                f"{done / el:.1f}/s  ETA {eta:.0f}m  {gate.line()}"
                + (f"  {pool.line()}" if pool.on else "")
                + (f"  failed {state['failed']}" if state["failed"] else ""), flush=True)
        if now - state["last_obs"] >= obs_every or done == total:
            obs.emit(model, done, done - state["last_obs_done"], now - state["last_obs"],
                     state["failed"])
            state["last_obs"], state["last_obs_done"] = now, done

    def pump(batch, size=None):
        with ThreadPoolExecutor(size or workers) as ex:
            for f in as_completed([ex.submit(work, it) for it in batch]):
                f.result()

    # PRE-FLIGHT: hold 96 in-flight for up to 60s on the first N items and check the busy rate
    # before committing the run. The items are cached, so the main run reuses them.
    if preflight and items:
        pf = items[:preflight_n]
        saved = gate.target
        gate.target = float(min(96, workers))
        gate.events.clear()
        gate.last_breach = 0.0
        log(f"  [preflight] {len(pf)} items at {int(gate.target)} in-flight for up to 60s ...", flush=True)
        t_pf = time.time()
        with ThreadPoolExecutor(min(workers, 96)) as ex:
            futs = [ex.submit(work, it) for it in pf]
            for f in as_completed(futs):
                f.result()
                if time.time() - t_pf >= 60:
                    break
        n, _n429, rate = gate.window_stats()
        p50, p95 = obs.percentiles()
        log(f"  [preflight] 429 rate {rate:.1%} over {n} reqs  p50 {p50:.2f}s  p95 {p95:.2f}s", flush=True)
        if rate > 0.20:
            raise PreflightAbort(f"429 rate {rate:.1%} over 20% -- pool too busy")
        gate.target = saved
        gate.events.clear()
        gate.last_adjust = time.time()
        gate.last_breach = 0.0
        state["done"] = 0
        state["last_obs_done"] = 0
        state["t0"] = state["last_log"] = state["last_obs"] = time.time()

    pump(items)

    # FINAL SWEEP: no run ends with silently missing items. Re-run every failed item (they were
    # never cached), up to `sweeps` passes. Items skipped by the spend cap are left alone.
    for s in range(1, sweeps + 1):
        failed = [it for it in items
                  if item_id(it) in res.errors and res.errors[item_id(it)] != "spend-cap"]
        if not failed:
            break
        res.sweeps = s
        log(f"  [sweep {s}] {len(failed)} failed items -> re-run", flush=True)
        state["total"] = len(failed)
        state["done"] = 0
        state["failed"] = 0
        state["t0"] = state["last_log"] = time.time()
        pump(failed)
    still = [i for i, e in res.errors.items() if e != "spend-cap"]
    if still:
        log(f"  [sweep] {len(still)} items still failing after {sweeps} sweeps", flush=True)

    res.elapsed = time.time() - t_start
    res.cached = cache.hits
    log(f"done: {res.summary()}", flush=True)
    return res


def _default_item_id(item):
    if isinstance(item, dict):
        for k in ("id", "item_id", "uid", "key", "claim_id"):
            if k in item:
                return item[k]
        return hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return str(item)


def preflight_probe(model, n=96, base=None, key=None, transport=None, log=print,
                    obs_path=None, max_tokens=1, sleep=time.sleep):
    """Standalone preflight: n tiny chat calls at up to 96 in-flight, no items needed.
    Costs a fraction of a cent and proves the key, the base URL and the pool's health."""
    items = [{"id": f"pf{i}", "text": "ping"} for i in range(n)]
    return run(items,
               lambda it: {"messages": [{"role": "user", "content": it["text"]}],
                           "max_tokens": max_tokens, "temperature": 0},
               model=model, workers=min(96, n), gate_target=min(96, n), gate_ceiling=96,
               gate_floor=8, base=base, key=key, transport=transport, log=log,
               obs_path=obs_path, attempts=3, timeout=60, sweeps=1, sleep=sleep,
               progress_every=max(1, n // 2))
