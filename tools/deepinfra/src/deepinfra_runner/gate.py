"""RateGate: the concurrency controller for DeepInfra work.

Lifted verbatim (same defaults, same arithmetic) from
factchecking_with_LLMs/src/eval/scripts/build_eval/reader_lab.py, where it was
settled on 2026-09-14. The only changes are cosmetic: the print goes through an
injectable `log` so a library caller can silence it, and the observation writer
lives here instead of in module globals.

Why a rate controller and not a per-429 cut: DeepInfra documents a 200-concurrent
per-model account limit AND, separately, transient "model busy" 429s that fire
"even if you're under the limit" and clear on retry, with no rate-limit headers.
A gate that cuts on a single 429 unwinds the whole run to its floor during a
momentary busy-storm. This one reacts only to the RATE of 429s over a sliding
60s window, and retries every individual 429 on the same request instead.
"""
from __future__ import annotations

import collections
import json
import os
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Status codes treated as back-pressure rather than as errors.
THROTTLE_CODES = {429, 500, 502, 503, 529}

DEFAULT_OBS_PATH = Path("~/.cache/deepinfra_runner/gate_observations.jsonl").expanduser()


def default_obs_path():
    """Where observations go when the caller names no path: $DEEPINFRA_RUNNER_OBS, else
    ~/.cache/deepinfra_runner/gate_observations.jsonl."""
    return Path(os.environ.get("DEEPINFRA_RUNNER_OBS") or DEFAULT_OBS_PATH).expanduser()


class RateGate:
    """Concurrency limiter for a busy DeepInfra pool.

      - holds a `target` in-flight and only ever reacts to the RATE of 429s over a sliding
        window, never to a single 429 (per-request 429/5xx/transport are retried by the runner);
      - cuts the target by 20% (to a floor) when the windowed 429 rate exceeds a threshold;
      - recovers +10% every recover_s of clean running, back up to the ceiling.
    The clock is injectable so the controller is unit-testable offline with a fake clock."""

    def __init__(self, target=160, ceiling=200, floor=32, window_s=60.0,
                 breach_rate=0.05, cut_factor=0.8, recover_factor=0.10,
                 recover_s=30.0, min_sample=20, clock=time.time, log=print):
        self.target = float(target)
        self.ceiling = float(ceiling)
        self.floor = float(floor)
        self.window_s = window_s
        self.breach_rate = breach_rate
        self.cut_factor = cut_factor
        self.recover_factor = recover_factor
        self.recover_s = recover_s
        self.min_sample = min_sample
        self._clock = clock
        self._log = log or (lambda *a, **k: None)
        self.inflight = 0
        self.peak = 0
        self.n_cuts = 0
        self.n_ups = 0
        self.n_429_total = 0
        self.completed = 0
        self.events = collections.deque()   # (t, saw_429) per completed request
        now = clock()
        self.last_adjust = now
        self.last_breach = 0.0
        self.cv = threading.Condition()

    def acquire(self):
        """Take one in-flight seat, blocking until the target allows it."""
        with self.cv:
            while self.inflight >= int(self.target):
                self.cv.wait(0.5)
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)

    def try_acquire(self):
        """Non-blocking acquire. The machine-wide pool takes its slot FIRST and then asks
        the gate with this: if the gate is full it hands the slot straight back rather than
        sit on it, so a gate that has cut itself cannot hoard the machine's slots."""
        with self.cv:
            if self.inflight >= int(self.target):
                return False
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            return True

    def release(self):
        with self.cv:
            self.inflight -= 1
            self.cv.notify()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def _trim(self, now):
        w = self.events
        while w and now - w[0][0] > self.window_s:
            w.popleft()

    def record(self, saw_429):
        """One completed ATTEMPT (a response, a 429, or a transport failure). saw_429: True
        if this attempt was a 429 'model busy'. Adjusts the target on the WINDOW."""
        with self.cv:
            now = self._clock()
            self.completed += 1
            if saw_429:
                self.n_429_total += 1
            self.events.append((now, bool(saw_429)))
            self._trim(now)
            n = len(self.events)
            n429 = sum(1 for _, f in self.events if f)
            rate = n429 / n if n else 0.0
            # cut only on a windowed breach with enough sample, so a single 429 never cuts,
            # and at most one cut per half-window: after a cut the window is cleared, so
            # without this guard 20 quick completions could cut again within a second and
            # cascade 160 -> 32 in a couple of seconds (observed 2026-09-23).
            if (n >= self.min_sample and rate > self.breach_rate and self.target > self.floor
                    and now - self.last_breach >= self.window_s / 2):
                self.target = max(self.floor, self.target * self.cut_factor)
                self.n_cuts += 1
                self.last_breach = now
                self.last_adjust = now
                self.events.clear()          # fresh window after acting
                self._log(f"  [gate] 429 rate {rate:.1%} over {n} reqs -> target {int(self.target)}", flush=True)
            # recovery needs the same evidence bar as a cut: a thin window (n < min_sample)
            # with one 429 in it is not a breach and must not pin the target at the floor.
            elif (self.target < self.ceiling
                  and now - self.last_adjust >= self.recover_s
                  and now - self.last_breach >= self.recover_s
                  and (n < self.min_sample or rate <= self.breach_rate)):
                self.target = min(self.ceiling, self.target * (1 + self.recover_factor))
                self.n_ups += 1
                self.last_adjust = now
                self.cv.notify_all()

    def window_stats(self, now=None):
        with self.cv:
            now = self._clock() if now is None else now
            self._trim(now)
            n = len(self.events)
            n429 = sum(1 for _, f in self.events if f)
            return n, n429, (n429 / n if n else 0.0)

    def summary(self):
        return (f"gate target {int(self.target)} (peak {self.peak}, {self.n_ups} ups, "
                f"{self.n_cuts} cuts, {self.n_429_total} req saw 429)")

    def line(self):
        return f"gate {int(self.target)} (inflight {self.inflight}, peak {self.peak}, {self.n_cuts} cuts)"


class Observer:
    """Per-minute observation lines, appended to a shared jsonl so every run doubles as a
    concurrency probe. Same record schema as reader_lab's gate_observations.jsonl:
    ts, model, target, inflight, completed, reads_s, n429_window, rate_429, p50_latency,
    p95_latency, failed, cuts -- plus pool_slots_held / pool_waiters (this process) and
    pool_total_held (every process on the machine), so a log shows when a slow run is
    machine contention rather than the provider."""

    def __init__(self, gate, path=None, log=print, pool=None):
        self.gate = gate
        self.pool = pool
        self.path = Path(path).expanduser() if path else default_obs_path()
        self._log = log or (lambda *a, **k: None)
        self.lat = collections.deque()      # (t, latency_s), trimmed to 60s
        self.lock = threading.Lock()

    def latency(self, now, seconds):
        with self.lock:
            self.lat.append((now, seconds))
            while self.lat and now - self.lat[0][0] > 60.0:
                self.lat.popleft()

    def percentiles(self, now=None):
        now = now if now is not None else time.time()
        with self.lock:
            while self.lat and now - self.lat[0][0] > 60.0:
                self.lat.popleft()
            lats = sorted(x for _, x in self.lat)
        if not lats:
            return 0.0, 0.0
        return statistics.median(lats), lats[min(len(lats) - 1, int(0.95 * len(lats)))]

    def emit(self, model, done, interval_reads, interval_s, failed, extra=""):
        now = time.time()
        p50, p95 = self.percentiles(now)
        n_win, n429, rate = self.gate.window_stats(now)
        rps = interval_reads / interval_s if interval_s > 0 else 0.0
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model,
               "target": int(self.gate.target), "inflight": self.gate.inflight, "completed": done,
               "reads_s": round(rps, 2), "n429_window": n429, "rate_429": round(rate, 4),
               "p50_latency": round(p50, 2), "p95_latency": round(p95, 2), "failed": failed,
               "cuts": self.gate.n_cuts}
        if self.pool is not None:
            try:
                rec.update(pool_slots_held=self.pool.held, pool_waiters=self.pool.waiters,
                           pool_total_held=self.pool.total_held())
            except Exception:  # noqa: BLE001  observability must never kill a run
                pass
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001  observability must never kill a run
            pass
        self._log(f"  [obs] {json.dumps(rec)}{extra}", flush=True)
        return rec
