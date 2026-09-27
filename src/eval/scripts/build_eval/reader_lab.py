"""Reader lab: re-READ the documents already retrieved by an urn run under a
different read prompt or model, with no search spend, and compare the flags.

Branch reader-iteration (Daniel 2026-09-09): the outlet and timeline audits found
the reader emitting "1" on documents that state the claim (31% of flagged outlet
posts, 5% of flagged timeline posts). Every document region a run read is stored in
its results jsonl (`sent_ids`, `sents`), so a prompt can be re-tried on exactly the
same input. This script does that and reports the flag transitions and the score
movement against the hand-audit tags.

    uv run python -m eval.scripts.build_eval.reader_lab \
        --run eval/data/urn_runs/e2_tweets/results-00.jsonl \
        --claims <claims.json> --prompt v5 --out <dir> [--model M] [--workers 24] [--limit N]

claims.json: [{"claim_id": ..., "group": "confirming" | "clean" | "control" | ..., "post_id": ...}, ...]
Outputs <out>/<prompt>__<model_tag>.jsonl (one record per claim, old and new flags and
scores) and prints the summary. Cached per (prompt hash, model, claim, rank) in
<out>/cache/ so a rerun is free.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval import fit_urn, graded_urn  # noqa: E402
from eval.scripts.build_eval.reader_lab_prompts import CANON_SYS, MAPPERS, PROMPTS  # noqa: E402

PAD = 10
CUT_R1, CUT_R2 = -4.079, -2.075
import os  # noqa: E402
PROVIDERS = {"deepinfra": (None, None),   # eur defaults
             "groq": ("https://api.groq.com/openai/v1", os.environ.get("GROQ_API_KEY"))}
PROVIDER = ["deepinfra"]


def _endpoint():
    base, key = PROVIDERS[PROVIDER[0]]
    return (base or eur.EXTRACTION_BASE_URL), (key or eur.EXTRACTION_API_KEY)


WEIGHTS_CACHE = Path("eval/data/reader_lab/graded_weights.json")


def weights():
    """Per-flag weights for the old/new score columns. MEMOISED: fitting them parses the
    whole 263 MB e1_ctx run, and doing that while the script is already holding a run file
    was half of its peak memory (the 2026-09-10 fc_gold leg died there). Delete the file to
    refit."""
    if WEIGHTS_CACHE.exists():
        return json.load(open(WEIGHTS_CACHE))
    pop = fit_urn.load_population(Path("eval/data/populations/fc_gold.parquet"))
    rows = fit_urn.load_headline(Path("eval/data/urn_runs/e1_ctx/results-00.jsonl"), None, population=pop)
    w = graded_urn.fit_graded(rows)
    WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(w, open(WEIGHTS_CACHE, "w"))
    return w


def score(flags, w):
    fl = flags[:PAD] + ["I"] * max(0, PAD - len(flags))
    return sum(w.get(f, 0.0) for f in fl)


def _parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        i, k = text.find("{"), text.rfind("}")
        return json.loads(text[i:k + 1]) if 0 <= i < k else {}


class AdaptiveGate:
    """Concurrency cap for the model calls that hunts the provider's ceiling (Daniel
    2026-09-10). Additive increase, multiplicative decrease: after `up_after` clean
    completions the cap grows by `step`, a throttle (429 / 5xx / "busy") cuts it by 30%
    and holds it for `hold_s`. Throughput is inflight / latency, so the cap sits as close
    to the ceiling as the provider allows without a 429 storm, whatever the worker count."""

    def __init__(self, start=32, cap=96, floor=4, step=4, up_after=40, hold_s=30.0):
        self.limit, self.cap, self.floor, self.step, self.up_after, self.hold_s = start, cap, floor, step, up_after, hold_s
        self.inflight = 0; self.ok = 0; self.last_cut = 0.0; self.n_cuts = 0; self.n_ups = 0; self.peak = start
        self.cv = threading.Condition()

    def __enter__(self):
        with self.cv:
            while self.inflight >= self.limit:
                self.cv.wait(1.0)
            self.inflight += 1
        return self

    def __exit__(self, *exc):
        with self.cv:
            self.inflight -= 1
            self.cv.notify()

    def success(self):
        with self.cv:
            self.ok += 1
            if self.ok >= self.up_after and self.limit < self.cap and time.time() - self.last_cut > self.hold_s:
                self.limit = min(self.cap, self.limit + self.step); self.ok = 0; self.n_ups += 1
                self.peak = max(self.peak, self.limit)
                print(f"  [gate] up -> {self.limit}", flush=True)
                self.cv.notify_all()

    def throttle(self, why=""):
        with self.cv:
            self.ok = 0
            if time.time() - self.last_cut > 2.0:      # one cut per burst of 429s
                self.limit = max(self.floor, int(self.limit * 0.7)); self.last_cut = time.time(); self.n_cuts += 1
                print(f"  [gate] throttled ({why}), limit -> {self.limit}", flush=True)

    def summary(self):
        return f"gate limit {self.limit} (peak {self.peak}, {self.n_ups} ups, {self.n_cuts} cuts)"


class RateGate:
    """Concurrency limiter for a busy DeepInfra pool (Daniel 2026-09-14, clog/140926.md).

    DeepInfra documents a 200-concurrent-per-model account limit AND, separately, transient
    "model busy" 429s "even if you're under the limit" that clear on retry, with no rate-limit
    headers. The old AdaptiveGate cut 30% on ANY single 429 and recovered slowly, so a momentary
    busy-storm unwound the whole run to the floor. This gate instead:
      - holds a `target` in-flight and only ever reacts to the RATE of 429s over a sliding
        window, never to a single 429 (per-request 429/5xx/transport are retried in gated_post);
      - cuts the target by 20% (to a floor) when the windowed 429 rate exceeds a threshold;
      - recovers +10% every recover_s of clean running, back up to the ceiling.
    The clock is injectable so the controller is unit-testable offline with a fake clock."""

    def __init__(self, target=160, ceiling=190, floor=32, window_s=60.0,
                 breach_rate=0.05, cut_factor=0.8, recover_factor=0.10,
                 recover_s=30.0, min_sample=20, clock=time.time):
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

    def __enter__(self):
        with self.cv:
            while self.inflight >= int(self.target):
                self.cv.wait(0.5)
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        return self

    def __exit__(self, *exc):
        with self.cv:
            self.inflight -= 1
            self.cv.notify()

    def _trim(self, now):
        w = self.events
        while w and now - w[0][0] > self.window_s:
            w.popleft()

    def record(self, saw_429):
        """One completed request (success or terminal failure). saw_429: True if this request
        hit at least one 429 'model busy' while completing. Adjusts the target on the WINDOW."""
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
            # cut only on a windowed breach with enough sample, so a single 429 never cuts
            if n >= self.min_sample and rate > self.breach_rate and self.target > self.floor:
                self.target = max(self.floor, self.target * self.cut_factor)
                self.n_cuts += 1
                self.last_breach = now
                self.last_adjust = now
                self.events.clear()          # fresh window after acting
                print(f"  [gate] 429 rate {rate:.1%} over {n} reqs -> target {int(self.target)}", flush=True)
            elif (self.target < self.ceiling
                  and now - self.last_adjust >= self.recover_s
                  and now - self.last_breach >= self.recover_s
                  and n >= self.min_sample          # a cleared window is not a clean one (2026-09-18)
                  and rate <= self.breach_rate):
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


GATE = AdaptiveGate()
OBS_PATH = Path("eval/data/reader_lab/gate_observations.jsonl")
_LAT = collections.deque()          # (t, latency_s) for successful reads, trimmed to 60s
THROTTLE_CODES = {429, 500, 502, 503, 529}
# DeepInfra serves gpt-oss json mode from a separate pool that answered 429 "Model busy" to
# every request on 2026-09-10 while plain requests passed; False = parse JSON out of content.
OSS_JSON_MODE = [True]


def gated_post(url, headers, body, timeout, attempts=None):
    """POST through GATE, retrying a 429 'model busy' / 5xx / transport failure on the SAME
    request with jittered exponential backoff (1, 2, 4, 8, ... capped at 30s) before giving up.
    RateGate: reacts only to the windowed 429 RATE (record()), never to a single 429; legacy
    AdaptiveGate: keeps its per-429 multiplicative cut (success()/throttle()). Raises after
    `attempts` so the caller can mark the read failed for the end-of-run sweep."""
    legacy = isinstance(GATE, AdaptiveGate)
    attempts = attempts or (5 if legacy else 6)
    saw_429 = False
    last = None
    for attempt in range(attempts):
        with GATE:
            try:
                r = eur._SESSION.post(url, headers=headers, json=body, timeout=timeout)
            except Exception as e:  # noqa: BLE001  (connection reset, timeout: pressure signals too)
                last = e; r = None
        if r is not None and r.status_code not in THROTTLE_CODES:
            r.raise_for_status()
            if legacy:
                GATE.success()
            else:
                GATE.record(saw_429)
            return r
        if r is not None and r.status_code == 429:
            saw_429 = True
        if legacy:
            GATE.throttle(f"http {r.status_code} {body.get('model')}{' json' if 'response_format' in body else ''} {r.text[:40]!r}" if r is not None else type(last).__name__)
        if r is not None:
            last = Exception(f"http {r.status_code}: {r.text[:120]}")
        if attempt < attempts - 1:
            time.sleep(min(30.0, 2.0 ** attempt) + random.random())
    if not legacy:
        GATE.record(saw_429)
    raise last


def emit_observation(model, done, spent, interval_reads, interval_s, failed, extra=""):
    """One observation line to the run log AND appended to the shared gate_observations.jsonl,
    so every run doubles as a concurrency probe: target, in-flight, completed, reads/s, 429
    count+rate over the window, p50/p95 latency, failed count."""
    now = time.time()
    while _LAT and now - _LAT[0][0] > 60.0:
        _LAT.popleft()
    lats = sorted(l for _, l in _LAT)
    p50 = statistics.median(lats) if lats else 0.0
    p95 = lats[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else 0.0
    if isinstance(GATE, RateGate):
        n_win, n429, rate = GATE.window_stats(now)
        target = int(GATE.target)
    else:
        n_win, n429, rate = 0, 0, 0.0
        target = GATE.limit
    rps = interval_reads / interval_s if interval_s > 0 else 0.0
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model,
           "target": target, "inflight": GATE.inflight, "completed": done,
           "reads_s": round(rps, 2), "n429_window": n429, "rate_429": round(rate, 4),
           "p50_latency": round(p50, 2), "p95_latency": round(p95, 2), "failed": failed,
           "cuts": GATE.n_cuts}
    try:
        OBS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OBS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001  observability must never kill a run
        pass
    print(f"  [obs] {json.dumps(rec)}{extra}", flush=True)


# Completion budget for a thinking read. DeepInfra admits a gpt-oss request against the
# max_tokens it RESERVES, not the tokens it returns: measured 2026-09-10 on real read
# prompts, 8000 gives 9 of 24 concurrent requests a 429 "Model busy" where 1000 gives
# 0 of 24 at the same moment. Reads never come close — 187 smoke reads had median 119
# completion tokens and a maximum of 382 — so the cap costs nothing and buys concurrency.
THINK_MAX_TOKENS = [1000]
READ_TIMEOUT = [300]   # per-request read timeout (s); raised via --read-timeout


def read_one(sys_prompt, claim_block, ids, sents, model, reasoning=None, mapper=None):
    """read_doc's logic on a stored region, with the prompt, model and reasoning chosen here.
    reasoning: None = the model's default (no thinking for instruct models; gpt-oss runs at
    "low"); "off" strips thinking where the model allows it; "low"/"medium"/"high" set
    reasoning_effort. Thinking models are called without forced-JSON mode (it suppresses
    their reasoning, extract_tweet_claims.py) and the JSON is parsed out of the content."""
    doc = "\n".join(f"[{i}] {s}" for i, s in zip(ids, sents))
    body_model = model or eur.VERIFICATION_MODEL
    # DeepInfra's Kimi models all emit reasoning_content (Kimi-K2.5 / K2.6 are tagged
    # "reasoning"; even Kimi-K2-Instruct-0905 was observed thinking, clog/140926.md). Treated
    # as thinking so max_tokens is raised and forced-JSON stays off (it suppresses reasoning),
    # otherwise they burn the 400-token instruct budget on reasoning and return empty reads.
    thinking_id = ("Thinking" in body_model or "gpt-oss" in body_model
                   or "Kimi" in body_model or "moonshot" in body_model)
    eff = reasoning
    if "gpt-oss" in body_model and eff in (None, "off"):
        eff = "low"
    think = thinking_id or eff in ("low", "medium", "high")
    body = {"model": body_model, "temperature": 0,
            "max_tokens": THINK_MAX_TOKENS[0] if think else 1000,   # 400 truncated 15 V4-Pro reads (2026-09-18)
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": f"{claim_block}\n\nDOCUMENT:\n{doc}"}]}
    if eff in ("low", "medium", "high"):
        body["reasoning_effort"] = eff
    if (not think or "gpt-oss" in body_model) and not ("gpt-oss" in body_model and not OSS_JSON_MODE[0]):
        body["response_format"] = {"type": "json_object"}   # gpt-oss keeps json mode fine at max_tokens 8000
    t0 = time.time()
    base, key = _endpoint()
    r = gated_post(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"}, body, READ_TIMEOUT[0])
    j = r.json()
    msg = j["choices"][0]["message"]
    obj = _parse_json(msg.get("content"))
    if mapper:
        obj = mapper(obj, ids, sents)
    usage = j.get("usage") or {}
    cost = usage.get("estimated_cost")
    if cost is None:   # provider did not return estimated_cost -> price from list rates
        pt, ct = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
        if "Kimi-K2.6" in body_model:      # DeepInfra list $0.75 / $3.50 per M
            cost = pt * 0.75e-6 + ct * 3.50e-6
        elif "Kimi-K2.5" in body_model:    # DeepInfra list $0.45 / $2.25 per M
            cost = pt * 0.45e-6 + ct * 2.25e-6
        else:                              # groq gpt-oss-120b list $0.15 / $0.75 per M
            cost = pt * 0.15e-6 + ct * 0.75e-6
    meta = {"cost": cost, "prompt_tok": usage.get("prompt_tokens"), "completion_tok": usage.get("completion_tokens"),
            "latency_s": round(time.time() - t0, 2), "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or ""), "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or "")}
    d = obj.get("direction")
    if isinstance(d, int):
        d = str(d)
    ev, normed = eur._norm_evidence(obj.get("evidence") or [], set(ids))
    if d not in eur.DIRECTIONS or ev is None:
        fail = {"direction": "I", "evidence": [], "reason": "", "qc_flag": "read-failed",
                "raw": obj, **meta}
        if j["choices"][0].get("finish_reason") == "length":
            fail["runaway"] = True      # hit max_tokens at temperature 0: a re-run repeats it
        return fail, cost
    flag = "evidence-normalised" if normed else ""
    if d != "I" and not ev:
        d, flag = "X", "empty-directional"
    return {"direction": d, "evidence": ev, "reason": (obj.get("reason") or "")[:120],
            "qc_flag": obj.get("qc_flag") or flag, **meta}, cost


def canon_claim(claim, model, cache):
    """One cached call per claim: the plain-proposition rewrite (CANON_SYS)."""
    ck = cache / ("canon_" + hashlib.sha256(f"{prompt_hash(CANON_SYS)}|{model}|{claim}".encode()).hexdigest()[:24] + ".json")
    if ck.exists():
        return json.load(open(ck))["claim"]
    body = {"model": model, "temperature": 0, "max_tokens": 4000, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": CANON_SYS}, {"role": "user", "content": f"CLAIM: {claim}"}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    base, key = _endpoint()
    for attempt in range(4):
        r = eur._SESSION.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()
    out = (_parse_json(r.json()["choices"][0]["message"].get("content")).get("claim") or "").strip() or claim
    json.dump({"claim": out, "orig": claim}, open(ck, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--prompt", default="v5", choices=sorted(PROMPTS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--reasoning", default=None, choices=["off", "low", "medium", "high"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=190,
                    help="thread pool size; effective in-flight is min(workers, gate target)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--provider", default="deepinfra", choices=sorted(PROVIDERS))
    ap.add_argument("--canon", action="store_true", help="rewrite each claim with CANON_SYS (same model) before reading")
    # RateGate (default, Daniel 2026-09-14): rate-controlled, per-request retry, end-of-run sweep.
    ap.add_argument("--gate-target", type=int, default=160, help="starting in-flight target (RateGate)")
    ap.add_argument("--gate-ceiling", type=int, default=190, help="hard in-flight ceiling per model (RateGate); 190 leaves headroom for other sessions under the 200 account limit")
    ap.add_argument("--gate-floor", type=int, default=32, help="floor the RateGate may cut down to")
    ap.add_argument("--gate-min-sample", type=int, default=20,
                    help="completions a 60 s window needs before its 429 rate can cut (raise for slow "
                         "reasoning models: at ~20 completions/min one 429 reads as a 5%% breach)")
    ap.add_argument("--preflight", action="store_true",
                    help="probe 60s at 96 in-flight on the first --preflight-n units; abort if 429 rate > 20%%")
    ap.add_argument("--preflight-n", type=int, default=200, help="units to use for --preflight")
    ap.add_argument("--legacy-gate", action="store_true",
                    help="use the old AdaptiveGate (AIMD, per-429 cut) with --gate-start/--gate-cap")
    ap.add_argument("--gate-start", type=int, default=32, help="AdaptiveGate initial concurrency (--legacy-gate)")
    ap.add_argument("--gate-cap", type=int, default=96, help="AdaptiveGate ceiling (--legacy-gate)")
    ap.add_argument("--no-oss-json", action="store_true",
                    help="read gpt-oss without forced JSON mode (DeepInfra json pool busy)")
    ap.add_argument("--cache-only", action="store_true",
                    help="score from the read cache alone, make no API calls; a read that is "
                         "not cached stays unread (offline re-scoring, and the memory check)")
    ap.add_argument("--max-tokens", type=int, default=THINK_MAX_TOKENS[0],
                    help="completion budget for a thinking read (DeepInfra admits on the "
                         "reservation, so a high cap costs concurrency)")
    ap.add_argument("--read-timeout", type=int, default=300,
                    help="per-request read timeout in seconds (thinking reads can run minutes)")
    a = ap.parse_args()
    THINK_MAX_TOKENS[0] = a.max_tokens
    READ_TIMEOUT[0] = a.read_timeout
    global GATE
    if a.legacy_gate:
        GATE = AdaptiveGate()
        GATE.limit, GATE.cap, GATE.peak = a.gate_start, a.gate_cap, a.gate_start
    else:
        GATE = RateGate(target=a.gate_target, ceiling=a.gate_ceiling, floor=a.gate_floor,
                        min_sample=a.gate_min_sample)
    OSS_JSON_MODE[0] = not a.no_oss_json

    PROVIDER[0] = a.provider
    sys_prompt = PROMPTS[a.prompt]
    ph = prompt_hash(sys_prompt)
    model = a.model or eur.VERIFICATION_MODEL
    mtag = model.split("/")[-1] + (f"+r{a.reasoning}" if a.reasoning else "") + ("+canon" if a.canon else "") + ("" if a.provider == "deepinfra" else f"@{a.provider}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"; cache.mkdir(exist_ok=True)
    want = json.load(open(a.claims))
    if a.limit:
        want = want[:a.limit]
    by_id = {c["claim_id"]: c for c in want}
    # STREAM the run file. Holding every matched record — the whole 263 MB e1_ctx run with
    # every `sents` list and every stored prompt, query and snippet — is what killed the
    # 2026-09-10 fc_gold leg at 528 reads. Keep only the claim, its date, the per-document
    # {rank, domain, old flag} the output record needs, and the regions still to be read.
    order, claim_of, docs_meta, jobs = [], {}, {}, []
    with open(a.run) as f:
        for line in f:
            r = json.loads(line)
            cid = r.get("review_url")
            if cid not in by_id or r.get("excluded"):
                continue
            order.append(cid)
            claim_of[cid] = (r.get("claim_resolved") or r["claim_text"], r.get("claim_date_shown"))
            dm = []
            for d in r.get("results") or []:
                od = (d.get("read") or {}).get("direction")
                if od:
                    dm.append({"rank": d["rank"], "domain": d.get("domain"), "old": od})
                if d.get("read_status") != "ok" or not d.get("sents"):
                    continue
                jobs.append((cid, d["rank"], None, None) if a.cache_only
                            else (cid, d["rank"], d["sent_ids"], d["sents"]))
            docs_meta[cid] = dm
    missing = [c for c in by_id if c not in claim_of]
    if missing:
        print(f"WARN {len(missing)} claims not in run (excluded or absent)")
    w = weights()

    canon = {}
    if a.canon:
        with ThreadPoolExecutor(a.workers) as ex:
            fut = {ex.submit(canon_claim, claim_of[cid][0], model, cache): cid for cid in order}
            for f in as_completed(fut):
                canon[fut[f]] = f.result()
        print(f"canonicalised {len(canon)} claims, {sum(1 for c in order if canon[c] != claim_of[c][0])} changed", flush=True)
    blocks = {cid: (f"CLAIM: {canon.get(cid) or claim_of[cid][0]}\n(claimed on "
                    f"{claim_of[cid][1] or 'unknown date'}; judge the document's bearing on "
                    f"this exact proposition)") for cid in order}
    print(f"claims {len(order)}  reads {len(jobs)}  prompt {a.prompt} ({ph})  model {model} @ {a.provider}"
          f"{'  CACHE-ONLY (no API calls)' if a.cache_only else ''}", flush=True)

    def ckey(cid, rank):
        return cache / (hashlib.sha256(f"{ph}|{model}{'+r' + a.reasoning if a.reasoning else ''}{'|canon' if a.canon else ''}{'' if a.provider == 'deepinfra' else '@' + a.provider}|{cid}|{rank}".encode()).hexdigest()[:24] + ".json")

    lock = threading.Lock()
    done = [0]; spent = [0.0]; failed = [0]; t0 = [time.time()]
    last_log = [t0[0]]; last_obs = [t0[0]]; last_obs_done = [0]
    results = {}

    def gate_line():
        g = int(GATE.target) if isinstance(GATE, RateGate) else GATE.limit
        return f"gate {g} (inflight {GATE.inflight}, peak {GATE.peak}, {GATE.n_cuts} cuts)"

    def work(job):
        cid, rank, ids, sents = job
        ck = ckey(cid, rank)
        res, cost = None, 0.0
        if ck.exists():
            res = json.load(open(ck))
            if res.get("qc_flag") == "read-failed" and res.get("err"):   # transport failure (429, timeout): retry, not cached
                res = None
        if res is None and a.cache_only:
            return                      # offline pass: an unread slot simply stays unread
        if res is None:
            err = None
            try:
                # gated_post already retries 429/5xx/transport per request with backoff
                res, cost = read_one(sys_prompt, blocks[cid], ids, sents, model, a.reasoning, MAPPERS.get(a.prompt))
            except Exception as e:  # noqa: BLE001  request exhausted its retries
                err = repr(e)
            if res is None:
                res = {"direction": "I", "evidence": [], "reason": "", "qc_flag": "read-failed", "err": err}
            json.dump(res, open(ck, "w"))
        with lock:
            results[(cid, rank)] = res
            done[0] += 1; spent[0] += cost
            if res.get("qc_flag") == "read-failed":
                failed[0] += 1
            now = time.time()
            if res.get("latency_s") is not None:
                _LAT.append((now, res["latency_s"]))
                while _LAT and now - _LAT[0][0] > 60.0:
                    _LAT.popleft()
            if done[0] % 50 == 0 or done[0] == len(jobs) or now - last_log[0] >= 60:
                last_log[0] = now
                el = max(1e-9, now - t0[0])
                print(f"  {done[0]}/{len(jobs)}  ${spent[0]:.3f}  {el:.0f}s  "
                      f"{done[0] / el:.1f} reads/s  ETA {el / done[0] * (len(jobs) - done[0]) / 60:.0f}m  "
                      f"{gate_line()}", flush=True)
            if now - last_obs[0] >= 60 or done[0] == len(jobs):
                emit_observation(model, done[0], spent[0], done[0] - last_obs_done[0],
                                 now - last_obs[0], failed[0])
                last_obs[0] = now; last_obs_done[0] = done[0]

    # PRE-FLIGHT: hold 96 in-flight for 60s on the first N units and check the busy rate before
    # committing the whole run. Reads are cached, so they are reused by the main run, not wasted.
    if a.preflight and isinstance(GATE, RateGate) and not a.cache_only:
        pf = jobs[:a.preflight_n]
        saved = GATE.target; GATE.target = 96.0; GATE.events.clear(); GATE.last_breach = 0.0
        print(f"  [preflight] {len(pf)} units at 96 in-flight for up to 60s ...", flush=True)
        t_pf = time.time()
        with ThreadPoolExecutor(min(a.workers, 96)) as ex:
            futs = [ex.submit(work, j) for j in pf]
            for f in as_completed(futs):
                f.result()
                if time.time() - t_pf >= 60:
                    break
        _n, _n429, rate = GATE.window_stats()
        lats = sorted(l for _, l in _LAT)
        p95 = lats[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else 0.0
        print(f"  [preflight] 429 rate {rate:.1%} over {_n} reqs  p95 {p95:.2f}s", flush=True)
        if rate > 0.20:
            print("  [preflight] ABORT: 429 rate over 20% -- pool too busy", flush=True)
            sys.exit(2)
        GATE.target = saved; GATE.events.clear(); GATE.last_adjust = time.time(); GATE.last_breach = 0.0
        done[0] = 0; last_obs_done[0] = 0; t0[0] = time.time(); last_log[0] = t0[0]; last_obs[0] = t0[0]

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            f.result()

    # FINAL SWEEP: no run should end with silently missing reads. Delete every read-failed cache
    # entry (transport-exhausted OR malformed body) and re-run it, up to 2 sweeps.
    if not a.cache_only:
        for sweep in (1, 2):
            fj = [j for j in jobs
                  if (results.get((j[0], j[1])) or {}).get("qc_flag") == "read-failed"
                  and not (results.get((j[0], j[1])) or {}).get("runaway")]
            if not fj:
                break
            print(f"  [sweep {sweep}] {len(fj)} read-failed reads -> delete cache + re-run", flush=True)
            for j in fj:
                ck = ckey(j[0], j[1])
                if ck.exists():
                    ck.unlink()
            with ThreadPoolExecutor(a.workers) as ex:
                for f in as_completed([ex.submit(work, j) for j in fj]):
                    f.result()
        still = sum(1 for j in jobs if (results.get((j[0], j[1])) or {}).get("qc_flag") == "read-failed")
        runaway = sum(1 for j in jobs if (results.get((j[0], j[1])) or {}).get("runaway"))
        if still:
            print(f"  [sweep] {still} reads still failing after 2 sweeps (recorded read-failed; "
                  f"{runaway} hit max_tokens and were not re-run)", flush=True)

    # A cache-only pass leaves every unread slot carrying its OLD flag, so the file is a
    # MIXTURE of readers. Name it so it can never be mistaken for a finished re-read.
    outf = out / f"{a.prompt}__{mtag}{'.partial-cache-only' if a.cache_only else ''}.jsonl"
    trans = {}
    rows = []
    with open(outf, "w") as f:
        for cid in order:
            old, new, docs = [], [], []
            for d in docs_meta[cid]:
                od = d["old"]
                nr = results.get((cid, d["rank"]))
                nd = nr["direction"] if nr else od
                old.append(od); new.append(nd)
                trans[(od, nd)] = trans.get((od, nd), 0) + 1
                docs.append({"rank": d["rank"], "domain": d["domain"], "old": od, "new": nd,
                             "new_evidence": (nr or {}).get("evidence", []),
                             "new_reason": (nr or {}).get("reason", "")})
            rec = {"claim_id": cid, "post_id": by_id[cid].get("post_id"), "group": by_id[cid].get("group"),
                   "claim": claim_of[cid][0], "claim_canon": canon.get(cid),
                   "old_flags": old, "new_flags": new,
                   "old_score": round(score(old, w), 3), "new_score": round(score(new, w), 3), "docs": docs}
            rows.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metas = [v for v in results.values() if v.get("cost") is not None]
    if metas:
        n = len(metas)
        print(f"\nreads {n}  cost/read ${sum(m['cost'] for m in metas) / n:.6f}  prompt tok/read {sum(m.get('prompt_tok') or 0 for m in metas) / n:.0f}  "
              f"completion tok/read {sum(m.get('completion_tok') or 0 for m in metas) / n:.0f}  latency/read {sum(m['latency_s'] for m in metas) / n:.1f}s  "
              f"failed {sum(1 for m in results.values() if m.get('qc_flag') == 'read-failed')}  "
              f"quote-missing {sum(1 for m in results.values() if m.get('qc_flag') == 'quote-missing')}")
    print(f"\nspent ${spent[0]:.3f}  wrote {outf}  {GATE.summary()}")
    order = ["5", "4", "3", "X", "I", "2", "1"]
    print("\nflag transitions old -> new (rows old, cols new)")
    print("     " + "".join(f"{c:>6}" for c in order))
    for o in order:
        print(f"{o:>4} " + "".join(f"{trans.get((o, n), 0):>6}" for n in order))
    print("\nby group: n, mean old score, mean new score, old below r2 cut, new below r2 cut, old below r1 cut, new below r1 cut")
    groups = sorted({x["group"] for x in rows}, key=str)
    for g in groups:
        xs = [x for x in rows if x["group"] == g]
        n = len(xs)
        mo = sum(x["old_score"] for x in xs) / n; mn = sum(x["new_score"] for x in xs) / n
        o2 = sum(x["old_score"] < CUT_R2 for x in xs); n2 = sum(x["new_score"] < CUT_R2 for x in xs)
        o1 = sum(x["old_score"] < CUT_R1 for x in xs); n1 = sum(x["new_score"] < CUT_R1 for x in xs)
        print(f"  {str(g):>12}  n={n:3d}  {mo:+7.2f} -> {mn:+7.2f}   r1|r2 old {o2:3d}  new {n2:3d}   r1 old {o1:3d}  new {n1:3d}")


if __name__ == "__main__":
    main()
