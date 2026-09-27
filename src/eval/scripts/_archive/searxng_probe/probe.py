"""SearXNG upstream-engine blocking probe.

SearXNG has no API of its own — it scrapes the HTML of upstream engines (Google,
Bing, Brave, DuckDuckGo, Startpage, Qwant, Mojeek, Wikipedia, ...). Those engines
CAPTCHA/429 the scraper's IP under bursty load. The JSON response exposes exactly
which engines failed and why via `unresponsive_engines` ([engine, reason] pairs)
and which engines returned each hit via each result's `engines` list. The verifier
(`pipeline/search.py`) discards both; this probe captures them.

We query SearXNG directly (NOT through the verifier) so we control rate/concurrency
and keep the raw fields. Every per-call record is appended to data/<run>.jsonl for
re-analysis.

Subcommands:
  block     fire N varied queries at fixed pacing/concurrency; per-engine block table
  rate      sweep pacing x concurrency; find the throttle threshold
  recovery  hammer one engine to trigger a block, then poll to measure recovery time

Examples:
  uv run eval/scripts/searxng_probe/probe.py block --n 30 --interval 1.0 --workers 1
  uv run eval/scripts/searxng_probe/probe.py rate
  uv run eval/scripts/searxng_probe/probe.py recovery --engine google
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ENDPOINT = os.environ.get("SEARXNG_ENDPOINT", "http://localhost:8888/search")
DATA_DIR = Path(__file__).parent / "data"

# Varied fact-check-style queries (entities, stats, events, claims) so no single
# engine's topical coverage skews the block measurement.
QUERIES = [
    "did the opcw confirm chemical weapons use in syria",
    "nigeria population in 1960 census",
    "did kamala harris pass the california bar exam",
    "covid vaccine mrna myocarditis risk young men",
    "what percentage of the world lives in cities 2020",
    "did pfizer admit vaccine was not tested on transmission",
    "ukraine grain exports black sea deal 2022",
    "world bank gdp per capita united states 2021",
    "did the 2020 us election have widespread fraud",
    "amazon rainforest deforestation rate 2021",
    "did bill gates buy farmland in the united states",
    "global average temperature rise since pre-industrial",
    "did china build islands in the south china sea",
    "us national debt total 2023 trillion",
    "did the wuhan lab leak coronavirus origin",
    "number of refugees from syria civil war",
    "did elon musk buy twitter for 44 billion",
    "renewable energy share of global electricity 2022",
    "did india overtake china as most populous country",
    "great barrier reef coral bleaching percentage",
    "did russia annex crimea in 2014",
    "us inflation rate peak 2022 percent",
    "did the who declare covid a pandemic march 2020",
    "antarctica ice sheet mass loss per year",
    "did facebook change its name to meta",
    "global plastic production million tonnes per year",
    "did north korea test a hydrogen bomb",
    "unemployment rate united states 2023",
    "did the eu ban combustion engine cars 2035",
    "number of species going extinct per year estimate",
    "did saudi arabia normalize relations with israel",
    "world population total 2023 billion",
    "did the colorado river run dry lake mead",
    "electric vehicle sales share europe 2022",
    "did brazil elect lula president 2022",
    "carbon dioxide ppm atmosphere current level",
    "did the titan submersible implode 2023",
    "median household income united states 2022",
    "did japan release fukushima water into the ocean",
    "global life expectancy average years 2021",
]

_pace_lock = threading.Lock()
_last_dispatch = [0.0]


def _pace(interval: float) -> None:
    """Global min-interval gate between request dispatches (decouples rate from workers)."""
    if interval <= 0:
        return
    with _pace_lock:
        wait = interval - (time.time() - _last_dispatch[0])
        if wait > 0:
            time.sleep(wait)
        _last_dispatch[0] = time.time()


def fire(query: str, *, interval: float = 0.0, timeout: int = 20) -> dict:
    """Issue ONE query to SearXNG; return a flat record of the engine outcome.

    `engines_returned` = engines that contributed >=1 result (success).
    `unresponsive`     = [engine, reason] pairs (failure).
    Note: `number_of_results` is SearXNG's summed engine-reported total and is often
    0 even when `n_results` (deduped hits) > 0 — `n_results` is the real "got evidence".
    """
    _pace(interval)
    t0 = time.time()
    rec: dict = {"ts": t0, "query": query}
    try:
        resp = requests.get(
            ENDPOINT,
            params={"q": query, "format": "json", "categories": "general", "language": "en"},
            timeout=timeout,
        )
        resp.raise_for_status()
        d = resp.json()
        engines_returned: Counter = Counter()
        for r in d.get("results", []):
            for e in r.get("engines", []):
                engines_returned[e] += 1
        rec.update(
            latency=round(time.time() - t0, 3),
            status=resp.status_code,
            n_results=len(d.get("results", [])),
            number_of_results=d.get("number_of_results"),
            unresponsive=d.get("unresponsive_engines", []),
            engines_returned=dict(engines_returned),
        )
    except Exception as e:  # noqa: BLE001 — record the failure, don't crash the sweep
        rec.update(latency=round(time.time() - t0, 3), status=None, n_results=0,
                   number_of_results=None, unresponsive=[], engines_returned={},
                   error=str(e)[:200])
    return rec


def run_batch(queries: list[str], *, interval: float, workers: int, bang: str = "") -> list[dict]:
    """Fire `queries` through a `workers`-wide pool, dispatch-paced by `interval`.

    `bang` (e.g. "!brave !bing ") prefixes every query to restrict the engine set —
    lets us load-test a candidate engine config without editing settings.yml.
    """
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fire, bang + q, interval=interval) for q in queries]
        for f in as_completed(futs):
            out.append(f.result())
    out.sort(key=lambda r: r["ts"])
    return out


def write_jsonl(records: list[dict], name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# --------------------------------------------------------------------------- analysis


def engine_stats(records: list[dict]) -> dict[str, dict]:
    """Per-engine participation/return/block counts + reason histogram.

    An engine `participated` in a query if it either returned >=1 result OR appeared
    in unresponsive_engines. block_rate = blocked / participated.
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"participated": 0, "returned": 0, "blocked": 0, "reasons": Counter()}
    )
    for rec in records:
        seen = set()
        for e, n in rec.get("engines_returned", {}).items():
            stats[e]["returned"] += 1
            seen.add(e)
        for e, reason in rec.get("unresponsive", []):
            stats[e]["blocked"] += 1
            stats[e]["reasons"][reason] += 1
            seen.add(e)
        for e in seen:
            stats[e]["participated"] += 1
    return stats


def print_engine_table(records: list[dict]) -> None:
    stats = engine_stats(records)
    rows = []
    for e, s in stats.items():
        part = s["participated"]
        rate = s["blocked"] / part if part else 0.0
        top_reason = s["reasons"].most_common(1)[0][0] if s["reasons"] else "-"
        rows.append((e, part, s["returned"], s["blocked"], rate, top_reason))
    rows.sort(key=lambda r: (-r[4], -r[3]))
    print(f"\n{'engine':<14}{'part':>6}{'return':>8}{'block':>7}{'block%':>9}  top-reason")
    print("-" * 70)
    for e, part, ret, blk, rate, reason in rows:
        print(f"{e:<14}{part:>6}{ret:>8}{blk:>7}{rate*100:>8.0f}%  {reason}")


def summarize(records: list[dict]) -> dict:
    n = len(records)
    got = sum(1 for r in records if r.get("n_results", 0) > 0)
    empty = n - got
    # verifier's "throttled" definition: 0 deduped results AND >=1 unresponsive engine
    throttled = sum(1 for r in records if r.get("n_results", 0) == 0 and r.get("unresponsive"))
    errors = sum(1 for r in records if r.get("error"))
    lat = sorted(r["latency"] for r in records)
    p50 = lat[len(lat) // 2] if lat else 0
    return {"n": n, "got_results": got, "empty": empty, "throttled": throttled,
            "errors": errors, "p50_latency": p50}


def print_summary(records: list[dict], header: str) -> None:
    s = summarize(records)
    print(f"\n=== {header} ===")
    print(f"queries={s['n']}  got-results={s['got_results']} ({s['got_results']/max(s['n'],1)*100:.0f}%)  "
          f"empty={s['empty']}  throttled(verifier-def)={s['throttled']}  "
          f"errors={s['errors']}  p50-latency={s['p50_latency']}s")


# --------------------------------------------------------------------------- commands


def cmd_block(args: argparse.Namespace) -> None:
    queries = (QUERIES * ((args.n // len(QUERIES)) + 1))[: args.n]
    bang = (args.bang + " ") if args.bang else ""
    print(f"Firing {args.n} queries  interval={args.interval}s  workers={args.workers}  bang={bang!r} ...")
    t0 = time.time()
    records = run_batch(queries, interval=args.interval, workers=args.workers, bang=bang)
    wall = time.time() - t0
    tag = ("_" + args.bang.replace("!", "").replace(" ", "+")) if args.bang else ""
    name = f"block_n{args.n}_i{args.interval}_w{args.workers}{tag}"
    path = write_jsonl(records, name)
    print_summary(records, f"BLOCK n={args.n} interval={args.interval}s workers={args.workers} wall={wall:.0f}s")
    print_engine_table(records)
    print(f"\nraw -> {path}")


def cmd_rate(args: argparse.Namespace) -> None:
    """Sweep pacing x concurrency; find where throttling kicks in."""
    intervals = [float(x) for x in args.intervals.split(",")]
    workers = [int(x) for x in args.workers.split(",")]
    n = args.n
    all_records: list[dict] = []
    grid = []
    print(f"Rate sweep: intervals={intervals} x workers={workers}, n={n} each")
    bang = (args.bang + " ") if args.bang else ""
    for w in workers:
        for itv in intervals:
            queries = (QUERIES * ((n // len(QUERIES)) + 1))[:n]
            t0 = time.time()
            recs = run_batch(queries, interval=itv, workers=w, bang=bang)
            wall = time.time() - t0
            for r in recs:
                r["_cell"] = f"i{itv}_w{w}"
            all_records.extend(recs)
            s = summarize(recs)
            qps = n / wall if wall else 0
            grid.append((itv, w, qps, s["got_results"] / n, s["throttled"] / n, s["empty"] / n))
            print(f"  interval={itv:<4} workers={w}: {qps:.2f} q/s  "
                  f"got={s['got_results']}/{n}  throttled={s['throttled']}  empty={s['empty']}")
            time.sleep(args.cooldown)  # let engines recover between cells
    write_jsonl(all_records, f"rate_sweep_n{n}")
    print(f"\n{'interval':>9}{'workers':>9}{'q/s':>7}{'got%':>8}{'throttled%':>12}{'empty%':>9}")
    print("-" * 56)
    for itv, w, qps, got, thr, emp in grid:
        print(f"{itv:>9}{w:>9}{qps:>7.2f}{got*100:>7.0f}%{thr*100:>11.0f}%{emp*100:>8.0f}%")
    print(f"\nraw -> {DATA_DIR / f'rate_sweep_n{n}.jsonl'}")


def cmd_recovery(args: argparse.Namespace) -> None:
    """Hammer to trigger a block, then poll at intervals to measure recovery time.

    Targets a single engine via SearXNG !bang syntax so the block is isolated.
    """
    bang = f"!{args.engine} "
    print(f"Recovery probe for engine={args.engine!r}")
    print(f"Phase 1: hammering {args.engine} with {args.hammer} fast queries (no pacing)...")
    hammer_q = [bang + q for q in (QUERIES * ((args.hammer // len(QUERIES)) + 1))[: args.hammer]]
    run_batch(hammer_q, interval=0.0, workers=args.workers)

    print(f"Phase 2: polling every {args.poll}s for up to {args.max_wait}s ...")
    records = []
    t0 = time.time()
    recovered_at = None
    while time.time() - t0 < args.max_wait:
        rec = fire(bang + QUERIES[len(records) % len(QUERIES)])
        elapsed = round(time.time() - t0, 1)
        rec["elapsed"] = elapsed
        records.append(rec)
        blocked = any(e == args.engine for e, _ in rec.get("unresponsive", []))
        returned = rec.get("engines_returned", {}).get(args.engine, 0)
        state = "BLOCKED" if blocked else (f"OK({returned})" if returned else "silent")
        print(f"  t+{elapsed:>5}s  {state}  n_results={rec['n_results']}")
        if not blocked and returned and recovered_at is None:
            recovered_at = elapsed
            print(f"  -> {args.engine} recovered at t+{elapsed}s")
            if args.stop_on_recovery:
                break
        time.sleep(args.poll)
    write_jsonl(records, f"recovery_{args.engine}")
    print(f"\nrecovery_time({args.engine}) = "
          f"{recovered_at if recovered_at is not None else f'>{args.max_wait}s (not recovered)'}")
    print(f"raw -> {DATA_DIR / f'recovery_{args.engine}.jsonl'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("block", help="per-engine block table at a fixed rate")
    b.add_argument("--n", type=int, default=30)
    b.add_argument("--interval", type=float, default=1.0, help="min seconds between dispatches")
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--bang", default="", help='engine restriction, e.g. "!brave !bing"')
    b.set_defaults(func=cmd_block)

    r = sub.add_parser("rate", help="sweep pacing x concurrency for the throttle threshold")
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--intervals", default="2.0,1.0,0.5,0.0")
    r.add_argument("--workers", default="1,2,4")
    r.add_argument("--cooldown", type=float, default=20.0, help="rest between sweep cells")
    r.add_argument("--bang", default="", help='engine restriction, e.g. "!brave !bing"')
    r.set_defaults(func=cmd_rate)

    rc = sub.add_parser("recovery", help="trigger a block then poll for recovery time")
    rc.add_argument("--engine", default="google")
    rc.add_argument("--hammer", type=int, default=20)
    rc.add_argument("--workers", type=int, default=4)
    rc.add_argument("--poll", type=float, default=30.0)
    rc.add_argument("--max-wait", type=float, default=600.0)
    rc.add_argument("--stop-on-recovery", action="store_true")
    rc.set_defaults(func=cmd_recovery)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
