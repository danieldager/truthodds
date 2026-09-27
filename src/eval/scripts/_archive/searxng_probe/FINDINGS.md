# SearXNG upstream-engine blocking — findings

**Date:** 2026-06-05 · **Probe:** `probe.py` (block / rate / recovery) · raw → `data/*.jsonl`
**Setup:** self-hosted `searxng` container (port 8888), default engine set, single
home/office egress IP. Queried directly (not through the verifier) to keep the raw
`unresponsive_engines` + per-result `engines` fields.

## TL;DR

- **Visibility is solved.** Every block is directly observable: `unresponsive_engines`
  gives `[engine, reason]` (CAPTCHA / too many requests / access denied / timeout) and
  each result's `engines` list shows which engine returned it. `_fetch_searxng` currently
  discards both — capturing them is a free observability win.
- **On a single IP, SearXNG cannot carry tier-1 under bursty/concurrent load.** The default
  config drops to **~10% useful results** under the verifier's real condition (1s pacing,
  4 concurrent workers) — reproducing the cascade eval's 63% throttle (here 90%).
- **The blocking is per-engine and mostly IP-reputation, not pacing.** Google, Startpage,
  Mojeek, Qwant hard-block this IP (CAPTCHA / access-denied) on the *first cold request*.
  Brave is the only good-quality engine but rate-limits after a handful of rapid requests
  and SearXNG then suspends it with escalating backoff. DuckDuckGo is flaky (CAPTCHA/timeout).
- **The one load-proof engine, Bing, returns junk.** Bing never blocked across 200+ bursty
  requests — but through SearXNG's scraper it returns generic *homepages* (facebook.com,
  nytimes.com, a person's `.edu`), not query-specific evidence. "Reliable" but useless for
  a verifier, and it would silently defeat the empty+unresponsive throttle guard.
- **Recommendation:** demote SearXNG from a load-bearing tier-1; lean on the cascade
  fallback for batch/eval load, and for the free production tier-1 switch to the **official
  Brave Search API** (keyed, 2k free/mo, *not* scraped → no IP blocking). See below.

## Q1/Q2 — Which engines block, who survives (this IP)

Cold per-engine probe (one query each, 2s apart, fresh container):

| engine      | cold result        | under bursty load (4w/0s)        | verdict |
|-------------|--------------------|----------------------------------|---------|
| **bing**    | 10 results ✓       | **100% (0 blocks, n=120)**       | load-proof but **low-quality results** (homepages) |
| **brave**   | 19 results ✓ (good quality) | **2%** (98% "too many requests") | best quality, **fragile** — suspends fast |
| duckduckgo  | timeout / flaky    | ~10% (CAPTCHA)                   | unreliable |
| wikipedia   | 0 (no error)       | suspends under load              | narrow (entities only) |
| google      | **Suspended: CAPTCHA** | 100% blocked                 | IP-blocked, useless |
| startpage   | **Suspended: CAPTCHA** | 100% blocked                 | IP-blocked (uses Google), useless |
| mojeek      | **access denied**  | —                                | IP-blocked, useless |
| qwant       | **access denied**  | —                                | IP-blocked, useless |
| wikidata    | HTTP error         | 100% blocked                     | useless |

Default-config block table, verifier's real condition (1s pacing, 4 workers, n=40, fresh):

```
engine          part  return  block   block%  top-reason
brave             40       0     40     100%  Suspended: too many requests
google            40       0     40     100%  Suspended: CAPTCHA
startpage         40       0     40     100%  Suspended: CAPTCHA
wikipedia         20       0     20     100%  Suspended: too many requests
duckduckgo        40       4     36      90%  CAPTCHA
→ got-results 4/40 (10%), throttled 36/40
```

**Key quality finding.** Bing's 100% reliability is hollow. For "elon musk twitter 44
billion acquisition" Bing returns Elon Musk's Wikipedia page, `elon.edu`, a YouTube video,
a Forbes profile; for "world population 2023" it returns PBS/ABC/NYT/Reuters *homepages*;
for "facebook renamed meta" it returns `facebook.com` and the App Store page. These are
navigational hits, not the fact-check evidence a verifier needs. Brave (when not blocked)
returns relevant pages. So the trade is **quality (brave, fragile) vs reliability (bing,
junk)** — no single scraped engine on this IP gives both.

## Q3 — How long do blocks last (recovery)

`probe.py recovery --engine brave` (hammer → poll every 30s):

Hammer brave with 8 fast concurrent queries → it suspends ("too many requests") → poll
every 30s:

```
t+  0.0s  BLOCKED
t+ 30.0s  BLOCKED
t+ 60.1s  BLOCKED
t+ 90.1s  BLOCKED
t+121.7s  OK(20)   ← recovered
recovery_time(brave) = ~122s (~2 min)
```

So **brave's rate-limit suspension is short (~2 min)** — it's a recoverable throttle, not a
hard ban. This is the one mildly encouraging result: with strictly serial, slowly-paced
calls, brave can stay alive (hammer → 2 min cooldown → works again). It dies only under
*concurrent burst*. Contrast google/startpage, whose CAPTCHA suspension is hours-to-24h
(governed by SearXNG's `suspended_times` and the upstream IP flag) — effectively permanent
for a session.

The `Suspended:` prefix is SearXNG's **own** penalty: after an engine errors, SearXNG
suspends it for `suspended_times` (default ~1h for "too many requests", up to 24h for
CAPTCHA/access-denied) with escalating backoff on repeated failures. So two timescales
compound: (1) the upstream IP cooldown, and (2) SearXNG's suspension. Once Google/Startpage
CAPTCHA, SearXNG won't even retry them for hours — they stay dead for the rest of a session.
A single afternoon of testing semi-permanently flagged this IP across most engines, which is
itself the core operational reality of one-IP self-hosted SearXNG.

## Q4 — Rate / concurrency threshold

Effective threshold on this IP, post-light-use: **below 1 serial request/second** for the
quality engines. Brave collapses after ~7 rapid requests (4 workers, no pacing → 2% success)
and 1s-paced 4-worker load still drives the default config to 10%. Bing has **no** observed
threshold (0 blocks at 8 workers / 0 pacing, n=120) — but see the quality caveat. There is no
pacing/concurrency setting that makes the *quality* engines (brave/ddg) reliable under
concurrent load on a single IP; the limit is IP reputation, not request rate.

## Q5 — Config recommendation

**SearXNG cannot be the load-bearing free tier-1 on a single IP.** Two concrete paths:

### Recommended: replace the free tier-1 with the official Brave Search API
Keyed API, 2k free queries/mo, returns Brave's real index — **not scraped**, so no CAPTCHA /
IP-reputation failure mode at all. This is the clean fix and matches IDEA-012. Brave already
proved the highest result *quality* here; the API removes its only weakness (the scrape
block). SearXNG stays as a best-effort $0 option for low-rate/single-user traffic only.

### If keeping SearXNG: trim the engine set + serialize
- `searxng/settings.yml`: **disable** google, startpage, mojeek, qwant, wikidata (hard-blocked
  / low-value on this IP); keep brave, duckduckgo, wikipedia; keep bing **only as a
  last-resort floor**, flagged low-confidence (its homepage-junk would otherwise mask a real
  retrieval failure).
- Serialize SearXNG calls (1 worker) at ≥2–3s pacing — viable for production single-user
  nudge traffic (claims trickle in), **not** for batch evals.
- **For evals and any concurrent load, do not rely on SearXNG** — the cascade already escalates
  to Serper/Tavily/Exa and hit 0.768 with SearXNG mostly failing. Keep that as the design.
- **Cheap win regardless:** make `_fetch_searxng` capture `unresponsive_engines` (log /
  return it) so tier-1 health is observable in production instead of silent.

```yaml
# settings.yml — trim to the engines that don't hard-block this IP
engines:
  - name: google
    disabled: true
  - name: startpage
    disabled: true
  - name: mojeek
    disabled: true
  - name: qwant
    disabled: true
  - name: wikidata
    disabled: true
```

## Reproduce

```
uv run eval/scripts/searxng_probe/probe.py block --n 40 --interval 1.0 --workers 4      # default condition
uv run eval/scripts/searxng_probe/probe.py block --n 40 --interval 0.0 --workers 4 --bang "!bing"   # isolate one engine
uv run eval/scripts/searxng_probe/probe.py rate                                          # pacing x concurrency sweep
uv run eval/scripts/searxng_probe/probe.py recovery --engine brave --stop-on-recovery    # block-duration
```
Restart the container (`docker compose restart searxng`) to clear SearXNG-side suspensions
between runs; the upstream IP cooldowns persist regardless.
