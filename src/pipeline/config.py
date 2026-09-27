"""Pipeline v0.2 tunables. Shared constants (keys, model names, labels) live in `src/config.py`."""

# --- Tier 1: claims cache (see docs/tier1_cache_design.md; pinned by eval/scripts/cache_eval) ---
# The cosine cut is a RECALL filter only — the equivalence GATE, not this threshold, bounds
# precision (cache_eval Run 1: gated precision ~0.98 flat across the whole sweep). So keep it
# low to preserve recall; raising it just discards true equivalents for no precision gain.
SIMILARITY_THRESHOLD = 0.60        # ANN candidate cut (stage 1); the gate decides reuse (stage 2)
CACHE_TOP_K = 5                    # nearest cached claims (distinct clusters) sent to the gate
RECHECK_AFTER_DAYS = 30            # TTL for 'confident' verdicts; older hits re-verify
RECHECK_AFTER_DAYS_LOWCONF = 3     # short TTL for 'redundant'/'cap' verdicts (loop was cut off)
CACHE_GATE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"  # bidirectional equivalence judge

# --- Tier 2: trusted publishers (same set used to build eval_v1.parquet) ---
TRUSTED_PUBLISHERS = {
    "snopes.com",
    "factcheck.afp.com",
    "newschecker.in",
    "verafiles.org",
    "rumorscanner.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
}

# --- Tier 3: ClaimCheck loop ---
MAX_ROUNDS = 4                 # hard cap on synthesise-or-query iterations (single-provider)
SEARCH_RETRIEVE_K = 10         # results fetched per query. DO NOT RAISE: this plan silently
                               # ignores num and returns <=10 organic either way (29,830 cached
                               # responses, max 10 ever; probed live at num=10/20/30/100).
                               # Serper's own `credits` field reads 1 for all of those, so raising
                               # num is free but useless. Depth comes from `page`, not `num`, and
                               # search()'s cache key omits page. See clog/280826, oracle_probe S2b.
                               # (clog/120626 pinned 2 credits for num>10 from the then-live docs;
                               #  our key does not bill that way now.)
SEARCH_KEEP_K = 5              # results kept after credibility rerank (each = 1 summariser call)
REDUNDANCY_THRESHOLD = 0.9     # cosine sim above which a follow-up is redundant
SCRAPE_TIMEOUT = 10            # seconds per URL
DATE_CEILING_MODE = "today"    # "today" or "claim_date" (eval override)

# --- Provider cascade: lean on Serper, escalate to Exa when not confident after N rounds ---
# Serper (broad Google index) carries the load; Exa (neural) adds complementary sources when
# Serper stalls. Tavily dropped 2026-06-12 — the reachability test showed it resolves nothing
# Serper can't (clog/120626.md). SearXNG demoted (IP-block under load; searxng_probe FINDINGS).
SEARCH_CASCADE = ["serper", "exa"]
ROUNDS_PER_PROVIDER = 2         # Serper gets 2 rounds, then escalate to Exa (w/ Daniel, clog 260626)
FUTILITY_STALE_ROUNDS = 1       # conclude after N consecutive rounds with NO new relevant evidence
# (Exa still gets one shot first); stops re-asking for confirmation of a fabrication that isn't on
# the web. Validated 2026-06-30: -13% calls / -24% latency, verdict preserved 9/9 on fired claims.

SEARCH_TOP_K = SEARCH_RETRIEVE_K  # back-compat alias

# --- Per-domain scrape rate-limiting (random delay between requests to same host) ---
DOMAIN_DELAY_MIN = 2.0
DOMAIN_DELAY_MAX = 3.5

# --- Scrape blocklist (login walls only) ---
# Only domains with no public read at all (login walls). Newspapers with paywalls
# are NOT blocked here — when scrape returns None, the search-engine snippet
# fallback kicks in, so we still get partial evidence. Empirically, removing the newspaper
# block restored a major source of fact-check-relevant content (Reuters, AP, etc.).
SCRAPE_BLOCKLIST = {
    "facebook.com", "m.facebook.com", "l.facebook.com",
    "twitter.com", "x.com", "t.co",
    "instagram.com",
    "tiktok.com",
    "linkedin.com",
    "youtube.com", "youtu.be",
    # UGC platforms (Daniel 2026-07-21): 6.4% of raw Serper results in the Truth Odds
    # profile were reddit/threads/quora/medium/scribd-class pages — never evidence,
    # excluding them server-side frees result slots. NOTE: editing this set rotates the
    # search cache key by design (stale excludes must not persist).
    "reddit.com", "threads.com", "quora.com", "scribd.com",
    "fandom.com", "tumblr.com", "pinterest.com",
    # UGC stragglers surfaced by the 2026-07-23 unrated-pool census (unrated_pool_census.py):
    # platforms/forums/podcast hosts that appeared in results but are never evidence.
    # grokipedia.com added on Daniel's explicit call (LLM-generated encyclopedia).
    # NB: Google honors only a limited number of `-site:` operators (excess silently
    # ignored — see _fetch_serper); with 30+ entries some server-side excludes may leak,
    # but the client-side drop + scrape block still guarantee none becomes evidence.
    "bsky.app", "dailymotion.com", "nairaland.com", "usmessageboard.com",
    "patreon.com", "podcasts.apple.com", "open.spotify.com", "music.amazon.com",
    "audioboom.com", "feeds.acast.com", "pod.wave.co", "spreaker.com",
    "dokumen.pub", "localguidesconnect.com", "grokipedia.com",
}

# 2026-07-27 (Daniel): the pre-filter is now UGC ONLY — platforms where anyone posts
# and the "source" is an anonymous account. Everything else is retrievable and gets
# judged on its merits downstream. UNBLOCKED on this call: researchgate.net,
# academic.oup.com, jstor.org (scientific sources are worthwhile signal even when we
# can only reach an abstract — getting the text is a fetch problem to solve, not a
# reason to never see the result) and medium.com. Measured organic cost of the
# unblock: 0.36% of results. Source *quality* filtering moved to a POST-retrieval
# filter, decided after we inspect the first round of results.

# --- Fact-check domains (EVAL-ONLY retrieval exclusion) ---
# Excluded from search ONLY when an eval asks the verifier to reach a verdict WITHOUT reading a
# fact-checker — so we measure independent verification, not fact-check lookup. NOT used in
# production, where finding a fact-check is exactly what Tier 2/3 want. Our 9 gold-source publishers
# plus the major EN/FR dedicated fact-checkers Google surfaces for these claims. When excluding
# these starves a query below SEARCH_KEEP_K results, the Serper fetcher pulls a second page and
# backfills (search.py). Note: mixed news/FC domains (aap.com.au, 20minutes.fr, afp.com) are blocked
# whole — strict, since they're our gold sources; pure-news evidence sites (Reuters/AP) are NOT here.
FACT_CHECK_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "leadstories.com",
    "aap.com.au", "afp.com", "factcheck.afp.com", "factuel.afp.com", "20minutes.fr",
    "checkyourfact.com", "truthorfiction.com", "africacheck.org", "factcheckni.org",
    "healthfeedback.org", "sciencefeedback.co", "climatefeedback.org", "verafiles.org",
    "logically.ai", "misbar.com", "demagog.org.pl", "maldita.es", "newtral.es",
    "correctiv.org", "dpa-factchecking.com",
}
