"""All knobs for the claim-level verify loop, pinned to the v7.5 defaults
(pipeline/verify_tweet_claims.py PostVerifyConfig, 2026-07-21) plus the arm switches.

API keys come from the repo root `config.py` (.env: DEEPINFRA_API_KEY, SERPER_API_KEY,
EXA_API_KEY, JINA_API_KEY). Nothing here reads the environment directly.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    EXA_API_KEY, EXA_ENDPOINT, JINA_API_KEY, JINA_READER_ENDPOINT, SERPER_API_KEY,
    SERPER_ENDPOINT, VERIFICATION_API_KEY, VERIFICATION_BASE_URL, VERIFICATION_MODEL,
)

LOOP_VERSION = "v7.6-claim"

# --- Scrape tunables (pipeline/config.py) ------------------------------------------------
SCRAPE_TIMEOUT = 10
DOMAIN_DELAY_MIN = 2.0
DOMAIN_DELAY_MAX = 3.5

# UGC / login-wall domains: never retrieved, never scraped (pipeline/config.py SCRAPE_BLOCKLIST,
# 2026-07-27 policy: UGC only). Editing this rotates the Serper cache key by design.
SCRAPE_BLOCKLIST = {
    "facebook.com", "m.facebook.com", "l.facebook.com",
    "twitter.com", "x.com", "t.co",
    "instagram.com", "tiktok.com", "linkedin.com", "youtube.com", "youtu.be",
    "reddit.com", "threads.com", "quora.com", "scribd.com",
    "fandom.com", "tumblr.com", "pinterest.com",
    "bsky.app", "dailymotion.com", "nairaland.com", "usmessageboard.com",
    "patreon.com", "podcasts.apple.com", "open.spotify.com", "music.amazon.com",
    "audioboom.com", "feeds.acast.com", "pod.wave.co", "spreaker.com",
    "dokumen.pub", "localguidesconnect.com", "grokipedia.com",
}

# Fallback token prices (USD per 1M tokens) used ONLY when DeepInfra's usage object carries
# no `estimated_cost`. Derived from the run ledger's list-price meter (2026-08-28: a 472-in /
# 8-out call metered at $9.7e-5, i.e. ~$0.20/M in); the ledger notes real spend runs ~7x
# below that meter, so this is a conservative ceiling. Verify against DeepInfra's pricing
# page before quoting a dollar figure.
PRICE_PER_M = {"prompt": 0.20, "cached": 0.02, "completion": 0.80}


@dataclass
class ClaimVerifyConfig:
    # --- v7.5 loop constants (unchanged) ---
    tries_per_claim: int = 2       # targeted Serper queries before code marks it unsupported
    serper_k: int = 10             # results requested per query (Serper returns <=10 anyway)
    exa_k: int = 5
    cap_tok: int = 2500            # article cap fed to READ
    ctx_window: int = 3            # +-sentences of context around cited segs
    max_segs: int = 8
    open_max_tokens: int = 300     # QUERY
    read_max_tokens: int = 2500    # READ
    step_max_tokens: int = 3200    # RESOLVE
    exa_query_max_tokens: int = 400
    requery_max_tokens: int = 120
    triage_max_tokens: int = 300
    dossier_entries_cap: int = 5
    max_rounds: int = 0            # 0 = unlimited (tries accounting bounds the loop)
    # --- arm switches ---
    triage_enabled: bool = False   # v7.5 default is True; the arms read a FIXED count instead
    pages_per_round: int = 3       # docs read per query when triage is off (3 or 10)
    snippet_evidence: str = "fallback"  # "fallback" (v7.5 with triage off: only unreadable
                                        # hits contribute a snippet row) | "all" (every hit)
    exa_enabled: bool = False      # one Exa escalation round; 1k free req/mo — explicit go only
    date_ceiling: bool = True      # Serper tbs cd_max = claim date (M/D/YYYY, inclusive)
    ceiling_date: str | None = None  # fixed ISO ceiling for every claim instead of the claim
                                     # date (e.g. a comparison paper's submission date)
    origin_exclusion: bool = True  # exclude the claim's origin site (origin.origin_domain)
    code_bar: bool = True          # enforce the closure bar in code (_qualifying/_meets_bar);
                                   # False = take the RESOLVE model's proposed close as final
                                   # (guard event "bar-off-accepted")
    ugc_blocklist: bool = True     # SCRAPE_BLOCKLIST: Serper -site: suffix, client-side drop
    fc_undated_drop: bool = False  # drop UNDATED hits from fact-check hosts (the urn run's
                                   # fc-undated rule; on for fc-gold parity, off on AVeriTeC)
                                   # and scrape gate. Applied at pool construction (make_pools).
    # --- model ---
    model: str = VERIFICATION_MODEL
    temperature: float = 0.0
    llm_cache: bool = True         # exact-match response cache (disk_cache namespace "llm")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrchestrationConfig:
    """Concurrency knobs (pipeline/pools.py OrchestrationConfig, re-sized for this run).
    LLM 150 = measured flat latency to 192 concurrent with zero 429s (clog/100726).
    Serper 16 = a ramp from the validated 8 (x-ratelimit-limit 500 seen live).
    Scrape 64/2 = twice the validated global cap; scrape was the measured bottleneck."""
    llm_pool_size: int = 150
    llm_model: str = VERIFICATION_MODEL
    llm_ttft_timeout: float = 30.0
    llm_idle_timeout: float = 30.0
    llm_connect_timeout: float = 15.0
    serper_pool_size: int = 16
    serper_timeout: float = 20.0
    exa_pool_size: int = 3
    exa_timeout: float = 30.0
    scrape_pool_size: int = 64
    scrape_per_domain: int = 2
    scrape_wall_timeout: float = 120.0
    backoff_base: float = 1.0
    backoff_cap: float = 60.0
    backoff_jitter: float = 0.5
    parse_reasks: int = 1
    breaker_window: int = 30
    breaker_min_events: int = 10
    breaker_err_rate: float = 0.5
    breaker_cooldown: float = 30.0
    k_claims: int = 100            # claims in flight
    progress_every: float = 60.0   # s between flushed status lines (plus one every 10 claims)
    stall_alarm_s: float = 300.0
    n_shards: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["ClaimVerifyConfig", "OrchestrationConfig", "LOOP_VERSION", "SCRAPE_BLOCKLIST",
           "SCRAPE_TIMEOUT", "DOMAIN_DELAY_MIN", "DOMAIN_DELAY_MAX", "PRICE_PER_M", "SRC",
           "EXA_API_KEY", "EXA_ENDPOINT", "JINA_API_KEY", "JINA_READER_ENDPOINT",
           "SERPER_API_KEY", "SERPER_ENDPOINT", "VERIFICATION_API_KEY",
           "VERIFICATION_BASE_URL", "VERIFICATION_MODEL"]
