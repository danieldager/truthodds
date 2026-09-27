"""Shared configuration (API keys, model names, labels).

Pipeline-specific tunables (thresholds, caps, blocklists) live in
`pipeline/config.py`. This file holds only constants shared between the
pipeline and the eval/harvest/harmonisation tooling.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLMs (DeepInfra; same provider for extraction and verification) ---
# DeepSeek-V4-Flash restored on DeepInfra 2026-06-25 (was unresponsive 06-24, no warm capacity;
# the full 8-source harvest's judged_axis was enriched on the V3.1-Terminus stand-in, which matched
# V4-Flash on 83% of a PolitiFact sample — disagreements all on inherently multi-axis claims).
EXTRACTION_BASE_URL = "https://api.deepinfra.com/v1/openai"
EXTRACTION_API_KEY = os.environ.get("DEEPINFRA_API_KEY", "")
EXTRACTION_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

VERIFICATION_BASE_URL = EXTRACTION_BASE_URL
VERIFICATION_API_KEY = EXTRACTION_API_KEY
VERIFICATION_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
REPAIR_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # small model, JSON-syntax repair only (verify_text); degrades to safe default if unavailable

# --- Search provider ("serper" | "tavily" | "searxng" | "exa"). Eval can cascade these. ---
# Pinned in code, NOT read from the environment (2026-07-27). A stale
# SEARCH_PROVIDER=tavily in .env silently rerouted a whole run to a different search
# engine with no error; the results looked plausible and were only caught days later.
# To use another provider, pass provider="..." explicitly at the call site.
SEARCH_PROVIDER = "serper"

# Serper (Google SERP scraper) — production-grade, paid.
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_ENDPOINT = "https://google.serper.dev/search"

# Tavily (agent-native search; ranked results + cleaned page content). Free 1k/mo.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"

# Exa (neural/semantic search + content). Free 1k/mo. Good last-resort for long-tail claims.
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
EXA_ENDPOINT = "https://api.exa.ai/search"

# Brave Search API (real keyed API, NOT scraped → no IP blocking). Free 2k/mo, 1 query/sec.
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# SearXNG (self-hosted meta-search; free, run via repo-root docker-compose). Flaky but $0.
SEARXNG_ENDPOINT = os.environ.get("SEARXNG_ENDPOINT", "http://localhost:8888/search")

# --- Google Fact Check Tools API ---
FCTAPI_KEY = os.environ.get("GOOGLE_FCTAPI_KEY", "")
FCTAPI_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# --- Jina Reader (r.jina.ai) — off-IP body fetch for WAF-walled publishers (e.g. AFP/Akamai) ---
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_READER_ENDPOINT = "https://r.jina.ai/"  # prefix the full target URL; needs Bearer JINA_API_KEY

# --- Embedding model (claim -> vector) ---
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# --- Neon (claims DB) ---
NEON_DATABASE_URL = os.environ.get("NEON_DATABASE_URL", "")

# --- Verdict labels (AVeriTeC 4-class) ---
VERDICT_OPTIONS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence",
]
