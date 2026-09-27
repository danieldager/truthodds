from __future__ import annotations

import json
import random
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, quote

import numpy as np
import requests
import trafilatura
from rank_bm25 import BM25Okapi

from config import (
    CRED1_MIN_SCORE,
    CRED1_PATH,
    DOMAIN_DELAY_MAX,
    DOMAIN_DELAY_MIN,
    FCTAPI_ENDPOINT,
    FCTAPI_KEY,
    FCTAPI_MAX_RESULTS,
    MMR_FINAL_K,
    MMR_LAMBDA,
    MMR_SCORE_THRESHOLD,
    RERANK_TOP_K,
    SCRAPE_BLOCKLIST,
    SCRAPE_TIMEOUT,
    SERPER_API_KEY,
    SERPER_ENDPOINT,
    SERPER_TOP_K,
    WIKIPEDIA_MAX_RESULTS,
)
from pipeline.models import AtomicClaim, ClaimEvidence, EvidenceItem

# --- User-agent pool ---
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# --- Domain credibility data (loaded once at startup) ---
_cred1: Dict[str, float] = {}
_cred1_blocklist: Set[str] = set()
_data_loaded = False
_load_lock = threading.Lock()

BLOCKLIST_CATEGORIES = {"fake", "unreliable", "conspiracy"}

# --- Lazy-loaded reranking / embedding models ---
_cross_encoder = None
_embedding_model = None


def _load_domain_data() -> None:
    global _cred1, _cred1_blocklist, _data_loaded
    with _load_lock:
        if _data_loaded:
            return
        try:
            with open(CRED1_PATH) as f:
                raw = json.load(f)
            for domain, info in raw.items():
                score = info.get("credibility_score", 0.5)
                category = info.get("category", "")
                _cred1[domain] = score
                if category in BLOCKLIST_CATEGORIES or score < CRED1_MIN_SCORE:
                    _cred1_blocklist.add(domain)
            print(f"[domain] Loaded {len(_cred1)} domains, {len(_cred1_blocklist)} cred-blocked")
        except FileNotFoundError:
            print(f"[domain] Warning: {CRED1_PATH} not found — domain filtering disabled")
        _data_loaded = True


def _get_credibility(domain: str) -> float:
    return _cred1.get(domain, 0.5)


def _is_cred_blocked(domain: str) -> bool:
    return domain in _cred1_blocklist


def _is_scrape_blocked(url: str) -> bool:
    """Block structurally unscrapable domains (login walls, hard paywalls)."""
    domain = urlparse(url).netloc.lower().lstrip("www.")
    return domain in SCRAPE_BLOCKLIST or any(domain.endswith("." + d) for d in SCRAPE_BLOCKLIST)


# --- Per-domain rate limiting ---
_domain_locks: Dict[str, threading.Lock] = {}
_domain_last_seen: Dict[str, float] = {}
_global_lock = threading.Lock()


def _get_domain_lock(domain: str) -> threading.Lock:
    with _global_lock:
        if domain not in _domain_locks:
            _domain_locks[domain] = threading.Lock()
        return _domain_locks[domain]


def _scrape_with_rate_limit(url: str) -> Optional[str]:
    parsed = urlparse(url)
    domain = parsed.netloc.lower().lstrip("www.")

    lock = _get_domain_lock(domain)
    with lock:
        now = time.time()
        last = _domain_last_seen.get(domain, 0.0)
        delay = random.uniform(DOMAIN_DELAY_MIN, DOMAIN_DELAY_MAX)
        wait = delay - (now - last)
        if wait > 0:
            time.sleep(wait)
        _domain_last_seen[domain] = time.time()

    try:
        headers = {"User-Agent": random.choice(_USER_AGENTS)}
        resp = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
        if resp.status_code in (403, 404):
            return None
        resp.raise_for_status()
        return trafilatura.extract(resp.text, include_comments=False, include_tables=False)
    except Exception:
        return None


# --- Serper search ---
def _search(query: str, date_cutoff: str) -> List[str]:
    payload = {
        "q": query,
        "num": SERPER_TOP_K,
        "tbs": f"cdr:1,cd_min:01/01/1900,cd_max:{date_cutoff}",
    }
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(SERPER_ENDPOINT, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        urls = []
        for r in resp.json().get("organic", []):
            link = r.get("link", "")
            if link and not link.endswith(".pdf"):
                urls.append(link)
            if len(urls) >= SERPER_TOP_K:
                break
        return urls
    except Exception as e:
        print(f"[serper] Error for query '{query}': {e}")
        return []


# --- Google Fact Check Tools API ---
def _query_fctapi(claim_text: str) -> List[EvidenceItem]:
    if not FCTAPI_KEY:
        return []
    try:
        resp = requests.get(
            FCTAPI_ENDPOINT,
            params={"query": claim_text, "key": FCTAPI_KEY, "pageSize": FCTAPI_MAX_RESULTS},
            timeout=10,
        )
        resp.raise_for_status()
        items = []
        for fc in resp.json().get("claims", []):
            for review in fc.get("claimReview", []):
                publisher = review.get("publisher", {}).get("name", "fact-checker")
                rating = review.get("textualRating", "unrated")
                url = review.get("url", "")
                fc_text = fc.get("text", claim_text)
                text = f'Fact-check by {publisher}: The claim "{fc_text}" was rated "{rating}".'
                items.append(EvidenceItem(url=url, text=text, credibility=0.9))
        return items
    except Exception as e:
        print(f"[fctapi] Error: {e}")
        return []


# --- Wikipedia ---
def _query_wikipedia(claim_text: str) -> List[EvidenceItem]:
    try:
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": claim_text,
                "srlimit": WIKIPEDIA_MAX_RESULTS,
                "format": "json",
            },
            headers={"User-Agent": "misinform-pipeline/1.0 (research)"},
            timeout=10,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("query", {}).get("search", [])
        items = []
        for r in results:
            title = r.get("title", "")
            summary_resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
                headers={"User-Agent": "misinform-pipeline/1.0 (research)"},
                timeout=10,
            )
            if summary_resp.ok:
                data = summary_resp.json()
                text = data.get("extract", "")
                if text and len(text) > 100:
                    url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    items.append(EvidenceItem(url=url, text=text, credibility=0.75))
        return items
    except Exception as e:
        print(f"[wikipedia] Error: {e}")
        return []


# --- Model loaders ---
def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("[models] Loading cross-encoder (ms-marco-MiniLM-L6-v2)...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return _cross_encoder


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("[models] Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedding_model


# --- Evidence quality helpers ---
def _bm25_top_chunks(doc_text: str, claim_text: str, top_k: int = 2) -> str:
    """Return the top_k most relevant ~300-word chunks from doc_text."""
    words = doc_text.split()
    if len(words) <= 300:
        return doc_text
    window, stride = 300, 250
    chunks = []
    for i in range(0, len(words), stride):
        chunks.append(" ".join(words[i : i + window]))
    tokenized = [c.lower().split() for c in chunks]
    scores = BM25Okapi(tokenized).get_scores(claim_text.lower().split())
    top_idx = sorted(
        sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    )
    return " [...] ".join(chunks[i] for i in top_idx)


def _rerank_with_scores(
    claim_text: str, evidence: List[EvidenceItem], top_k: int
) -> tuple[List[EvidenceItem], np.ndarray]:
    """Cross-encoder rerank; returns (top_k items, their scores)."""
    if not evidence:
        return [], np.array([])
    scores = np.array(
        _get_cross_encoder().predict([(claim_text, item.text) for item in evidence])
    )
    if len(evidence) <= top_k:
        order = np.argsort(-scores)
        return [evidence[i] for i in order], scores[order]
    top_idx = np.argsort(-scores)[:top_k]
    return [evidence[i] for i in top_idx], scores[top_idx]


def _mmr_diversify(
    evidence: List[EvidenceItem],
    relevance_scores: np.ndarray,
    final_k: int,
    lambda_: float,
    stop_threshold: float = MMR_SCORE_THRESHOLD,
) -> List[EvidenceItem]:
    """MMR pass: greedily select up to final_k items, stopping early when marginal value drops."""
    if len(evidence) <= final_k:
        return evidence
    embs = _get_embedding_model().encode(
        [item.text for item in evidence], normalize_embeddings=True
    )
    min_s, max_s = relevance_scores.min(), relevance_scores.max()
    norm_rel = (relevance_scores - min_s) / (max_s - min_s + 1e-9)
    selected: List[int] = []
    remaining = list(range(len(evidence)))
    for _ in range(final_k):
        best_i, best_score = -1, -float("inf")
        for i in remaining:
            sim = max((float(np.dot(embs[i], embs[j])) for j in selected), default=0.0)
            score = lambda_ * float(norm_rel[i]) - (1 - lambda_) * sim
            if score > best_score:
                best_score, best_i = score, i
        if best_score < stop_threshold and selected:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return [evidence[i] for i in selected]


# --- Public API ---
def retrieve_evidence(
    claim: AtomicClaim,
    date_cutoff: Optional[str] = None,
    verbose: bool = False,
) -> ClaimEvidence:
    """
    Retrieve evidence for a claim from three sources in parallel:
      1. Google Fact Check Tools API (credibility=0.9, if key set)
      2. Wikipedia (credibility=0.75)
      3. Serper web search → Trafilatura scraping (credibility from CRED-1)
    """
    _load_domain_data()

    if date_cutoff is None:
        date_cutoff = datetime.today().strftime("%m/%d/%Y")

    import concurrent.futures

    # --- Run all sources in parallel ---
    serper_urls: List[str] = []
    cred_blocked: List[str] = []
    scrape_blocked: List[str] = []
    seen_urls: Set[str] = set()
    query_results: Dict[str, List[str]] = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        fctapi_future = executor.submit(_query_fctapi, claim.text)
        wiki_future = executor.submit(_query_wikipedia, claim.text)
        search_futures = {executor.submit(_search, q, date_cutoff): q for q in claim.queries}

        for future in concurrent.futures.as_completed(search_futures):
            query = search_futures[future]
            urls = future.result()
            query_results[query] = urls
            for url in urls:
                if url not in seen_urls:
                    seen_urls.add(url)
                    domain = urlparse(url).netloc.lower().lstrip("www.")
                    if _is_cred_blocked(domain):
                        cred_blocked.append(url)
                    elif _is_scrape_blocked(url):
                        scrape_blocked.append(url)
                    else:
                        serper_urls.append(url)

        structured_evidence = fctapi_future.result() + wiki_future.result()

    # --- Verbose: structured sources ---
    if verbose:
        fctapi_items = [e for e in structured_evidence if e.credibility == 0.9]
        wiki_items = [e for e in structured_evidence if e.credibility == 0.75]
        if fctapi_items:
            print(f"    [FCTAPI] {len(fctapi_items)} fact-check(s):")
            for item in fctapi_items:
                print(f"      • {item.text[:120]}")
        if wiki_items:
            print(f"    [Wikipedia] {len(wiki_items)} article(s):")
            for item in wiki_items:
                print(f"      • {item.url}")

        for query, urls in query_results.items():
            print(f"    query: {query}")
            for url in urls:
                domain = urlparse(url).netloc.lower().lstrip("www.")
                if _is_cred_blocked(domain):
                    tag = "CRED-BLOCKED"
                elif _is_scrape_blocked(url):
                    tag = "SCRAPE-BLOCKED"
                else:
                    tag = "ok"
                print(f"      [{tag}] {url}")

    # --- Scrape Serper URLs ---
    scraped_evidence: List[EvidenceItem] = []
    if serper_urls:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(_scrape_with_rate_limit, url): url for url in serper_urls}
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                text = future.result()
                domain = urlparse(url).netloc.lower().lstrip("www.")
                cred = _get_credibility(domain)
                if text and len(text.strip()) > 100:
                    chunk_text = _bm25_top_chunks(text.strip(), claim.text)
                    scraped_evidence.append(EvidenceItem(url=url, text=chunk_text, credibility=cred))
                    if verbose:
                        print(f"      ✓ scraped  cred={cred:.2f}  {url}")
                else:
                    if verbose:
                        print(f"      ✗ empty    cred={cred:.2f}  {url}")

    all_evidence = structured_evidence + scraped_evidence
    reranked, ce_scores = _rerank_with_scores(claim.text, all_evidence, RERANK_TOP_K)
    final_evidence = _mmr_diversify(reranked, ce_scores, MMR_FINAL_K, MMR_LAMBDA)

    if verbose:
        print(f"    [rerank+MMR] {len(all_evidence)} → {len(reranked)} → {len(final_evidence)} items")

    print(
        f"    → {len(all_evidence)} raw  {len(reranked)} reranked  {len(final_evidence)} final"
        f"  ({len(structured_evidence)} structured, {len(scraped_evidence)} scraped)"
        f"  |  {len(cred_blocked)} cred-blocked  {len(scrape_blocked)} scrape-blocked"
    )
    return ClaimEvidence(
        claim=claim,
        evidence=final_evidence,
        blocked_count=len(cred_blocked) + len(scrape_blocked),
        scraped_count=len(scraped_evidence),
    )
