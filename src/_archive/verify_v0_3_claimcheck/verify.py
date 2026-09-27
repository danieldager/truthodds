"""Tier 3: ClaimCheck-faithful verification loop.

One LLM throughout. Per claim:

  1. Planning      — produce ONE opening search query.
  2. Execution     — Serper top-k (no LLM); every result keeps its snippet.
  3. Summarisation — per *scraped* URL, an LLM summary + direct quotes. Snippet-only
                     docs (scrape failed) keep their snippet raw, no LLM call.
  4. Synthesis     — analyse the pool; emit an analysis AND a next_query (or null).
  5. Evaluation    — two evaluators on the final analysis: our 4-dim Likert and the
                     ClaimCheck 4-class.

Confidence-gated loop: re-query only when synthesis is not yet confident (it returns a
non-null next_query). Stop when next_query is null ("confident"), the follow-up is
redundant, or the search budget (MAX_ROUNDS) is exhausted.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from openai import OpenAI

from config import (
    SEARCH_PROVIDER,
    VERIFICATION_API_KEY,
    VERIFICATION_BASE_URL,
    VERIFICATION_MODEL,
)
from pipeline import credibility, disk_cache, search, verify_prompts as vp
from pipeline.config import (
    FUTILITY_STALE_ROUNDS,
    MAX_ROUNDS,
    REDUNDANCY_THRESHOLD,
    ROUNDS_PER_PROVIDER,
    SEARCH_KEEP_K,
    SEARCH_RETRIEVE_K,
)
from pipeline.embedding import embed
from pipeline.models import AtomicClaim, ClaimVerdict, EvidenceDoc, VerdictScores

# max_retries: the SDK retries 429/5xx with exponential backoff (and honours Retry-After).
# Raised from the default 2 → 8 after a -w6 eval saturated Groq's 250k-TPM and threw 208
# rate-limit errors that tainted a full run (clog/150626).
_client = OpenAI(base_url=VERIFICATION_BASE_URL, api_key=VERIFICATION_API_KEY, max_retries=8,
                 timeout=90.0)  # per-call cap: healthy calls finish <=30s (p99 28s), hangs sit at
# 62s/448s with a clean dead zone between — so 90s kills only true endpoint hangs (token-free on a
# non-streaming abort: no completed response returned => no usage billed) and the SDK auto-retries.

# --- Prompt-cache telemetry --------------------------------------------------
# DeepInfra caches byte-identical message prefixes automatically (no request param) and
# reports the hit in usage.prompt_tokens_details.cached_tokens. Our prompts put the large
# static instruction block in the system message and only variable content in the user
# message, so each step's system prefix is cache-eligible across claims. These run-level
# totals (thread-safe; verify() runs under eval worker pools) let a run confirm the cache is
# actually landing. reset_cache_stats() before a run, get_cache_stats() after.
_cache_lock = threading.Lock()
_cache_stats = {"llm_calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}


def reset_cache_stats() -> None:
    with _cache_lock:
        for k in _cache_stats:
            _cache_stats[k] = 0


def get_cache_stats() -> dict:
    """Run-level token totals + cache hit rate (% of prompt tokens served from cache)."""
    with _cache_lock:
        s = dict(_cache_stats)
    s["cached_pct"] = round(100 * s["cached_tokens"] / s["prompt_tokens"], 1) if s["prompt_tokens"] else 0.0
    return s


def _record_usage(usage) -> None:
    if usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or 0
    with _cache_lock:
        _cache_stats["llm_calls"] += 1
        _cache_stats["prompt_tokens"] += usage.prompt_tokens or 0
        _cache_stats["cached_tokens"] += cached
        _cache_stats["completion_tokens"] += usage.completion_tokens or 0


# Token caps. Raised for DeepSeek V4 Flash: when reasoning is on, reasoning_content counts
# toward completion tokens, so the cap must cover the thinking preamble AND the JSON answer or
# the JSON gets truncated. max_tokens is a ceiling, not a reservation — generous caps cost
# nothing when reasoning is off (output stays small).
_PLAN_MAX_TOKENS = 2000
_SUMMARISE_MAX_TOKENS = 4000
_SYNTHESISE_MAX_TOKENS = 8000
_EVALUATE_MAX_TOKENS = 8000
_DOC_TRUNCATE_CHARS = 12000


# =============================================================================
# Public entry point
# =============================================================================

def verify(
    claim: AtomicClaim,
    date_ceiling: str | None = None,
    exclude_urls: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    verbose: bool = False,
    model: str = VERIFICATION_MODEL,
    provider: str | None = None,
    providers: list[str] | None = None,
    rounds_per_provider: int | list[int] = ROUNDS_PER_PROVIDER,
    futility_stale: int = FUTILITY_STALE_ROUNDS,
    trace: dict | None = None,
    post_context: str | None = None,
    image_context: str | None = None,
) -> ClaimVerdict:
    """Run the Tier-3 loop. Always returns a verdict.

    Single provider (default): up to MAX_ROUNDS confidence-gated rounds with `provider`
    (else config.SEARCH_PROVIDER). Cascade: pass `providers=[...]` to escalate free-first —
    run a provider for `rounds_per_provider` rounds; if synthesis still isn't confident, move
    to the next provider (evidence accumulates). Stops early the moment it is confident.

    Each query retrieves SEARCH_RETRIEVE_K results; they are credibility-reranked (social
    dropped, trusted sources preferred) down to SEARCH_KEEP_K before summarising.

    date_ceiling/exclude_urls: eval leakage control. trace: if a dict is passed it is filled
    with the COMPLETE run record (every query/source/summary/decision + per-step timing).
    image_context: a text serialization of the post's image, fed as CLAIM CONTEXT (not evidence)
    to every claim-reasoning step — how the text-only verifier 'sees' image-borne claims.
    """
    t_start = time.time()
    cascade = providers or [provider or SEARCH_PROVIDER]
    # Per-provider round budgets (e.g. serper 3, exa 2). An int means uniform across the cascade.
    rpp = list(rounds_per_provider) if isinstance(rounds_per_provider, (list, tuple)) \
        else [rounds_per_provider] * len(cascade)
    if len(rpp) < len(cascade):
        rpp += [rpp[-1]] * (len(cascade) - len(rpp))
    max_rounds = sum(rpp[:len(cascade)]) if len(cascade) > 1 else MAX_ROUNDS
    exclude_urls = set(exclude_urls or [])
    evidence_pool: list[EvidenceDoc] = []
    seen_urls: set[str] = set()
    rounds: list[dict] = []  # full per-round trace
    counters = {"n_urls_seen": 0, "n_scraped": 0, "n_blocked_or_failed": 0,
                "n_snippet_used": 0, "n_irrelevant": 0, "n_date_filtered": 0,
                "n_reranked_out": 0, "n_search_errors": 0, "n_resampled": 0}
    llm_calls = 0
    pi = 0  # index of the current provider in the cascade

    def _search(query: str, prov: str) -> tuple[list[dict], float, str | None]:
        """Search → (results, elapsed_s, error). API failures surface as a counted error."""
        t = time.time()
        try:
            # exclude_domains (eval fact-check block) + min_results backfill: if the block starves a
            # query below SEARCH_KEEP_K, the provider pulls a second page so the verifier keeps ~5.
            st: dict = {}
            res = search.search(query, SEARCH_RETRIEVE_K, date_ceiling=date_ceiling, provider=prov,
                                exclude_domains=exclude_domains,
                                min_results=SEARCH_KEEP_K if exclude_domains else 0, stats=st)
            if st.get("resampled"):
                counters["n_resampled"] += 1  # this query was FC-starved → page-2 backfill
            return res, round(time.time() - t, 3), None
        except search.SearchError as e:
            counters["n_search_errors"] += 1
            if verbose:
                print(f"  SEARCH ERROR ({prov}): {e}")
            return [], round(time.time() - t, 3), str(e)

    def _consume(query: str, prov: str, results: list[dict], search_s: float, search_err: str | None) -> None:
        nonlocal llm_calls
        new = [r for r in results
               if r["url"] not in seen_urls and r["url"] not in exclude_urls]
        seen_urls.update(r["url"] for r in new)
        counters["n_urls_seen"] += len(new)
        # Credibility rerank: drop social, prefer trusted sources, keep top-K to summarise.
        kept_results = credibility.rerank(new, SEARCH_KEEP_K)
        counters["n_reranked_out"] += len(new) - len(kept_results)

        t = time.time()
        docs, stats = _gather_evidence(kept_results, claim.text, model)
        gather_s = round(time.time() - t, 3)
        for k in ("n_scraped", "n_blocked_or_failed", "n_snippet_used", "n_irrelevant"):
            counters[k] += stats[k]
        llm_calls += stats["llm_calls"]

        if date_ceiling:
            kept = []
            for d in docs:
                pub = _parse_iso(d.publication_date)  # parse mixed formats before comparing
                if pub and pub > date_ceiling and not credibility.is_reference(d.url):
                    counters["n_date_filtered"] += 1  # genuine post-claim NEWS (evergreen exempt)
                else:
                    kept.append(d)
            docs = kept
        evidence_pool.extend(docs)
        rounds.append({
            "round": len(rounds) + 1,
            "provider": prov,
            "query": query,
            "search": {"n_results": len(results), "n_new": len(new), "n_kept": len(kept_results),
                       "elapsed_s": search_s, "error": search_err},
            "gather_elapsed_s": gather_s,
            "docs": [d.model_dump() for d in docs],
            "synthesis": None,
        })
        if verbose:
            print(f"  [{prov}] +{len(docs)} docs (kept {len(kept_results)}/{len(new)}, "
                  f"scraped {stats['n_scraped']}, snippet {stats['n_snippet_used']}, irrel {stats['n_irrelevant']})")

    # --- Planning + opening search (first provider) ---
    q = _clean_query(_plan_query(claim.text, model, image_context), claim.text)
    llm_calls += 1
    past_queries = [q]
    if verbose:
        print(f"[round 1/{max_rounds}] [{cascade[pi]}] query: {q}")
    _consume(q, cascade[pi], *_search(q, cascade[pi]))
    rounds_this_provider = 1
    providers_used = [cascade[pi]]

    # --- Confidence-gated loop with provider escalation ---
    analysis = ""
    stopped_reason = "cap"
    resolving_provider = cascade[pi]
    prev_n_relevant = -1   # futility: relevant-evidence count last round (to detect "no new evidence")
    stale_rounds = 0       # consecutive rounds that added no NEW relevant evidence
    while True:
        t = time.time()
        decision = _synthesise(claim.text, evidence_pool, past_queries, model,
                               image_context=image_context)
        llm_calls += 1
        next_q = decision["next_query"]
        nudged = False
        if next_q and _is_redundant(next_q, past_queries):  # nudge once for a different angle
            decision = _synthesise(claim.text, evidence_pool, past_queries, model, nudge=True,
                                   image_context=image_context)
            llm_calls += 1
            next_q = decision["next_query"]
            nudged = True
        analysis = decision["analysis"]
        if rounds:
            rounds[-1]["synthesis"] = {"analysis": analysis, "next_query": next_q,
                                       "nudged": nudged, "elapsed_s": round(time.time() - t, 3)}

        # Hard corroboration gate (w/ Daniel, clog 260626): never conclude on <=1 relevant source
        # while a broader provider (Exa) is still untried — escalate instead. The 2-source "consider
        # Exa unless definitive" case is left to the cascade-aware synthesis prompt (the LLM emits a
        # next_query when it wants more).
        n_relevant = sum(1 for d in evidence_pool if d.relevant)
        exa_untried = pi < len(cascade) - 1
        force_escalate = (not next_q) and n_relevant <= 1 and exa_untried

        # Futility tracking: a round that adds NO new relevant evidence is "stale". Sustained
        # staleness means we're re-asking an unanswerable question — e.g. searching for confirmation
        # of a fabricated event that doesn't exist (the 4-round tail is ~84% false claims that already
        # land at flag). The ABSENCE of evidence is itself the finding, so don't keep paying to re-ask.
        if futility_stale:
            stale_rounds = stale_rounds + 1 if n_relevant <= prev_n_relevant else 0
            prev_n_relevant = n_relevant
        futile = bool(futility_stale) and stale_rounds >= futility_stale

        if not next_q and not force_escalate:
            stopped_reason = "confident"
            resolving_provider = cascade[pi]
            break

        # Futile + the neural engine (Exa) already tried (or none left) -> conclude on what we have.
        # If Exa is still untried, fall through: the escalation below gives it one shot first.
        if futile and next_q and not exa_untried:
            stopped_reason = "futile"
            break

        still_redundant = (next_q is not None) and _is_redundant(next_q, past_queries)
        # Escalate to the next provider when forced by the corroboration gate, stuck (still redundant
        # after a nudge), evidence has gone stale (futility -> give Exa one shot before concluding),
        # or this provider's round budget is spent without reaching confidence.
        if force_escalate or still_redundant or (futile and exa_untried) or rounds_this_provider >= rpp[pi]:
            if pi < len(cascade) - 1:
                pi += 1
                rounds_this_provider = 0
                providers_used.append(cascade[pi])
                if verbose:
                    print(f"  -> escalate to {cascade[pi]}"
                          + (" (corroboration: <=1 source)" if force_escalate else ""))
                # Re-tailor the query for a neural provider (Exa): it surfaces harder-to-find
                # sources, rewards precision, and — being a different engine — may benefit from
                # reusing/combining prior queries (the base 'do not repeat' rule is relaxed).
                if cascade[pi] == "exa":
                    esc = _synthesise(claim.text, evidence_pool, past_queries, model,
                                      escalate_to="exa", image_context=image_context)
                    llm_calls += 1
                    # Compose the STRONGEST single Exa query programmatically (the model alone tends to
                    # emit a short keyword query): the claim's core proposition + every distinct angle
                    # already tried + the model's new sub-question. Exa is neural and rewards a long,
                    # content-rich query, so we don't limit Exa to one focused gap-query.
                    next_q = _compose_exa_query(claim.text, past_queries, esc.get("next_query"))
                    if rounds and rounds[-1].get("synthesis"):
                        rounds[-1]["synthesis"]["escalated_query"] = next_q
            elif still_redundant:
                stopped_reason = "redundant"
                break

        if len(past_queries) >= max_rounds:
            stopped_reason = "cap"
            break
        next_q = _clean_query(next_q, claim.text)  # junk/empty (incl. forced-escalate null) -> claim
        past_queries.append(next_q)
        if verbose:
            print(f"[round {len(past_queries)}/{max_rounds}] [{cascade[pi]}] query: {next_q}")
        _consume(next_q, cascade[pi], *_search(next_q, cascade[pi]))
        rounds_this_provider += 1
        resolving_provider = cascade[pi]

    # --- Evaluation: Likert (ours) + 4-class (ClaimCheck), parallel on the same analysis ---
    t = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_likert = pool.submit(_evaluate_likert, claim.text, analysis, model,
                               image_context=image_context)
        f_4class = pool.submit(_evaluate_4class, claim.text, analysis, model,
                               image_context=image_context)
        likert = f_likert.result()
        four = f_4class.result()
    eval_s = round(time.time() - t, 3)
    llm_calls += 2

    # --- Misinformation evaluator: CONTEXT-AWARE, misinfo-native (design: verdict_nudge_design.md).
    # Runs only when a post/article context is supplied (article-extracted claims); the context-free
    # Likert above stays the cacheable backbone. One misinfo flag = the veracity<=3 cut, plus a type. ---
    post_flag = None
    post_flag_reason = ""
    misinfo_out = None
    if post_context:
        misinfo_out = _evaluate_misinfo(post_context, claim.text, analysis, model,
                                        image_context=image_context)
        post_flag = misinfo_out["misinfo"]
        post_flag_reason = f'{misinfo_out["misinfo_type"]}: {misinfo_out["justification"]}'
        llm_calls += 1

    total_s = round(time.time() - t_start, 2)
    if trace is not None:
        trace.update({
            "claim_text": claim.text,
            "cascade": cascade,
            "providers_used": providers_used,
            "resolving_provider": resolving_provider,
            "model": model,
            "date_ceiling": date_ceiling,
            "exclude_domains": sorted(exclude_domains) if exclude_domains else [],
            "image_context": image_context,
            "rounds": rounds,
            "final_analysis": analysis,
            "likert": {**{k: likert[k] for k in
                          ("veracity", "evidence_sufficiency", "evidence_agreement", "source_reliability")},
                       "justification": likert["justification"]},
            "fourclass": {"verdict": four["verdict"], "justification": four["justification"]},
            "nudge": {"flag": post_flag, "reason": post_flag_reason,
                      "post_used": post_context is not None,
                      "misinfo_type": (misinfo_out or {}).get("misinfo_type"),
                      "veracity_in_context": (misinfo_out or {}).get("veracity")},
            "eval_elapsed_s": eval_s,
            "stopped_reason": stopped_reason,
            "rounds_used": len(past_queries),
            "past_queries": past_queries,
            "evidence_urls": [d.url for d in evidence_pool],
            "funnel": dict(counters),
            "llm_calls": llm_calls,
            "total_elapsed_s": total_s,
        })

    return ClaimVerdict(
        claim=claim,
        scores=VerdictScores(
            veracity=likert["veracity"],
            evidence_sufficiency=likert["evidence_sufficiency"],
            evidence_agreement=likert["evidence_agreement"],
            source_reliability=likert["source_reliability"],
        ),
        tier_resolved=3,
        cap_hit=(stopped_reason == "cap"),
        redundant_exit=(stopped_reason == "redundant"),
        stopped_reason=stopped_reason,
        resolving_provider=resolving_provider,
        providers_used=providers_used,
        evidence_urls=[d.url for d in evidence_pool],
        justification=likert["justification"],
        rounds_used=len(past_queries),
        past_queries=past_queries,
        analysis=analysis,
        n_urls_seen=counters["n_urls_seen"],
        n_scraped=counters["n_scraped"],
        n_blocked_or_failed=counters["n_blocked_or_failed"],
        n_snippet_used=counters["n_snippet_used"],
        n_irrelevant=counters["n_irrelevant"],
        n_search_errors=counters["n_search_errors"],
        elapsed_seconds=total_s,
        llm_calls=llm_calls,
        verdict_4class=four["verdict"],
        verdict_4class_justification=four["justification"],
        post_flag=post_flag,
        post_flag_reason=post_flag_reason,
    )


# =============================================================================
# LLM helper
# =============================================================================

def _chat(messages: list[dict], max_tokens: int, model: str, temperature: float = 0.1,
          reasoning_effort: str | None = None) -> str:
    thinking = reasoning_effort not in (None, "none")  # "none" disables thinking
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    # DeepSeek V4 Flash on DeepInfra misroutes the JSON answer into reasoning_content when
    # response_format=json_object is combined with thinking (content comes back empty). So use
    # json mode only when NOT thinking; otherwise rely on the prompt's "output strict JSON"
    # instruction + the parser (which already extracts a prose-wrapped object).
    if not thinking:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = _client.chat.completions.create(**kwargs)
    _record_usage(resp.usage)
    msg = resp.choices[0].message
    content = msg.content or ""
    if not content and thinking:  # answer can land in reasoning_content under thinking
        content = getattr(msg, "reasoning_content", None) or ""
    return content


# =============================================================================
# Module 1 — Planning
# =============================================================================

def _clean_query(q: str | None, fallback: str | None) -> str | None:
    """Guard against a junk/empty LLM query (e.g. a bare ':' that returns nothing) — fall back to
    the claim text so a malformed query never wastes a round (clog 260626)."""
    q = (q or "").strip().strip(":;,.").strip()
    return q if len(q) >= 5 and any(c.isalnum() for c in q) else fallback


_MONTH_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%m/%d/%Y", "%Y/%m/%d",
                  "%b %Y", "%B %Y", "%Y-%m", "%Y")


def _parse_iso(s: str | None) -> str | None:
    """Normalize a publication-date string (ISO, 'Oct 2, 2025', '2025-12-05', 'May 2026', …) to ISO
    YYYY-MM-DD so the date_ceiling compares correctly. None if unparseable (caller keeps the doc — we
    never drop on a date we can't read). Fixes the mixed-format string-compare bug (clog 260626)."""
    if not s:
        return None
    s = s.strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(0)
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _plan_query(claim_text: str, model: str, image_context: str | None = None) -> str:
    try:
        raw = _chat(vp.build_plan_messages(claim_text, image_context), _PLAN_MAX_TOKENS, model,
                    temperature=0.2)
    except Exception as e:
        print(f"[plan] error: {e}")
        return claim_text
    return vp.parse_plan(raw, fallback=claim_text)


# =============================================================================
# Modules 2+3 — Execution + Summarisation
# =============================================================================

def _summarise_one(claim_text: str, url: str, doc_text: str, model: str) -> dict | None:
    """LLM summary of a scraped page. Cached on (url, claim, model). None on error."""
    cache_key = disk_cache.make_key(url, claim_text, model)
    cached = disk_cache.get("summarise", cache_key)
    if cached is not None:
        return cached
    try:
        raw = _chat(
            vp.build_summarise_messages(claim_text, url, doc_text, _DOC_TRUNCATE_CHARS),
            _SUMMARISE_MAX_TOKENS, model,
        )
        data = vp.parse_summarise(raw)
        disk_cache.set_("summarise", cache_key, data)
        return data
    except Exception as e:
        print(f"[summarise] error for {url}: {e}")
        return None  # don't cache errors


def _gather_evidence(results: list[dict], claim_text: str, model: str) -> tuple[list[EvidenceDoc], dict]:
    """Get full page text per URL, summarise it, and build evidence. Snippet ALWAYS kept.

    Full text comes from the provider's `content` (Tavily raw_content) when present, else
    from scraping (Serper). A page with full text gets summary+quotes+date (relevant ones);
    no full text keeps its snippet only; no text and no snippet is dropped.
    """
    stats = {"n_scraped": 0, "n_blocked_or_failed": 0, "n_snippet_used": 0,
             "n_irrelevant": 0, "llm_calls": 0}
    if not results:
        return [], stats

    # 1. Full page text: use provider-supplied content, else scrape (in parallel).
    texts: dict[str, str] = {r["url"]: r["content"] for r in results if r.get("content")}
    to_scrape = [r["url"] for r in results if not r.get("content")]
    if to_scrape:
        with ThreadPoolExecutor(max_workers=len(to_scrape)) as pool:
            futures = {pool.submit(search.scrape, u): u for u in to_scrape}
            for fut in as_completed(futures):
                text = fut.result()
                if text:
                    texts[futures[fut]] = text

    # 2. Parallel summarise of pages that have full text.
    summaries: dict[str, dict | None] = {}
    if texts:
        with ThreadPoolExecutor(max_workers=len(texts)) as pool:
            futs = {pool.submit(_summarise_one, claim_text, u, t, model): u
                    for u, t in texts.items()}
            for fut in as_completed(futs):
                summaries[futs[fut]] = fut.result()
                stats["llm_calls"] += 1

    # 3. Build evidence docs — snippet always kept.
    docs: list[EvidenceDoc] = []
    for r in results:
        u = r["url"]
        prov = r.get("provider", "")
        snippet = (r.get("snippet") or "").strip()
        idx_date = r.get("date")
        if u in texts:
            s = summaries.get(u)
            if s is None:
                # Summariser call FAILED (e.g. a transient 429) — this is NOT a relevance
                # signal. Fall back to the snippet so the evidence isn't silently dropped AND
                # mislabelled irrelevant under load (clog/150626 latent bug).
                if snippet:
                    stats["n_snippet_used"] += 1
                    docs.append(EvidenceDoc(
                        url=u, provider=prov, snippet=snippet, scraped=False, relevant=True,
                        publication_date=idx_date,
                    ))
                else:
                    stats["n_blocked_or_failed"] += 1
            elif s["relevant"]:
                stats["n_scraped"] += 1
                docs.append(EvidenceDoc(
                    url=u, provider=prov, snippet=snippet, scraped=True, relevant=True,
                    publication_date=s["publication_date"] or idx_date,
                    summary=s["summary"], quotes=s["quotes"],
                ))
            else:
                stats["n_scraped"] += 1
                stats["n_irrelevant"] += 1
                docs.append(EvidenceDoc(
                    url=u, provider=prov, snippet=snippet, scraped=True, relevant=False,
                    publication_date=idx_date,
                ))
        elif snippet:
            stats["n_snippet_used"] += 1
            docs.append(EvidenceDoc(
                url=u, provider=prov, snippet=snippet, scraped=False, relevant=True,
                publication_date=idx_date,
            ))
        else:
            stats["n_blocked_or_failed"] += 1
    return docs, stats


# =============================================================================
# Module 4 — Synthesis
# =============================================================================

def _synthesise(claim_text: str, pool: list[EvidenceDoc], past_queries: list[str],
                model: str, nudge: bool = False, escalate_to: str | None = None,
                image_context: str | None = None) -> dict:
    try:
        raw = _chat(
            vp.build_synthesise_messages(claim_text, _format_evidence(pool), past_queries, nudge,
                                         escalate_to=escalate_to, image_context=image_context),
            _SYNTHESISE_MAX_TOKENS, model,
        )
        return vp.parse_synthesise(raw)
    except Exception as e:
        print(f"[synthesise] error: {e}")
        return {"analysis": "Synthesis failed; treating as no usable evidence.", "next_query": None}


# =============================================================================
# Module 5 — Evaluation
# =============================================================================

def _evaluate_likert(claim_text: str, analysis: str, model: str, temperature: float = 0.1,
                     image_context: str | None = None) -> dict:
    try:
        raw = _chat(vp.build_likert_messages(claim_text, analysis, image_context),
                    _EVALUATE_MAX_TOKENS, model, temperature=temperature)
        return vp.parse_likert(raw)
    except Exception as e:
        print(f"[evaluate_likert] error: {e}")
        return {"veracity": 3, "evidence_sufficiency": 1, "evidence_agreement": 1,
                "source_reliability": 1, "justification": f"eval error: {e}"}


def _evaluate_4class(claim_text: str, analysis: str, model: str,
                     image_context: str | None = None) -> dict:
    try:
        raw = _chat(vp.build_fourclass_messages(claim_text, analysis, image_context),
                    _EVALUATE_MAX_TOKENS, model)
        return vp.parse_fourclass(raw)
    except Exception as e:
        print(f"[evaluate_4class] error: {e}")
        return {"verdict": "Not Enough Evidence", "justification": f"eval error: {e}"}


def _evaluate_misinfo(context: str, claim_text: str, analysis: str, model: str,
                      temperature: float = 0.1, image_context: str | None = None) -> dict:
    """Misinformation-native, CONTEXT-AWARE evaluator (for article/post-extracted claims).
    Reuses the validated veracity anchoring; the single misinfo flag is the veracity<=3 cut."""
    try:
        raw = _chat(vp.build_misinfo_messages(claim_text, context, analysis, image_context),
                    _EVALUATE_MAX_TOKENS, model, temperature=temperature)
        return vp.parse_misinfo(raw)
    except Exception as e:
        print(f"[evaluate_misinfo] error: {e}")
        return {"veracity": 3, "misinfo": True, "misinfo_type": "UNSUPPORTED",
                "justification": f"eval error: {e}"}


# =============================================================================
# Helpers
# =============================================================================

def _compose_exa_query(claim_text: str, past_queries: list[str], model_query: str | None) -> str:
    """Build the strongest single Exa (neural) query: the claim's core proposition + every distinct
    angle already tried (the prior queries) + the model's new sub-question, joined into one rich
    semantic query. Exa searches by meaning and rewards a long, descriptive, content-rich query, so
    on escalation we compose rather than limit ourselves to a single focused gap-query."""
    parts: list[str] = []
    seen: set[str] = set()
    for q in [claim_text, *(past_queries or []), model_query or ""]:
        q = (q or "").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            parts.append(q)
    return "  ".join(parts)


def _is_redundant(query: str, past_queries: list[str]) -> bool:
    """Cosine similarity >= REDUNDANCY_THRESHOLD against any past query (semantic dedup)."""
    if not past_queries:
        return False
    q_vec = embed(query)
    return any(float(np.dot(q_vec, embed(p))) >= REDUNDANCY_THRESHOLD for p in past_queries)


def _format_evidence(pool: list[EvidenceDoc]) -> str:
    blocks: list[str] = []
    n = 0
    for d in pool:
        body: list[str] = []
        if d.summary:
            body.append(f"    summary: {d.summary}")
        if d.quotes:
            body.append("    quotes: " + " | ".join(f'"{q}"' for q in d.quotes))
        # Snippet is a fallback ONLY for a relevant doc we couldn't scrape. A scraped doc is
        # represented by its richer summary+quotes; a scraped-but-irrelevant doc contributes
        # nothing and is omitted entirely (rather than fed to synthesis as a bare-URL line).
        if d.relevant and not d.scraped and d.snippet:
            body.append(f"    snippet: {d.snippet}")
        if not body:
            continue
        n += 1
        date = d.publication_date or "unknown date"
        blocks.append("\n".join([f"[{n}] {d.url} (date: {date})", *body]))
    return "\n".join(blocks)
