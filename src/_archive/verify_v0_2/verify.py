"""Tier 3: ClaimCheck-style verification loop.

Five modules, one LLM throughout. Per claim:

  1. Planning    — produce ONE opening search query.
  2. Execution   — SearXNG top-3 (no LLM).
  3. Summarisation — per scraped URL, LLM call producing claim-relevant summary.
  4. Synthesis   — analyse pool; emit verdict-ready analysis OR next query.
  5. Evaluation  — assign 4-dimension Likert scores from the analysis.

Loop is between Synthesis and Execution. Hard cap MAX_ROUNDS. At cap, Synthesis
is forced to produce an analysis (no more queries) and we move to Evaluation.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI

from config import VERIFICATION_API_KEY, VERIFICATION_BASE_URL, VERIFICATION_MODEL
from pipeline import disk_cache, search
from pipeline.config import MAX_ROUNDS, REDUNDANCY_THRESHOLD, SEARCH_TOP_K
from pipeline.embedding import embed
from pipeline.models import AtomicClaim, ClaimVerdict, EvidenceDoc, VerdictScores


_client = OpenAI(base_url=VERIFICATION_BASE_URL, api_key=VERIFICATION_API_KEY)

# --- Token caps (sized to accommodate reasoning preamble on gpt-oss-120b) ---
_PLAN_MAX_TOKENS = 1000
_SUMMARISE_MAX_TOKENS = 2000
_SYNTHESISE_MAX_TOKENS = 4000
_EVALUATE_MAX_TOKENS = 2000

_DOC_TRUNCATE_CHARS = 12000


# =============================================================================
# Public entry point
# =============================================================================

def verify(
    claim: AtomicClaim,
    date_ceiling: str | None = None,
    exclude_urls: list[str] | None = None,
    verbose: bool = False,
    eval_sees_raw: bool = False,
) -> ClaimVerdict:
    """Run the Tier 3 ClaimCheck loop. Always returns a verdict.

    Production usage: leave date_ceiling and exclude_urls unset.

    Eval usage (AVeriTeC): pass date_ceiling=YYYY-MM-DD to post-filter docs
    published after the claim date; pass exclude_urls=[fact_check_url] to drop
    the gold fact-check article by exact match. Post-filter is lenient on
    undated docs (kept) to avoid sacrificing recall to date-extraction misses.
    """
    t_start = time.time()
    exclude_urls = list(exclude_urls or [])
    n_post_filtered_by_date = 0
    n_excluded_by_url = 0

    evidence_pool: list[EvidenceDoc] = []
    past_queries: list[str] = []

    # Diagnostics counters
    n_urls_seen = 0
    n_blocked_or_failed = 0
    n_snippet_used = 0
    n_irrelevant = 0
    llm_calls = 0

    def _consume(results: list[dict]) -> None:
        """Run gather_evidence, update counters, extend pool."""
        nonlocal n_urls_seen, n_blocked_or_failed, n_snippet_used, n_irrelevant, llm_calls
        nonlocal n_excluded_by_url, n_post_filtered_by_date

        # Pre-filter: drop excluded URLs (eval-time fact-check article)
        pre_excluded = [r for r in results if r["url"] in exclude_urls]
        n_excluded_by_url += len(pre_excluded)
        results = [r for r in results if r["url"] not in exclude_urls]

        n_urls_seen += len(results) + len(pre_excluded)
        new_docs, stats = _gather_evidence(results, claim.text)
        n_blocked_or_failed += stats["n_blocked_or_failed"]
        n_snippet_used += stats["n_snippet_used"]
        n_irrelevant += stats["n_irrelevant"]
        llm_calls += stats["llm_calls"]

        # Post-filter: drop docs whose extracted publication_date is after the
        # claim's date_ceiling. Lenient on missing dates (kept).
        if date_ceiling:
            kept_docs = []
            for d in new_docs:
                if d.publication_date and d.publication_date > date_ceiling:
                    n_post_filtered_by_date += 1
                else:
                    kept_docs.append(d)
            new_docs = kept_docs

        evidence_pool.extend(new_docs)
        if verbose:
            print(f"  pre-excluded: {len(pre_excluded)}, "
                  f"scraped: {stats['n_scraped']}/{len(results)}, "
                  f"snippet-fallback: {stats['n_snippet_used']}, "
                  f"summarised relevant: {len(new_docs) + (n_post_filtered_by_date - 0)}, "
                  f"date-filtered: {n_post_filtered_by_date}, "
                  f"irrelevant: {stats['n_irrelevant']}")

    # --- Planning: opening query ---
    q = _plan_initial_query(claim.text)
    llm_calls += 1
    past_queries.append(q)
    if verbose:
        print(f"[round 1/{MAX_ROUNDS}] query: {q}")

    results = search.search(q, SEARCH_TOP_K)
    if verbose:
        print(f"[round 1/{MAX_ROUNDS}] urls: {len(results)} — " + " ".join(r["url"] for r in results))
    _consume(results)

    # --- Synthesis loop ---
    analysis: str | None = None
    redundant_exit = False
    for round_idx in range(MAX_ROUNDS):
        is_last = (round_idx == MAX_ROUNDS - 1)
        decision = _synthesise(
            claim.text,
            evidence_pool,
            past_queries,
            round_num=round_idx + 1,
            max_rounds=MAX_ROUNDS,
            force_evaluate=is_last,
        )
        llm_calls += 1
        if verbose:
            extra = f" → {decision.get('next_query')!r}" if decision.get("action") == "search" else ""
            print(f"[round {round_idx + 1}/{MAX_ROUNDS}] synthesis: {decision.get('action')}{extra}")

        if decision["action"] == "evaluate":
            analysis = decision["analysis"]
            break

        next_q = decision["next_query"]
        # Only abort on redundancy when we already have evidence — if the pool is
        # empty, the model can't differentiate its follow-up; let it keep trying
        # (search results vary on wording, and the cap will stop us eventually).
        if evidence_pool and _is_redundant(next_q, past_queries):
            if verbose:
                print(f"[round {round_idx + 1}/{MAX_ROUNDS}] redundant query — forcing evaluation")
            forced = _synthesise(
                claim.text,
                evidence_pool,
                past_queries,
                round_num=round_idx + 1,
                max_rounds=MAX_ROUNDS,
                force_evaluate=True,
            )
            llm_calls += 1
            analysis = forced["analysis"]
            redundant_exit = True
            break

        past_queries.append(next_q)
        next_round = round_idx + 2
        if verbose:
            print(f"[round {next_round}/{MAX_ROUNDS}] query: {next_q}")
        results = search.search(next_q, SEARCH_TOP_K)
        if verbose:
            print(f"[round {next_round}/{MAX_ROUNDS}] urls: {len(results)} — " + " ".join(r["url"] for r in results))
        _consume(results)

    assert analysis is not None, "synthesis loop must produce an analysis"

    # --- Evaluation: Likert (ours) and 4-class (ClaimCheck-style) on the same analysis ---
    cap_hit = (len(past_queries) == MAX_ROUNDS)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_likert = pool.submit(
            _evaluate, claim.text, analysis,
            evidence_pool if eval_sees_raw else None,
        )
        f_4class = pool.submit(_evaluate_4class, claim.text, analysis)
        eval_result = f_likert.result()
        eval_4class = f_4class.result()
    llm_calls += 2

    return ClaimVerdict(
        claim=claim,
        scores=VerdictScores(
            veracity=eval_result["veracity"],
            evidence_coverage=eval_result["evidence_coverage"],
            evidence_consistency=eval_result["evidence_consistency"],
            source_quality=eval_result["source_quality"],
        ),
        tier_resolved=3,
        cap_hit=cap_hit,
        redundant_exit=redundant_exit,
        evidence_urls=[d.url for d in evidence_pool],
        justification=eval_result["justification"],
        rounds_used=len(past_queries),
        past_queries=past_queries,
        analysis=analysis,
        n_urls_seen=n_urls_seen,
        n_blocked_or_failed=n_blocked_or_failed,
        n_snippet_used=n_snippet_used,
        n_irrelevant=n_irrelevant,
        elapsed_seconds=round(time.time() - t_start, 2),
        llm_calls=llm_calls,
        verdict_4class=eval_4class["verdict"],
        verdict_4class_justification=eval_4class["justification"],
    )


# =============================================================================
# Module 1 — Planning
# =============================================================================

_PLAN_SYSTEM = """You are a fact-checker preparing to verify a claim using web search. Your only job is to produce ONE Google search query that would best surface evidence about the claim.

Guidelines:
- Use specific named entities, dates, numbers, and exact phrases from the claim.
- Prefer terms a journalist or institution would use, not informal phrasing.
- Aim for the most direct evidence that could either confirm or contradict the claim.
- Do not use search operators (site:, intitle:, quotes, AND/OR). Plain text only.

Output ONLY the query, on a single line. No explanation, no JSON, no quotes."""


def _plan_initial_query(claim_text: str) -> str:
    resp = _client.chat.completions.create(
        model=VERIFICATION_MODEL,
        messages=[
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": f"Claim: {claim_text}\n\nProduce the opening search query."},
        ],
        temperature=0.2,
        max_tokens=_PLAN_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Some reasoning models prefix with a "Final answer:" line; take the last non-empty line.
    lines = [ln.strip().strip('"').strip("'") for ln in raw.splitlines() if ln.strip()]
    query = lines[-1] if lines else ""
    return query or claim_text


# =============================================================================
# Module 3 — Summarisation (per URL)
# =============================================================================

_SUMMARISE_SYSTEM = """You are a fact-checker reading ONE source article to assess a specific claim. Decide whether the article speaks to the claim, and if so, write a short summary of WHAT THE ARTICLE SAYS — using only the article's text, never your own background knowledge.

Output JSON:
{
  "relevant": true | false,
  "publication_date": "YYYY-MM-DD" or null,
  "summary": "1-3 sentences. Begin with 'According to this article (dated <date>):' if a date is known. State only what the article asserts about the claim. Include numbers, names, dates, direct quotes when present."
}

Rules:
- relevant=false if the article is off-topic, a paywall stub, an error page, a navigation page, or otherwise unhelpful for fact-checking THIS claim.
- Do NOT take a position on whether the article supports or refutes the claim — only state what the article says.
- Do NOT cite outside knowledge. The summary must be attributable to the article text alone."""


def _summarise_for_claim(
    claim_text: str, url: str, doc_text: str, source: str
) -> EvidenceDoc:
    cache_key = disk_cache.make_key(url, claim_text, source)
    cached = disk_cache.get("summarise", cache_key)
    if cached is not None:
        return EvidenceDoc(**cached)

    body_label = "Article text" if source == "scrape" else "Google search snippet (article body unavailable)"
    user = (
        f"Claim: {claim_text}\n\n"
        f"Article URL: {url}\n"
        f"{body_label} (truncated to {_DOC_TRUNCATE_CHARS} chars):\n"
        f"{doc_text[:_DOC_TRUNCATE_CHARS]}"
    )
    try:
        resp = _client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=_SUMMARISE_MAX_TOKENS,
        )
        data = _parse_json(resp.choices[0].message.content or "")
        doc = EvidenceDoc(
            url=url,
            publication_date=data.get("publication_date") or None,
            summary=str(data.get("summary", "")).strip(),
            relevant=bool(data.get("relevant", False)),
            source=source,  # type: ignore[arg-type]
        )
        disk_cache.set_("summarise", cache_key, doc.model_dump())
        return doc
    except Exception as e:
        print(f"[summarise] error for {url}: {e}")
        # Don't cache errors — let the next run retry.
        return EvidenceDoc(url=url, publication_date=None, summary="", relevant=False, source=source)  # type: ignore[arg-type]


# Minimum snippet length to bother summarising (search-engine snippets are typically 100-300 chars).
_MIN_SNIPPET_CHARS = 40


def _gather_evidence(results: list[dict], claim_text: str) -> tuple[list[EvidenceDoc], dict]:
    """Parallel scrape + summarise. When scrape fails, fall back to the search snippet.

    `results` is the search output: list of {"url": str, "snippet": str}.

    Returns (kept_docs, stats) with stats:
      n_scraped: full-page scrapes that succeeded
      n_snippet_used: URLs where snippet was used in place of a failed scrape
      n_blocked_or_failed: URLs with no usable content at all (scrape failed AND no usable snippet)
      n_irrelevant: summarised docs marked relevant=False
      llm_calls: summarisation LLM calls
    """
    stats = {"n_scraped": 0, "n_snippet_used": 0, "n_blocked_or_failed": 0,
             "n_irrelevant": 0, "llm_calls": 0}
    if not results:
        return [], stats

    # 1. Parallel scrape
    scraped: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(results)) as pool:
        futures = {pool.submit(search.scrape, r["url"]): r["url"] for r in results}
        for fut in as_completed(futures):
            u = futures[fut]
            text = fut.result()
            if text:
                scraped[u] = text
    stats["n_scraped"] = len(scraped)

    # 2. Build summariser inputs. For each result: prefer full scrape; otherwise
    # fall back to the search snippet if it's long enough to carry any signal.
    to_summarise: list[tuple[str, str, str]] = []  # (url, text, source)
    for r in results:
        u = r["url"]
        if u in scraped:
            to_summarise.append((u, scraped[u], "scrape"))
        else:
            snippet = (r.get("snippet") or "").strip()
            if len(snippet) >= _MIN_SNIPPET_CHARS:
                to_summarise.append((u, snippet, "snippet"))
                stats["n_snippet_used"] += 1
            else:
                stats["n_blocked_or_failed"] += 1

    if not to_summarise:
        return [], stats

    # 3. Parallel summarise
    docs: list[EvidenceDoc] = []
    with ThreadPoolExecutor(max_workers=len(to_summarise)) as pool:
        futures = [
            pool.submit(_summarise_for_claim, claim_text, u, t, src)
            for (u, t, src) in to_summarise
        ]
        for fut in as_completed(futures):
            doc = fut.result()
            stats["llm_calls"] += 1
            if doc.relevant and doc.summary:
                docs.append(doc)
            else:
                stats["n_irrelevant"] += 1
    return docs, stats


# =============================================================================
# Module 4 — Synthesis
# =============================================================================

_SYNTHESISE_SYSTEM = """You are a fact-checker who has gathered evidence about a claim through one or more rounds of web search. Your job is to:

1. Synthesize the evidence into a coherent analysis: what does it collectively say about the claim? Note agreements, contradictions, gaps, and reliability of sources.
2. Decide: is the evidence sufficient for a confident assessment, OR is a specific sub-question unresolved AND likely to be answered by one more targeted query?

Output JSON. Choose ONE shape.

To proceed to evaluation:
{
  "action": "evaluate",
  "analysis": "<3-6 sentence synthesis>"
}

To request one more search:
{
  "action": "search",
  "analysis": "<3-6 sentence synthesis>",
  "next_query": "<a search query targeting the specific gap; must NOT paraphrase any prior query>"
}

Rules:
- Choose "search" only if you can name a specific unresolved sub-question AND a query that could plausibly find evidence for it.
- Choose "evaluate" if evidence is sufficient, OR if further search is unlikely to resolve the gap (e.g., the gap is one of judgement, not retrieval).
- The "analysis" field MUST be present in both shapes. It is the durable summary the evaluator will read.
- If the evidence pool is EMPTY (no relevant docs found in any prior round), your next_query MUST take a substantively different angle — different entities, different aspects, different terminology — NOT a paraphrase of any previous query. Light rewordings hit the same Google results and waste the search budget. Try a fiscal angle, a different actor, a broader concept, or a different time period."""

_SYNTHESISE_FORCE_SUFFIX = """

**THIS CALL: action MUST be "evaluate".** The "search" option is unavailable — the search budget is exhausted. Produce the analysis based on the evidence currently in the pool. If the pool is empty or unhelpful, the analysis should say so honestly."""


def _synthesise(
    claim_text: str,
    evidence_pool: list[EvidenceDoc],
    past_queries: list[str],
    round_num: int,
    max_rounds: int,
    force_evaluate: bool,
) -> dict:
    system = _SYNTHESISE_SYSTEM + (_SYNTHESISE_FORCE_SUFFIX if force_evaluate else "")
    user = _format_synthesis_user(claim_text, evidence_pool, past_queries, round_num, max_rounds)

    try:
        resp = _client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=_SYNTHESISE_MAX_TOKENS,
        )
        data = _parse_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"[synthesise] error: {e}")
        return {"action": "evaluate", "analysis": "Synthesis failed; treating as no usable evidence."}

    action = data.get("action")
    if force_evaluate or action != "search":
        return {"action": "evaluate", "analysis": str(data.get("analysis", "")).strip()}

    next_q = str(data.get("next_query", "")).strip()
    if not next_q:
        return {"action": "evaluate", "analysis": str(data.get("analysis", "")).strip()}
    return {
        "action": "search",
        "analysis": str(data.get("analysis", "")).strip(),
        "next_query": next_q,
    }


def _format_synthesis_user(
    claim_text: str,
    evidence_pool: list[EvidenceDoc],
    past_queries: list[str],
    round_num: int,
    max_rounds: int,
) -> str:
    return (
        f"Claim: {claim_text}\n\n"
        f"Evidence pool ({len(evidence_pool)} items):\n"
        f"{_format_evidence(evidence_pool) if evidence_pool else '(empty)'}\n\n"
        f"Previous queries (do not repeat):\n"
        f"{_format_queries(past_queries)}\n\n"
        f"Round {round_num} of {max_rounds}.\n\n"
        f"Decide: proceed to evaluation, or request one more search?"
    )


# =============================================================================
# Module 5 — Evaluation
# =============================================================================

_EVALUATE_SYSTEM = """You are a fact-checker assigning a structured assessment to a claim, based on a synthesized analysis of evidence prepared by an analyst. The analysis is the only context to use — do not add outside knowledge.

Score each of four dimensions on a 1-5 Likert scale. Each integer has a specific definition below. Score each dimension independently against its own definition. They are not mutually exclusive.

**Use the full scale, including 2 and 4 deliberately.** Partial, leaning, or qualified evidence belongs at 2 or 4 — do NOT collapse every judgement to 1, 3, or 5. The middle of the scale is the most informative range for typical fact-checks.

veracity — does the evidence in the analysis support the claim as true?
  5: Strong evidence the claim is true, including all load-bearing qualifiers (numbers, dates, named entities). Multiple credible sources confirm.
  4: Evidence leans toward the claim. Most sources support the main assertion, but a load-bearing qualifier may be partially off (e.g., approximate number, close-but-not-exact date).
  3: Mixed or indeterminate. Evidence neither clearly supports nor clearly refutes the claim's main assertion.
  2: Evidence leans against the claim. At least one credible source contradicts a load-bearing element.
  1: Strong evidence the claim is false. Multiple credible sources contradict.

evidence_coverage — does the analysis address the claim's substance?
  5: Multiple sources speak directly to the claim's main assertion AND its load-bearing qualifiers.
  4: Multiple sources directly address the claim, though not all of its qualifiers.
  3: At least one source touches the claim partially (e.g., covers the topic but not the specific qualifier).
  2: Sources are only topically adjacent; they don't speak to the specific assertion.
  1: No retrieved source addresses the claim's substance.

evidence_consistency — do the credible sources in the analysis agree?
  5: Strong unanimous agreement across THREE OR MORE credible sources, whether supporting OR refuting the claim.
  4: Sources broadly agree; only minor wording or scope differences.
  3: Some disagreement among sources, with one side clearly weaker in credibility. **ALSO use 3 if only ONE source is available** — single-source "agreement" is vacuous and cannot support a higher score.
  2: Notable disagreement, with at least one credible source on each side.
  1: Credible sources actively contradict each other on a load-bearing element.

source_quality — how reliable are the sources cited in the analysis?
  5: Primary sources (official records, peer-reviewed papers, named experts in their area, court/legislative documents) or top-tier institutional reporting.
  4: Multiple mainstream journalistic outlets; at least some institutional or named-expert voices.
  3: Mixed — some reputable outlets but corroboration is thin or relies on secondary reporting.
  2: Mostly blogs, opinion pieces, or low-traffic sites; few mainstream voices.
  1: Only weak, fringe, or anonymous sources; or no usable sources retrieved.

Output strict JSON, no prose outside the JSON:
{
  "veracity": <1-5>,
  "evidence_coverage": <1-5>,
  "evidence_consistency": <1-5>,
  "source_quality": <1-5>,
  "justification": "<one or two sentences citing what in the analysis drives the scores>"
}"""


def _evaluate(
    claim_text: str,
    analysis: str,
    evidence_pool: list[EvidenceDoc] | None = None,
) -> dict:
    """Returns {veracity, evidence_coverage, evidence_consistency, source_quality, justification}."""
    user_parts = [f"Claim: {claim_text}\n\nSynthesized analysis:\n{analysis}"]
    if evidence_pool:
        user_parts.append(
            "Raw evidence pool (for cross-reference; the analysis is your primary input):\n"
            + _format_evidence(evidence_pool)
        )
    user_parts.append("Assign the four scores.")
    user = "\n\n".join(user_parts)

    try:
        resp = _client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[
                {"role": "system", "content": _EVALUATE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=_EVALUATE_MAX_TOKENS,
        )
        data = _parse_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"[evaluate] error: {e}")
        return _default_eval("parse error")

    try:
        return {
            "veracity": _clamp_1_5(data["veracity"]),
            "evidence_coverage": _clamp_1_5(data["evidence_coverage"]),
            "evidence_consistency": _clamp_1_5(data["evidence_consistency"]),
            "source_quality": _clamp_1_5(data["source_quality"]),
            "justification": str(data.get("justification", "")).strip(),
        }
    except (KeyError, TypeError, ValueError) as e:
        print(f"[evaluate] malformed scores: {e}; raw={data!r}")
        return _default_eval("malformed scores")


def _default_eval(note: str) -> dict:
    return {
        "veracity": 3,
        "evidence_coverage": 1,
        "evidence_consistency": 1,
        "source_quality": 1,
        "justification": note,
    }


# =============================================================================
# Module 5b — ClaimCheck-style 4-class evaluator (parallel to the Likert one)
# =============================================================================
# Adapted verbatim from ClaimCheck (Devasier et al., arXiv 2510.01226), Listing 4 + verdict
# descriptions. Run alongside the Likert evaluator so we can compare both on the same analysis.

_EVALUATE_4CLASS_LABELS = ("Supported", "Refuted",
                          "Conflicting Evidence/Cherrypicking", "Not Enough Evidence")

_EVALUATE_4CLASS_SYSTEM = """Determine the Claim's veracity from the synthesized analysis, following these steps:

1. Briefly summarize the key insights from the fact-check in at most one paragraph.
2. Write one paragraph about which of the Decision Options applies best, and emit the chosen option at the end.

Decision Options:
Supported | Refuted | Conflicting Evidence/Cherrypicking | Not Enough Evidence

Rules:

Supported - The claim is directly and clearly backed by strong, credible evidence. Minor uncertainty or lack of detail does not disqualify a claim from being Supported if the main point is well-evidenced.
- Use Supported if the overall weight of evidence points to the claim being true, even if there are minor caveats or not every detail is confirmed.

Refuted - The claim is contradicted by strong, credible evidence, or is shown to be fabricated, deceptive, or false in its main point.
- Use Refuted if the central elements of the claim are disproven, even if some minor details are unclear.
- Lack of any credible sources supporting the claim does NOT mean "Not Enough Evidence" — it means the claim is Refuted.

Conflicting Evidence/Cherrypicking - Only use this if there are reputable sources that directly and irreconcilably contradict each other about the main point of the claim, and no clear resolution is possible after careful analysis.
- Do NOT use this for minor disagreements, incomplete evidence, or if most evidence points one way but a few sources disagree.

Not Enough Evidence - Only use this if there is genuinely no relevant evidence available after a thorough search, AND the claim is too vague or ambiguous to evaluate.
- Do NOT use this if there is some evidence, even if it is weak, or if the claim is mostly clear but not every detail is confirmed.
- This is a last-resort option only.

Output JSON, no prose outside the JSON:
{
  "verdict": "<one of: Supported | Refuted | Conflicting Evidence/Cherrypicking | Not Enough Evidence>",
  "justification": "<the one-paragraph decision rationale from step 2>"
}"""


def _evaluate_4class(claim_text: str, analysis: str) -> dict:
    """Returns {verdict, justification}. Verdict is one of the 4 AVeriTeC classes."""
    user = (
        f"Claim: {claim_text}\n\n"
        f"Synthesized analysis:\n{analysis}\n\n"
        f"Assign one of the four decision options."
    )
    try:
        resp = _client.chat.completions.create(
            model=VERIFICATION_MODEL,
            messages=[
                {"role": "system", "content": _EVALUATE_4CLASS_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=_EVALUATE_MAX_TOKENS,
        )
        data = _parse_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"[evaluate_4class] error: {e}")
        return {"verdict": "Not Enough Evidence", "justification": f"eval error: {e}"}

    verdict = str(data.get("verdict", "")).strip()
    # Lenient normalisation in case model returns e.g. "Conflicting Evidence" without the slash.
    if verdict not in _EVALUATE_4CLASS_LABELS:
        low = verdict.lower()
        if "support" in low:
            verdict = "Supported"
        elif "refut" in low or "false" in low:
            verdict = "Refuted"
        elif "conflict" in low or "cherry" in low:
            verdict = "Conflicting Evidence/Cherrypicking"
        else:
            verdict = "Not Enough Evidence"
    return {"verdict": verdict, "justification": str(data.get("justification", "")).strip()}


# =============================================================================
# Redundancy check
# =============================================================================

def _is_redundant(query: str, past_queries: list[str]) -> bool:
    """Cosine similarity ≥ REDUNDANCY_THRESHOLD against any past query."""
    if not past_queries:
        return False
    q_vec = embed(query)
    for past in past_queries:
        sim = float(np.dot(q_vec, embed(past)))
        if sim >= REDUNDANCY_THRESHOLD:
            return True
    return False


# =============================================================================
# Helpers
# =============================================================================

def _format_evidence(pool: list[EvidenceDoc]) -> str:
    blocks: list[str] = []
    for i, d in enumerate(pool, 1):
        date = d.publication_date or "unknown date"
        blocks.append(f"[{i}] {d.url}  (date: {date})\n    {d.summary}")
    return "\n".join(blocks)


def _format_queries(queries: list[str]) -> str:
    if not queries:
        return "(none)"
    return "\n".join(f"{i}. {q}" for i, q in enumerate(queries, 1))


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Some reasoning models wrap JSON in prose; grab the outermost object.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def _clamp_1_5(v) -> int:
    return max(1, min(5, int(v)))
