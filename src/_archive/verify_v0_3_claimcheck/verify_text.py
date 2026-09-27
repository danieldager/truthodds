"""EXPERIMENTAL raw-text verification loop (Option A, design: docs/verdict_nudge_design.md).

Verifies RAW TEXT (headline / tweet / post, spin intact) instead of a normalized atomic claim.
Claim decomposition is IMPLICIT: a controller LLM, seeing the cumulative state (raw text, every
query + provider, all snippets + full-read summaries, its own running analysis + open questions),
picks the next action each round:
  VERDICT  -> decide now (snippets settle it, OR no credible evidence for a main claim -> flag)
  READ     -> scrape + summarise specific promising results before deciding (lazy scrape)
  SEARCH   -> new query; the model picks serper OR escalates to exa itself (serper is forced first)
Round cap is high (default 10) to observe empirically how many rounds raw claims actually need.
Reuses search / scrape / summarise / credibility primitives. No cache, no date logic beyond ceiling.
"""
from __future__ import annotations
import time

from pipeline import search, credibility, verify_prompts as vp
from pipeline.verify import _chat, VERIFICATION_MODEL
from pipeline.config import SEARCH_RETRIEVE_K, FACT_CHECK_DOMAINS
from config import REPAIR_MODEL


def _is_factcheck(url: str) -> bool:
    u = (url or "").lower()
    return any(d in u for d in FACT_CHECK_DOMAINS)


def _too_similar(q: str, past: list[str], thresh: float = 0.75) -> bool:
    """True if q shares >=thresh of the smaller token set with any prior query (catches near-dup
    refinements like progressively adding quoted terms, which exact-match dedup misses)."""
    qt = set(q.split())
    if not qt:
        return True
    for p in past:
        pt = set(p.split())
        if pt and len(qt & pt) / min(len(qt), len(pt)) >= thresh:
            return True
    return False


# Function words that carry no search signal; a query that adds only these (or reorders
# existing terms) has nothing new to look up. Years are NOT here — "+2026" is a real refinement.
_QUERY_STOPWORDS = frozenset(
    "a an the of to in on at for and or nor but vs with without is are was were be been "
    "by from as that this these those how why what who whom when where which new".split())


def _content_tokens(q: str) -> set[str]:
    """Lower-cased content tokens of a query — function words and 1-2 char noise dropped."""
    return {t for t in q.lower().split() if len(t) > 2 and t not in _QUERY_STOPWORDS}


def _redundant_query(q: str, past: list[str]) -> bool:
    """True only when q introduces NO new content token beyond the prior queries — i.e. it is a
    reordering/rewording with nothing fresh to search. A query that adds any discriminating term
    (a state, a year, a named person, an exact figure) is a genuine REFINEMENT, not a duplicate,
    and must run on Serper rather than trigger a jump to the neural engine. This replaces the
    overlap-coefficient `_too_similar` for escalation: that flagged every superset refinement as a
    dup, which over-escalated to Exa (audit 2026-07-12, verify_eval_roadmap)."""
    if not past:
        return False
    seen: set[str] = set().union(*(_content_tokens(p) for p in past))
    return not (_content_tokens(q) - seen)

_SUMMARISE_TRUNCATE = 8000
_MAX = {"plan": 220, "controller": 900, "verdict": 400, "summarise": 700}
_STALE_LIMIT = 2   # consecutive rounds adding no NEW evidence -> conclude (futility)
_MAX_SEARCHES = 5  # distinct searches before wind-down (one pointed Exa shot, then verdict) — no cap-hits


def _chat_parsed(messages, max_tokens, model, parse_fn, default, counters, temperature=0.1):
    """Call the model + parse. On parse failure, hand the raw text to a small REPAIR_MODEL that fixes
    the JSON syntax verbatim, then re-parse; if that still fails, return `default` (never hard-error)."""
    counters["llm_calls"] += 1
    raw = _chat(messages, max_tokens, model, temperature=temperature)
    try:
        return parse_fn(raw)
    except Exception:
        try:
            counters["llm_calls"] += 1
            fixed = _chat(vp.build_repair_messages(raw), max(max_tokens, 1400), REPAIR_MODEL, temperature=0.0)
            out = parse_fn(fixed)
            counters["json_repairs"] += 1
            return out
        except Exception:
            counters["parse_fails"] += 1
            return default


def _render_state(raw_text: str, evidence: list[dict], steps: list[str],
                  analysis: str, open_questions: list[str], past_queries: list[str] | None = None) -> str:
    lines = [f"RAW TEXT:\n{raw_text}", ""]
    lines.append("STEPS SO FAR: " + (" | ".join(steps) if steps else "(none)"))
    if past_queries:
        lines.append("QUERIES ALREADY TRIED (do NOT repeat): " + " ; ".join(past_queries))
    if analysis:
        lines.append(f"\nRUNNING ANALYSIS:\n{analysis}")
    if open_questions:
        lines.append("\nOPEN QUESTIONS:\n- " + "\n- ".join(open_questions))
    lines.append("\nEVIDENCE (index. [provider, READ|snippet] url):")
    if not evidence:
        lines.append("  (none yet)")
    for e in evidence:
        tag = "READ" if e["scraped"] else "snippet"
        body = (e.get("summary") or e["snippet"] or "").strip().replace("\n", " ")
        quotes = ("  quotes: " + " / ".join(e["quotes"])) if e.get("quotes") else ""
        rel = "" if e.get("relevant", True) else " [judged not relevant]"
        lines.append(f"  {e['idx']}. [{e['provider']}, {tag}]{rel} {e['url']}\n     {body[:500]}{quotes[:300]}")
    return "\n".join(lines)


def _search(query: str, provider: str, seen: set, evidence: list[dict],
            date_ceiling, exclude_domains, exclude_urls, counters, fc_block: bool = False) -> int:
    """Run one search; append NEW results as snippet-only evidence. Returns count added."""
    try:
        results = search.search(query, SEARCH_RETRIEVE_K, date_ceiling=date_ceiling,
                                provider=provider, exclude_domains=exclude_domains)
    except search.SearchError as e:
        counters["search_errors"] += 1
        return 0
    new = [r for r in results if r["url"] not in seen and r["url"] not in exclude_urls]
    seen.update(r["url"] for r in new)
    if fc_block:  # simulate a live environment: no fact-checker articles reach the model
        before = len(new)
        new = [r for r in new if not _is_factcheck(r["url"])]
        counters["factcheck_blocked"] += before - len(new)
    # Keep the full result set (up to RETRIEVE_K) for the controller to triage over — rerank only
    # drops unreadable social/login-wall domains and orders by credibility (it does NOT cut to KEEP_K:
    # the model decides what to read, per the snippet-triage design).
    kept = credibility.rerank(new, SEARCH_RETRIEVE_K)
    for r in kept:
        evidence.append({"idx": len(evidence), "provider": provider, "url": r["url"],
                         "snippet": (r.get("snippet") or "").strip(), "scraped": False,
                         "summary": None, "quotes": [], "relevant": True,
                         "date": r.get("date")})
    counters["searches"] += 1
    return len(kept)


def _read(indices: list[int], evidence: list[dict], raw_text: str, model: str, counters, temperature=0.1) -> int:
    """Scrape + summarise selected results in place. Returns count newly read."""
    n = 0
    for i in indices:
        if not (0 <= i < len(evidence)) or evidence[i]["scraped"]:
            continue
        e = evidence[i]
        content = search.scrape(e["url"])
        if not content:
            counters["scrape_failed"] += 1
            e["scraped"] = True  # mark attempted so we don't retry endlessly
            e["summary"] = "(page could not be fetched — snippet only)"
            continue
        try:
            raw = _chat(vp.build_summarise_messages(raw_text, e["url"], content, _SUMMARISE_TRUNCATE),
                        _MAX["summarise"], model, temperature=temperature)
            s = vp.parse_summarise(raw)
            counters["llm_calls"] += 1
            e.update(scraped=True, relevant=bool(s.get("relevant", False)),
                     summary=(s.get("summary") or "").strip(), quotes=s.get("quotes") or [],
                     date=e["date"] or s.get("publication_date"))
            n += 1
        except Exception as ex:
            e["scraped"] = True
            e["summary"] = f"(summarise error: {ex})"
    counters["reads"] += n
    return n


def verify_text(raw_text: str, *, date_ceiling: str | None = None,
                exclude_domains: list[str] | None = None, exclude_urls: list[str] | None = None,
                block_factcheck: bool = False, pointed_exa: bool = False,  # ablation: 0/9 nudge flips -> off
                temperature: float = 0.1,  # survey passes 0.0 for reproducible veracity; dev stays 0.1
                model: str = VERIFICATION_MODEL, max_rounds: int = 10,
                verbose: bool = False, trace: dict | None = None) -> dict:
    """Verify RAW TEXT for misinformation. Returns {veracity, misinfo, misinfo_type, justification,
    n_rounds, provider_sequence, ...}. If `trace` is passed it is filled with the full run record.
    block_factcheck=True keeps ALL fact-checker articles out of the evidence (simulate live env)."""
    t_start = time.time()
    exclude_urls = set(exclude_urls or [])
    # block_factcheck: exclude FC domains server-side AND client-guard (see _search) so no fact-check
    # ever reaches the model — the "novel post, no fact-check published yet" case.
    eff_exclude = sorted(set(exclude_domains or []) | (set(FACT_CHECK_DOMAINS) if block_factcheck else set()))
    seen: set = set()
    evidence: list[dict] = []
    steps: list[str] = []
    provider_seq: list[str] = []
    counters = {"searches": 0, "reads": 0, "scrape_failed": 0, "search_errors": 0,
                "llm_calls": 0, "factcheck_blocked": 0, "json_repairs": 0, "parse_fails": 0}
    analysis, open_questions = "", []

    # --- Step 0: mandatory first Serper query ---
    plan = _chat_parsed(vp.build_plan_text_messages(raw_text), _MAX["plan"], model, vp.parse_plan_text,
                        {"main_claim": raw_text[:80], "query": raw_text[:120]}, counters, temperature=temperature)
    main_claim, q = plan["main_claim"], plan["query"] or raw_text[:120]
    n = _search(q, "serper", seen, evidence, date_ceiling, eff_exclude, exclude_urls, counters,
                fc_block=block_factcheck)
    steps.append(f'SEARCH(serper) "{q}" -> {n}')
    provider_seq.append("serper")
    past_queries = [q.strip().lower()]
    stale_rounds = 0
    n_distinct_searches = 1
    exa_pointed_done = False
    round_log = [{"round": 1, "action": "PLAN → SEARCH(serper)", "provider": "serper", "query": q,
                  "read_indices": [], "analysis": f"Identified main claim to test: {main_claim}",
                  "open_questions": [], "result": f"{n} results"}]
    if verbose:
        print(f"[plan] main_claim: {main_claim}\n[r1] SEARCH(serper) '{q}' -> {n} results", flush=True)

    # --- Controller loop ---
    stopped = "verdict"
    for rnd in range(2, max_rounds + 2):
        state = _render_state(raw_text, evidence, steps, analysis, open_questions, past_queries)
        d = _chat_parsed(vp.build_controller_messages(state), _MAX["controller"], model, vp.parse_controller,
                         {"analysis": analysis, "open_questions": open_questions, "action": "VERDICT",
                          "read_indices": [], "query": "", "provider": "serper"}, counters, temperature=temperature)
        analysis, open_questions, action = d["analysis"], d["open_questions"], d["action"]
        rlog = {"round": rnd, "action": action, "provider": d.get("provider", ""),
                "query": d.get("query", ""), "read_indices": d.get("read_indices", []),
                "analysis": analysis, "open_questions": open_questions, "result": ""}
        round_log.append(rlog)

        if action == "VERDICT":
            rlog["result"] = "evidence decisive → conclude"
            break
        if action == "READ" and d["read_indices"]:
            got = _read(d["read_indices"], evidence, raw_text, model, counters, temperature=temperature)
            steps.append(f"READ {d['read_indices']} -> {got}")
            rlog["result"] = f"read {got} source(s) in full"
            stale_rounds = 0 if got else stale_rounds + 1
            if verbose:
                print(f"[r{rnd}] READ {d['read_indices']} -> {got} summarised", flush=True)
        elif action == "SEARCH" and d["query"]:
            prov, qn = d["provider"], d["query"].strip().lower()
            if _too_similar(qn, past_queries):  # dedup: skip near-duplicate/refinement queries
                steps.append(f'SEARCH({prov}) "{d["query"]}" -> SKIPPED(similar)')
                rlog["result"] = "SKIPPED (near-duplicate of a prior query)"
                stale_rounds += 1
                if verbose: print(f"[r{rnd}] SEARCH({prov}) '{d['query']}' -> SKIPPED (near-dup)", flush=True)
            else:
                n = _search(d["query"], prov, seen, evidence, date_ceiling, eff_exclude, exclude_urls,
                            counters, fc_block=block_factcheck)
                steps.append(f'SEARCH({prov}) "{d["query"]}" -> {n}')
                rlog["result"] = f"{n} new results"
                provider_seq.append(prov); past_queries.append(qn); n_distinct_searches += 1
                stale_rounds = 0 if n > 0 else stale_rounds + 1
                if verbose: print(f"[r{rnd}] SEARCH({prov}) '{d['query']}' -> {n} results", flush=True)
        else:
            rlog["result"] = "no actionable decision → conclude"
            break  # empty/invalid action -> verdict on what we have

        # Wind-down: futility (no new evidence) OR search budget exhausted. Before concluding, give Exa
        # ONE pointed shot at the still-open question, then verdict. Guarantees no cap-hits.
        if stale_rounds >= _STALE_LIMIT or n_distinct_searches >= _MAX_SEARCHES:
            if pointed_exa and not exa_pointed_done and open_questions:
                pq = (main_claim + " — " + " ".join(open_questions[:2]))[:300]
                if not _too_similar(pq.strip().lower(), past_queries):
                    m = _search(pq, "exa", seen, evidence, date_ceiling, eff_exclude, exclude_urls,
                                counters, fc_block=block_factcheck)
                    provider_seq.append("exa"); past_queries.append(pq.strip().lower())
                    steps.append(f'SEARCH(exa,pointed) "{pq[:55]}..." -> {m}')
                    if verbose: print(f"[r{rnd}] pointed-EXA on open question -> {m} results", flush=True)
                exa_pointed_done = True
                stale_rounds = 0
                continue
            stopped = "futile" if stale_rounds >= _STALE_LIMIT else "budget"
            if verbose: print(f"[r{rnd}] wind-down ({stopped}) -> verdict", flush=True)
            break
    else:
        stopped = "cap"

    # --- Final verdict on the raw text ---
    ev_str = _render_state(raw_text, evidence, [], "", [])
    verdict = _chat_parsed(vp.build_verdict_text_messages(raw_text, analysis, ev_str), _MAX["verdict"], model,
                           vp.parse_misinfo, {"veracity": 3, "misinfo": True, "misinfo_type": "UNSUPPORTED",
                                              "justification": "(verdict parse failed)"}, counters, temperature=temperature)
    n_rounds = len(provider_seq) + counters["reads"]  # search rounds + read rounds ~ controller iterations
    out = {**verdict, "main_claim": main_claim, "analysis": analysis, "open_questions": open_questions,
           "n_search_rounds": len(provider_seq), "provider_sequence": provider_seq,
           "n_evidence": len(evidence), "n_read": counters["reads"], "stopped": stopped,
           "llm_calls": counters["llm_calls"], "elapsed_s": round(time.time() - t_start, 1),
           "counters": counters}
    if verbose:
        print(f"[verdict] veracity={out['veracity']} misinfo={out['misinfo']} type={out['misinfo_type']} "
              f"| {len(provider_seq)} searches, {counters['reads']} reads, {out['elapsed_s']}s", flush=True)
    if trace is not None:
        trace.update({"raw_text": raw_text, "main_claim": main_claim, "steps": steps,
                      "rounds": round_log, "evidence": evidence, "analysis": analysis,
                      "open_questions": open_questions, "verdict": verdict, **out})
    return out
