"""Post-level verify loop (design: docs/verify_tweet_claims_loop_design.md, locked READ = v4 2026-07-10).

One post = one accumulating loop over its checkworthy claims (v6, Daniel's Arm-C design
2026-07-13 — "just focus on resolving claims"):

    CONTEXT (v7.4, once, before round 1): resolve the post's t.co link -> scrape the linked
    article -> LLM pins dangling referents into each claim (the loop then runs on the
    RESOLVED text; originals in the record) and adjudicates SELF-SOURCED claims (linked page
    on the outlet's own domain IS the saying/act -> supported, no search). The article is
    then discarded — never evidence; a third-party linked domain is excluded like the origin.
    Claims the extractor filed "unresolved" regain eligibility when resolution succeeds.

    per round:  QUERY (LLM: one keyword query for the most at-risk under-tried open claims)
                -> serper (origin + linked domain excluded) -> TRIAGE -> scrape gauntlet
                -> READ per doc (pointer output)
                -> resolve pointers (±3-sentence context windows, cited core marked;
                   ±2 measured RISKY 2026-07-13 — 60% of refute entries lose material content)
                -> RESOLVE (LLM: re-judge each open claim's DOSSIER, set statuses, note gaps)
    routing is CODE: while any open claim is under its per-claim budget -> QUERY again;
    then ONE Exa round for all unresolved claims; then done. No conclude decision, no
    verdict call — the post verdict is a CODE RULE over the claim labels (code_verdict:
    nudge iff >= nudge_min_refuted refuted; other thresholds TBD, counts in the record).

Search budget is PER CLAIM: each claim gets `tries_per_claim` (2) targeted Serper queries;
an open claim at 2/2 is code-marked unsupported (guard event) and rotation moves on. State
per claim = a DOSSIER: status, tried n/2, type, code-computed bar check, dieted evidence
(all refute-direction entries kept; snippet superseded by same-domain full read dropped;
cap dossier_entries_cap); closed claims freeze to a one-line resolution and auto-re-expand
if contrary evidence lands later. RESOLVE's per-claim "gap" notes brief the next QUERY.

READ stance is ADVISORY scaffolding — RESOLVE re-judges every claim from the resolved text.

POINTER INVARIANT (Daniel, 2026-07-14 — respect it in every prompt edit): models OUTPUT only
pointers (claim ids, sentence numbers, result ids, status enums, short queries/gap notes) —
never evidence sentences; models are INPUT only resolved text (windows, snippets, claim texts,
with id labels attached) — never bare pointers to dereference. Code does all resolution
(resolve_windows, snippet attachment, resolution one-liners, the verdict justification).

The pipeline fn `verify_post(post, pools)` is harness-compatible (pipeline/harness.py).
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import unicodedata
import urllib.request
from dataclasses import dataclass

from pipeline.credibility import is_primary_source, is_trusted_factchecker
from pipeline.pools import Pools
from pipeline.read_select import claim_keywords, is_junk, is_off_topic, numbered_block
from pipeline import disk_cache
from pipeline.search import _scrape_jina, newsguard_score_map

log = logging.getLogger("verify_tweet_claims")

# ---------------------------------------------------------------------------
# Query redundancy (moved here from verify_text.py when that module was archived,
# 2026-07-13 — this was its one live function)
# ---------------------------------------------------------------------------

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

# Bump on ANY behavior change; every run record carries it. Ledger of versions, their runs
# and audits: docs/verify_loop_versions.md
LOOP_VERSION = "v7.5"  # stamp lags behind docs/verify_loop_versions.md changes; bump per behavior change

# =============================================================================
# Config
# =============================================================================

@dataclass
class PostVerifyConfig:
    tries_per_claim: int = 2     # targeted Serper queries per claim before code marks it
                                 # unsupported and rotation moves on (v5); one final Exa round
                                 # then re-attempts ALL unresolved claims together
    triage_enabled: bool = True  # snippet triage picks WHICH hits to read and HOW MANY (no cap, v5)
    pages_per_round: int = 3     # scrapes kept per round when triage is OFF (fixed walk)
    triage_max_tokens: int = 300
    serper_k: int = 10           # results requested per query
    exa_enabled: bool = True     # one Exa escalation round when opens remain / query goes stale
    date_ceiling: bool = False   # benchmark parity mode: cap search results at the post/claim
    #                              date (Serper tbs, Exa endPublishedDate) — ClaimCheck does
    #                              this (cd_max=claim date); OFF in production (a live nudge
    #                              tool wants the newest evidence)
    exa_k: int = 5               # Exa results (page text comes bundled — no scrape)
    cap_tok: int = 2500          # article cap fed to READ (knee measured 2026-07-10)
    ctx_window: int = 3          # ±sentences of mechanical context around cited segs
    max_segs: int = 8            # cited-sentence cap per evidence entry (v2: READ sometimes
                                 # marks whole articles; past ~8 the citation signal is junk)
    open_max_tokens: int = 300    # QUERY call budget (targets + one keyword query)
    read_max_tokens: int = 2500  # 1500 length-capped once on a 12-claim post (dev-500 run
    #                              2026-07-15, the only error in 355 posts); pointer JSON for
    #                              many claims x many segs needs the headroom
    step_max_tokens: int = 3200   # RESOLVE call budget (kept name for script compat)
    exa_query_max_tokens: int = 400
    requery_max_tokens: int = 120  # one forced diversified Serper query before an Exa jump
    dossier_entries_cap: int = 5  # per-claim entries shown to RESOLVE — direction-aware: ALL
                                  # refute-direction entries are always kept (v6 diet; guards
                                  # count over the FULL evidence list regardless)
    nudge_min_refuted: int = 1    # code verdict (v6): nudge iff >= this many refuted claims.
                                  # Conflicting/unsupported concentration thresholds TBD w/
                                  # Daniel — counts are in the record for offline tuning.
    link_context_enabled: bool = True  # v7.4 CONTEXT step: resolve the post's linked article
                                  # to pin claim referents BEFORE searching; the article is
                                  # NEVER evidence (Daniel 2026-07-20)
    context_max_tokens: int = 900
    context_text_cap: int = 6000  # linked-article chars fed to the CONTEXT call
    max_rounds: int = 0           # 0 = unlimited (production); N stops after round N with
                                  # stopped="round-cap" (profiling runs; open claims stay
                                  # distinguishable via coerced_open)


CFG = PostVerifyConfig()

# =============================================================================
# Preprocessing (scrape text -> deduped numbered sentences)
# =============================================================================

_UNI = {"‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ", "…": "..."}
_BOILER = re.compile(r"^(sign up|subscribe|read more|related( articles| stories)?:?|share this|"
                     r"follow us|advertisement|newsletter|click here|watch:|read:|more:|"
                     r"support (our|independent))", re.I)
_SENT = re.compile(r'(?<=[.!?"])\s+(?=[A-Z"\'0-9(])')


def clean_text(text: str) -> str:
    for k, v in _UNI.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    lines, seen = [], set()
    for ln in text.splitlines():
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if not ln or _BOILER.match(ln) or (len(ln) < 25 and not re.search(r"[.!?]$", ln)):
            continue
        if ln in seen:
            continue
        seen.add(ln)
        lines.append(ln)
    return "\n".join(lines)


def sentences(text: str) -> list[str]:
    """Sentence-split + sentence-level dedupe (scrapes repeat content; 59 dups measured)."""
    out, seen = [], set()
    for para in text.split("\n"):
        for s in _SENT.split(para):
            s = s.strip()
            key = " ".join(s.split()).lower()
            if s and key not in seen:
                seen.add(key)
                out.append(s)
    return out


# numbered_block moved to pipeline/read_select.py (keyword-anchored selection over the cap,
# ORIGINAL sentence indices preserved so READ citations still resolve; scrape audit 2026-07-12)

# =============================================================================
# Candidate ranking — quality-tiered, then the provider's own relevance order
# =============================================================================

def _domain_of(url: str) -> str:
    m = re.findall(r"https?://([^/]+)", url or "")
    return re.sub(r"^www\.", "", m[0]).lower() if m else ""


# Domains NewsGuard rates under a different name: rebrands and same-org editions. Curated
# by hand (Daniel spotted ms.now sitting in the UNRATED bin, 2026-07-21) — deliberately NOT
# fuzzy-matched, because near-identical names are often unrelated organisations
# (nyp.org = a hospital, not the NY Post; ajc.org = a advocacy group, not the Atlanta paper).
NG_ALIASES = {
    "ms.now": "msnbc.com",                  # MSNBC -> MS NOW rebrand
    "reutersconnect.com": "reuters.com",
    "dailymail.com": "dailymail.co.uk",     # US edition
    "arabnews.pk": "arabnews.com",
    "chinadailyhk.com": "chinadaily.com.cn",
}

# Prefixes that never change the publisher (mobile/section subdomains)
_NG_PREFIXES = ("m.", "amp.", "mobile.", "news.", "www.")


def _ng_of(dom: str, scores: dict) -> float | None:
    dom = (dom or "").lower().removeprefix("www.")
    s = scores.get(dom)
    if s is None and dom in NG_ALIASES:
        s = scores.get(NG_ALIASES[dom])
    if s is None:
        for pre in _NG_PREFIXES:            # m.economictimes.com -> economictimes.com
            if dom.startswith(pre):
                s = scores.get(dom[len(pre):])
                if s is not None:
                    break
    if s is None and dom.count(".") > 1:  # try the registrable parent (news.site.com -> site.com)
        s = scores.get(".".join(dom.split(".")[-2:]))
    return s


def _source_tier(dom: str, url: str, scores: dict) -> int:
    """0 primary (credibility.is_primary_source: gov TLD patterns + allowlist),
    1 NewsGuard >= 75, 2 NewsGuard 60-75, 3 unrated, 4 NewsGuard < 60."""
    if is_primary_source(dom) or ".gov/" in (url or ""):
        return 0
    s = _ng_of(dom, scores)
    if s is None:
        return 3
    return 1 if s >= 75 else 2 if s >= 60 else 4


def rank_hits(hits: list[dict], scores: dict | None = None) -> list[dict]:
    """Stable sort by source-quality tier (_source_tier), then provider relevance order —
    also fixes the near-dup order-dependence (the press release now outranks the article
    that copied it). NB: tier 1's NG>=75 cutoff is READ-ORDERING only; the corroboration
    gate's single-source cutoff is NG_STRONG=90 — distinct thresholds, don't conflate."""
    if scores is None:
        scores = newsguard_score_map()
    return sorted(hits, key=lambda h: _source_tier(
        _domain_of(h.get("url") or ""), h.get("url") or "", scores))  # sorted() is stable


# =============================================================================
# Closure guards — evidence-grounded gates on STEP's ledger updates (v4, 2026-07-13)
# =============================================================================
# Two web-ground-truthed audits showed the STEP prompt's prose closure rules are gameable
# (28% of supported/refuted closes rested on unrated blogs counted as "independent reliable
# sources"). The counting moves here, into code; STEP keeps the semantic work (same-event,
# exact-proposition, attribution-vs-content). All pure functions — replayable over saved runs.

NG_STRONG = 90.0    # a single secondary at/above this closes a claim alone
NG_RELIABLE = 60.0  # reliable-secondary floor; two independent ones close a claim

SUPPORT_STANCES = {"supports", "partially-supports"}
REFUTE_STANCES = {"refutes", "partially-refutes"}
_EXACT_STANCE = {"supported": "supports", "refuted": "refutes"}  # v7: the anchoring stance per close

SNIPPET_MIN_OVERLAP = 0.15  # v7.2: min claim-token share a snippet must carry to count as a voice


def _content_tokens(text: str | None) -> set:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", (text or "").lower()) if len(w) > 3}


def _on_topic(snippet_text: str | None, claim_toks: set) -> bool:
    if not claim_toks:
        return False
    return len(claim_toks & _content_tokens(snippet_text)) / len(claim_toks) >= SNIPPET_MIN_OVERLAP


# Same-publisher families: distinct domains, one editorial voice — never two "independent"
# voices and never independent corroboration of each other (v7.2, audit: @TheSun closed on
# thesun.co.uk + thescottishsun.co.uk counted as 2 voices; roster lists the US domain so
# origin-exclusion missed the UK sisters).
_PUB_FAMILIES = [
    {"thesun.co.uk", "thescottishsun.co.uk", "the-sun.com", "thesun.ie"},
    {"foxnews.com", "foxbusiness.com", "nation.foxnews.com", "radio.foxnews.com"},
    {"nypost.com", "pagesix.com"},
    {"dailymail.co.uk", "mailonsunday.co.uk", "thisismoney.co.uk"},
]


def _voice_key(dom: str) -> str:
    """Collapse a domain to its publisher voice: family membership first, else the
    registrable base (news.yahoo.com -> yahoo.com) so subdomains never count twice."""
    d = (dom or "").lower().removeprefix("www.")
    for fam in _PUB_FAMILIES:
        if d in fam or any(d.endswith("." + f) for f in fam):
            return sorted(fam)[0]
    parts = d.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "net", "org", "ac", "gov", "go", "gob"):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else d

# opinion-shaped URL paths: commentary may quote (attribution) but never establish a fact.
# All 11 opinion leaks in the v3 runs matched this (probe 2026-07-13, clog 130726).
_OPINION_URL = re.compile(r"/(opinions?|commentary|comment|op-eds?|opeds?|editorials?|"
                          r"blogs?|columns?|columnists?|voices|perspectives?)/", re.I)

_WIKIPEDIA = "wikipedia.org"


def _rel_info(dom: str, url: str, scores: dict) -> tuple[str, float | None]:
    """Reliability tag for an evidence entry: PRIMARY / RELIABLE (ng attached) / UNRATED /
    UNRELIABLE. Wikipedia is NG-unrated but counts as a reliable secondary voice (Daniel
    2026-07-13) — one voice toward the 2-source branch, never single-sufficing. Professional
    fact-checkers count as STRONG reliable voices regardless of NewsGuard coverage (Daniel
    2026-07-15: 'generally trust fact checkers'; NG's map is US-centric — Africa Check,
    Boom Live etc. were UNRATED and starved true claims)."""
    if is_primary_source(dom) or ".gov/" in (url or ""):
        return "PRIMARY", None
    if is_trusted_factchecker(dom):
        return "RELIABLE", NG_STRONG          # single full-read may close, like NG>=90
    ng = _ng_of(dom, scores)
    if dom == _WIKIPEDIA or dom.endswith("." + _WIKIPEDIA):
        # explicit mid-tier score (Daniel 2026-07-21): counts as a reliable voice in the
        # 2-source branch and in figures' NG 70-79 bin; never single-sufficing (< NG_STRONG)
        return "RELIABLE", ng if ng is not None else 75.0
    if ng is None:
        return "UNRATED", None
    return ("RELIABLE", ng) if ng >= NG_RELIABLE else ("UNRELIABLE", ng)


def _as_cid(x, n: int) -> int | None:
    """Claim/result id from LLM JSON: int or numeric string ("3" -> 3), validated 1..n.
    isinstance(int) alone silently dropped string-typed ids (post-50 audit bug)."""
    if isinstance(x, bool):
        return None
    if isinstance(x, str) and x.strip().isdigit():
        x = int(x.strip())
    if isinstance(x, int) and 1 <= x <= n:
        return x
    return None


def _qualifying(evidence: list[dict], cid: int, status: str, claims: list[dict]) -> list[dict]:
    """Evidence entries that can count toward closing `cid` as `status`: matching claim,
    not republication-flagged, and not an opinion piece — UNLESS the claim is an attribution
    ("X said Y"), where an op-ed quoting X does confirm the saying (Daniel 2026-07-13).
    Full-read entries additionally need a stance in the closing direction. Snippet-only
    entries qualify ONLY from high-reliability sources (PRIMARY or NG >= NG_STRONG) and only
    as corroborating voices — direction-less (stance inference from a snippet is unreliable;
    the full-read anchor carries the direction), never the basis of a close (amendment
    2026-07-13: NYT/WSJ-class sources are systematically unscrapeable — their snippets must
    pull weight, but a claim never closes on snippets alone). v7.2 (Daniel 2026-07-15): a
    snippet voice must additionally be ON-TOPIC — share >= SNIPPET_MIN_OVERLAP of the claim's
    content tokens. Snippets bind to claims by QUERY TARGETING, not content, so an unrelated
    result (or scraped player chrome: 'Skip to player') could count as a corroborating voice;
    measured: 5/503 v7 closes depended on such snippets, all junk."""
    want = SUPPORT_STANCES if status == "supported" else REFUTE_STANCES
    attribution = (claims[cid - 1].get("t") == "attribution")
    claim_toks = _content_tokens(claims[cid - 1].get("c"))
    out = []
    for e in evidence:
        if e.get("claim_id") != cid or e.get("republication"):
            continue
        if e.get("opinion") and not attribution:
            continue
        if e.get("snippet_only"):
            if (e.get("rel") == "PRIMARY" or
                    (e.get("rel") == "RELIABLE" and (e.get("ng") or 0) >= NG_STRONG)) and \
                    _on_topic(e.get("text"), claim_toks):
                out.append(e)
            continue
        if (e.get("stance") or "").lower() in want:
            out.append(e)
    return out


def _meets_bar(entries: list[dict], exact: str | None = None) -> tuple[bool, str, list[str]]:
    """Corroboration bar (locked w/ Daniel 2026-07-12; snippet amendment 2026-07-13;
    institutional demotion 2026-07-21): >=1 full-read secondary with NG >= NG_STRONG, or
    >=2 independent voices (institutional/reliable) on distinct domains of which AT LEAST
    ONE is full-read —
    a high-reliability snippet may fill a non-sole slot. UNRATED corroborates narratively
    only; UNRELIABLE (NG < NG_RELIABLE) is never a basis. Returns (ok, detail,
    snippet_domains_counted) — the third element feeds guard_events so audits can measure
    how often snippet corroboration fires. True author/wire independence is backlog #3;
    distinct-domain counting also collapses a snippet duplicating a read source's domain.
    v7 (2026-07-15 audits): `exact` names the stance that must ANCHOR the close ("supports"
    / "refutes") — at least one full-read entry must carry the EXACT stance; partial-
    direction and snippet entries corroborate but never anchor. Basis: 15/23 refutations
    were wrong, many resting on partially-refutes/adjacent entries; ~12-15% of supported
    closes laundered a load-bearing specific through partially-supports."""
    full = [e for e in entries if not e.get("snippet_only")]
    if exact is not None and not any((e.get("stance") or "").lower() == exact for e in full):
        return False, f"no full-read '{exact}' anchor — partial/snippet evidence only", []
    # v7.5 (Daniel 2026-07-21): the INSTITUTIONAL tier (code rel PRIMARY — gov TLDs,
    # .edu, allowlist) no longer closes a claim ALONE. It is a strong voice in the
    # 2-voice branch; single-close authority awaits READ-level detection of truly
    # PRIMARY/definitive records (roadmap). NG>=NG_STRONG single-close unchanged.
    strong = sorted({e["domain"] for e in full
                     if e.get("rel") == "RELIABLE" and (e.get("ng") or 0) >= NG_STRONG})
    if strong:
        return True, f"strong secondary (NG>={NG_STRONG:.0f}): {strong}", []
    # voices are PUBLISHERS, not domain strings (v7.2): subdomains and same-family sister
    # sites (thesun.co.uk / thescottishsun.co.uk) collapse to one voice
    full_doms = {_voice_key(e["domain"]) for e in full if e.get("rel") in ("PRIMARY", "RELIABLE")}
    snip_doms = sorted({_voice_key(e["domain"]) for e in entries if e.get("snippet_only")} - full_doms)
    if full_doms and len(full_doms) + len(snip_doms) >= 2:
        used = snip_doms[:max(0, 2 - len(full_doms))] if len(full_doms) < 2 else []
        detail = f"2+ independent reliable voices: {sorted(full_doms)}"
        if used:
            detail += f" + high-reliability snippet(s): {snip_doms}"
        return True, detail, snip_doms if len(full_doms) < 2 else []
    if not entries:
        return False, "no qualifying evidence entries", []
    tally = {}
    for e in entries:
        key = ("snippet:" if e.get("snippet_only") else "") + (e.get("rel") or "?")
        tally[key] = tally.get(key, 0) + 1
    extra = f" (single reliable: {sorted(full_doms)})" if full_doms else \
        (" (high-reliability snippets alone never close)" if snip_doms else "")
    return False, f"below bar: {tally}{extra}", []


def _apply_step_ledger(step: dict, ledger: dict[int, str], evidence: list[dict],
                       targeted: set[int], claims: list[dict], final: bool,
                       rnd: int) -> tuple[list[dict], list[int]]:
    """Apply STEP's proposed ledger through the closure guards. Mutates `ledger`; returns
    (guard_events, reopened) — reopened lists claims a guard held open after STEP tried to
    close them (the loop forces another search round for these, budget permitting)."""
    n = len(claims)
    events: list[dict] = []
    reopened: list[int] = []
    for row in (step.get("ledger") or []):
        raw = row.get("claim_id")
        cid = _as_cid(raw, n)
        if cid is None:
            if raw is not None:
                events.append({"round": rnd, "guard": "cid-invalid", "claim_id": raw})
            continue
        if isinstance(raw, str):
            events.append({"round": rnd, "guard": "cid-coerced", "claim_id": cid})
        st = (row.get("status") or "").lower()
        if st not in ("supported", "refuted", "conflicting", "unsupported", "open"):
            continue
        if st == "unsupported" and ledger.get(cid) == "open" and cid not in targeted \
                and not final:
            # "we searched and found nothing" needs an actual search that targeted it; at
            # the forced conclude an untargeted open claim IS honestly unsupported — allowed
            events.append({"round": rnd, "guard": "unsupported-untargeted", "claim_id": cid,
                           "refused": st})
            reopened.append(cid)
            continue
        if st in ("supported", "refuted") and ledger.get(cid) != st:
            qual = _qualifying(evidence, cid, st, claims)
            ok, why, snip_used = _meets_bar(qual, _EXACT_STANCE[st])
            if not ok:
                events.append({"round": rnd,
                               "guard": "close-unbacked" if not qual else "close-below-bar",
                               "claim_id": cid, "refused": st, "detail": why})
                if ledger.get(cid, "open") == "open":
                    reopened.append(cid)
                continue
            if snip_used:  # audit visibility: how often does snippet corroboration decide?
                events.append({"round": rnd, "guard": "snippet-corroboration",
                               "claim_id": cid, "status": st, "snippets": snip_used})
            # v7.3: a one-sided close may not silently win a two-sided dossier — if the
            # OPPOSITE direction also clears the full closing bar, the honest label is
            # conflicting (its own bar is met by construction). Found via the ChildrensHD
            # study flip (2026-07-17): the flawed study's own abstract, a PRIMARY, closed
            # "supported" over two 90+ refutations sitting in the same dossier. 12/1016
            # closes on dev-500 A+B would coerce — all in known failure classes.
            opp = "refuted" if st == "supported" else "supported"
            opp_ok, _, _ = _meets_bar(_qualifying(evidence, cid, opp, claims),
                                      _EXACT_STANCE[opp])
            if opp_ok:
                events.append({"round": rnd, "guard": "both-directions-qualify",
                               "claim_id": cid, "refused": st, "coerced": "conflicting"})
                ledger[cid] = "conflicting"
                continue
        if st == "conflicting" and ledger.get(cid) != st:
            # v7.1: "conflicting" is not an escape hatch from the refuted/supported bars —
            # a genuine conflict needs at least one FULL-READ qualifying entry in EACH
            # direction (AVeriTeC parity smoke 2026-07-15: anchor-refused refutations were
            # resettling as conflicting, which had no bar at all).
            has_sup = any(not e.get("snippet_only")
                          for e in _qualifying(evidence, cid, "supported", claims))
            has_ref = any(not e.get("snippet_only")
                          for e in _qualifying(evidence, cid, "refuted", claims))
            if not (has_sup and has_ref):
                events.append({"round": rnd, "guard": "conflict-one-sided", "claim_id": cid,
                               "refused": st,
                               "detail": f"full-read support={has_sup} refute={has_ref}"})
                if ledger.get(cid, "open") == "open":
                    reopened.append(cid)
                continue
        ledger[cid] = st
    return events, reopened


# =============================================================================
# Circularity guard — republished origin content is not independent evidence
# =============================================================================

_REPRINT = re.compile(r"(originally (published|appeared)|republished from|reprinted from|"
                      r"cross-?posted from|this article (was|is) from|courtesy of)", re.I)

# v2: when the POSTING outlet is a wire agency, its wire copy on other domains is its own voice
_WIRE_BYLINE = {
    "apnews.com": re.compile(r"\(ap\)\s*[—-]|by [a-z .'-]{3,40},? associated press|"
                             r"associated press writer", re.I),
    "reuters.com": re.compile(r"\(reuters\)|by [a-z .'-]{3,40},? reuters", re.I),
    "afp.com": re.compile(r"\(afp\)|agence france-presse", re.I),
}

# v2: block/error/consent pages that clear the length floor must not reach READ
_JUNK = re.compile(r"(not authorized to access|access denied|page not found|"
                   r"problem finding that (article|page)|enable (javascript|cookies)|"
                   r"subscribe to (continue|read)|to continue reading|are you a robot|"
                   r"verify you are (a )?human|tollbit|captcha|log in to continue)", re.I)


def origin_tokens(domain: str, handle: str) -> list[str]:
    """Name tokens identifying the origin outlet, e.g. thegrayzone.com -> 'grayzone'."""
    toks = set()
    core = re.sub(r"\.(com|org|net|news|co|io|us|uk)(\.[a-z]{2})?$", "", (domain or "").lower())
    core = re.sub(r"^(www\.|the)", "", core)
    if len(core) >= 5:
        toks.add(core)
    h = re.sub(r"(news|media|official)$", "", (handle or "").lower().lstrip("@").replace("the", "", 1))
    if len(h) >= 5:
        toks.add(h)
    return sorted(toks)


# Syndication hosts that carry other outlets' articles verbatim — a page here that names the
# origin outlet early is the origin's own copy, not independent corroboration (v7, 2026-07-15
# audit: HuffPost->yahoo and Fox->yahoo closed "supported" on their own syndicated articles).
_SYND_HOSTS = ("yahoo.com", "msn.com", "aol.com", "archive.ph", "archive.today", "archive.is",
               "ground.news")


def _is_synd_host(dom: str) -> bool:
    return any(dom == h or dom.endswith("." + h) for h in _SYND_HOSTS)


def is_republication(text: str, origin_toks: list[str], origin_domain: str = "",
                     doc_domain: str = "") -> bool:
    """An explicit reprint phrase naming the origin outlet nearby, or (v2) a wire-agency
    byline when the posting outlet IS that agency, or (v7) a syndication host whose page
    names the origin outlet early. Deliberately NARROW otherwise: a page that merely
    mentions the outlet (e.g. to dispute its reporting) must NOT trip this — that judgment
    belongs to READ's republication flag (Daniel, 2026-07-10)."""
    wire = _WIRE_BYLINE.get(origin_domain)
    if wire and wire.search(text[:2000]):
        return True
    if doc_domain and origin_toks and _is_synd_host(doc_domain) \
            and any(t in text[:2500].lower() for t in origin_toks):
        return True
    if not origin_toks:
        return False
    for m in _REPRINT.finditer(text[:5000]):
        neigh = text[max(0, m.start() - 100):m.end() + 150].lower()
        if any(t in neigh for t in origin_toks):
            return True
    return False


def _shingles(text: str, n: int = 8, span: int = 5000) -> set:
    words = re.sub(r"\W+", " ", text[:span].lower()).split()
    return {" ".join(words[i:i + n]) for i in range(0, max(len(words) - n + 1, 0))}


def near_dup(sh_a: set, sh_b: set, thresh: float = 0.5) -> bool:
    if not sh_a or not sh_b:
        return False
    inter = len(sh_a & sh_b)
    return inter / max(min(len(sh_a), len(sh_b)), 1) >= thresh


# =============================================================================
# Prompts
# =============================================================================

# =============================================================================
# CONTEXT step (v7.4) — resolve the post's linked article to pin claim referents.
# The article text is used ONLY here and then discarded: it is the outlet's chosen
# source and NEVER counts as evidence (its final URL and domain are excluded from
# retrieval). Two jobs (Daniel 2026-07-20): (1) rewrite claims into self-contained
# form — referents pinned from the article, nothing imported; the loop then runs
# ENTIRELY on the resolved text (the original is kept in the record), and claims
# the extractor filed "unresolved" regain eligibility when resolution succeeds.
# (2) single-source-of-truth adjudication: when the linked page is the outlet's OWN
# venue of the saying/act the claim describes (their interview, their broadcast,
# their publication act), the claim is trivially true — closed "supported" here,
# no search spent. Code requires the link to be the outlet's own domain for this.
# =============================================================================

CONTEXT_SYSTEM = """You prepare the claims of a news outlet's social-media post for fact-checking. You receive the POST, the article the post links to (the outlet's chosen source — it will NOT be used as evidence), and the CLAIMS extracted from the post.
For each claim, two judgments:
1. "resolved": rewrite the claim so it stands alone — use the article ONLY to pin referents the post leaves dangling: full names and roles, places, dates, and antecedents of "the report", "she", "it", "this". Unnamed actors are dangling referents too — "an SNL star", "the couple", "a GOP senator", "the study" become the full name the article gives them; a claim about an unnamed someone does NOT stand alone when the article names them. The proposition must stay EXACTLY what the post asserts: never import the article's facts, figures, qualifiers, or details the post itself does not state, and never strengthen or weaken the claim. Return null only when the claim needs no pinning or the article does not identify the referent.
2. "channel" — ONLY for claims of the form "X said/wrote/revealed Y" (a reported saying by a named person or organization): through which channel did X make the statement, per the article? Exactly one of:
   - "to-this-outlet": X spoke to this outlet — its interview, its show, "told us", quotes gathered by this outlet's own reporter;
   - "own-piece-here": the linked page IS X's own article, column, or report published by this outlet — X must be the page's author (byline) or the page must present itself as X's own piece; a page written by the outlet's staff or wires that quotes X's remarks, newsletter, broadcast, or essay made elsewhere is "elsewhere", however long the quotes;
   - "elsewhere": X made the statement at any event, document, or platform outside this page — a briefing, court, speech, social media, another outlet — which this article reports or quotes;
   - "unclear": the article does not show where the statement was made.
   An article quoting a statement is not its channel unless the statement was made TO this outlet. For every claim that is not a reported saying — events, actions, figures, anything about the world — "channel" is "n/a": the article reporting a thing is never the place the thing happened.
Output strictly valid JSON, nothing else:
{"claims": [{"claim_id": <n>, "resolved": <string or null>, "channel": "to-this-outlet"|"own-piece-here"|"elsewhere"|"unclear"|"n/a"}]}
Cover every claim id given."""

_TCO = re.compile(r"https?://t\.co/\w+")
_SOCIAL_HOSTS = ("twitter.com", "x.com", "t.co", "facebook.com", "instagram.com",
                 "youtube.com", "youtu.be", "rumble.com", "tiktok.com")


def _is_social_host(dom: str) -> bool:
    """Exact-or-subdomain match — substring matching burned us ('x.com' in 'vox.com',
    't.co' in 'nypost.com', smoke 2026-07-20)."""
    d = (dom or "").lower().removeprefix("www.")
    return any(d == h or d.endswith("." + h) for h in _SOCIAL_HOSTS)


# An outlet's links do not always resolve to its registered domain: rebrands and sister
# brands break every origin comparison (exclusion, republication, origin_link). Found by
# the 2026-07-20 CONTEXT smoke; grow as mismatches surface.
_ORIGIN_ALIASES = {
    "msnbc.com": {"ms.now"},          # MSNBC -> MS NOW rebrand
    "the-sun.com": {"thesun.co.uk"},  # US site links to the UK edition
    "nypost.com": {"pagesix.com"},    # NY Post's gossip brand
}


def _origin_domains(origin: str) -> set[str]:
    o = (origin or "").strip().lower().removeprefix("www.")
    return ({o} | _ORIGIN_ALIASES.get(o, set())) if o else set()


def _final_url(u: str, timeout: int = 12) -> str | None:
    """Follow the t.co redirect chain to the real URL. t.co's edge 403s Python's TLS
    fingerprint regardless of headers (measured 2026-07-20) but serves plain 301s to
    curl, which then follows the whole shortener chain (t.co -> trib.al -> site) — so
    curl is the primary resolver and urllib the fallback."""
    try:
        r = subprocess.run(["curl", "-sL", "-o", "/dev/null", "-w", "%{url_effective}",
                            "--max-time", str(timeout), u],
                           capture_output=True, text=True, timeout=timeout + 3)
        out = (r.stdout or "").strip()
        if out.startswith("http") and "//t.co/" not in out:
            return out
    except Exception:
        pass
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(u, method=method,
                                         headers={"User-Agent": "curl/8.4.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.geturl()
        except Exception:
            continue
    return None


async def _context_step(post: dict, pools: Pools, cfg: PostVerifyConfig) -> dict | None:
    """Resolve the post's link, rewrite claims to self-contained form, adjudicate
    self-sourced claims. Mutates post["claims"] in place (c text, cw recovery,
    self_sourced flag); returns the context record for the trace, or None if the
    post has no link. The scraped article text does not leave this function."""
    origins = _origin_domains(post.get("domain") or "")
    links = _TCO.findall(post.get("text") or "")
    if not links:
        return None
    rec = {"link": links[0], "url": None, "domain": None, "origin_link": False,
           "resolved": [], "self_sourced": [], "recovered": [], "failed": None,
           "text_len": 0, "jina": False}
    final, dom, text, saw_social = None, None, None, False
    for link in links[:3]:
        u = await asyncio.to_thread(_final_url, link)
        if not u:
            continue
        d = _domain_of(u)
        if _is_social_host(d):
            saw_social = True
            continue   # media/self links; the article link is what we want
        t = await pools.scrape.scrape(u)
        if not (t and len(t) > 400):
            # gate-free Jina retry (Daniel 2026-07-21): scrape()'s high-value gating is an
            # EVIDENCE-credibility economy; here the page is the outlet's own article and
            # low-NG domains are exactly the ones we need (smoke: Newsmax/OANN recovered)
            j = await asyncio.to_thread(_scrape_jina, u)
            if j and len(j) > len(t or ""):
                disk_cache.set_("scrape", u, {"text": j, "source": "jina"})
                t = j
                rec["jina"] = True
        if t and len(t) > 400:
            final, dom, text = u, d, clean_text(t)
            break
        if final is None:
            final, dom, text = u, d, None   # keep URL for exclusion even if unscrapeable
            rec["text_len"] = max(rec["text_len"], len(t or ""))
    if final is None:
        rec["failed"] = "social-links-only" if saw_social else "no-resolvable-link"
    else:
        rec["url"], rec["domain"] = final, dom
        rec["origin_link"] = any(o in (dom or "") for o in origins)
        if text is None:
            rec["failed"] = "scrape-thin" if rec["text_len"] else "scrape-failed"
        else:
            rec["text_len"] = len(text)
    if rec["failed"]:
        # every CONTEXT miss is inspectable later: reason + resolved URL in the run log
        # AND in the record (rec travels in the trace as record["context"])
        log.warning("[context] %s for %s -> %s (len=%d, jina=%s)", rec["failed"],
                    post.get("url"), rec.get("url"), rec["text_len"], rec["jina"])
        return rec
    # claims in scope: everything the loop would verify, plus "unresolved" claims that a
    # successful rewrite recovers (Daniel 2026-07-20); other filtered categories stay out
    scoped = [c for c in post["claims"]
              if c.get("cw", True) or c.get("cat") == "unresolved"]
    if not scoped:
        return rec
    lines = "\n".join(f"{i}. [{c.get('t') or 'assertion'}] {c['c']}"
                      for i, c in enumerate(scoped, 1))
    site = "the outlet's own site" if rec["origin_link"] else "third-party site"
    user = "\n\n".join([
        f"POST (@{post.get('handle', '?')}, {post.get('date', 'date unknown')}):\n{post['text']}",
        f"LINKED ARTICLE ({final} — {site}):\n{text[:cfg.context_text_cap]}",
        f"CLAIMS:\n{lines}",
        "Output the judgments for every claim."])
    obj = await pools.llm.chat_json(
        [{"role": "system", "content": CONTEXT_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=cfg.context_max_tokens, label="context")
    for row in (obj.get("claims") or []):
        i = _as_cid(row.get("claim_id"), len(scoped))
        if i is None:
            continue
        c = scoped[i - 1]
        r = row.get("resolved")
        if isinstance(r, str) and r.strip() and r.strip() != c["c"]:
            c["c_orig"] = c["c"]
            c["c"] = r.strip()
            rec["resolved"].append({"claim_id": i, "from": c["c_orig"], "to": c["c"]})
            if not c.get("cw", True) and c.get("cat") == "unresolved":
                c["cw"] = True             # referent pinned -> claim is now verifiable
                c["recovered_by_link"] = True
                rec["recovered"].append(i)
        # single-source-of-truth: CODE requires (1) the outlet's own domain, (2) an
        # ATTRIBUTION claim — a world event is never self-sourced, only a saying whose
        # channel is the outlet itself — and (3) the model's channel judgment in
        # {to-this-outlet, own-piece-here}. Two prompt designs measured 2026-07-20 before
        # this one: direct boolean = 29% precision (reporting conflated with venue);
        # free-text venue = worse ("this article" for the outlet's own reporting).
        ch = (row.get("channel") or "").strip().lower()
        if ch in ("to-this-outlet", "own-piece-here") and rec["origin_link"] \
                and c.get("cw", True) and c.get("t") == "attribution":
            c["self_sourced"] = True
            rec["self_sourced"].append(i)
            rec.setdefault("venues", {})[i] = ch
    return rec


QUERY_SYSTEM = """You write the next web search for a fact-check of a news outlet's social-media post. You receive the POST (with its date), the OPEN claims still needing evidence — each with its search budget ("tried n/2"), its type, and a GAP note saying what evidence is missing — and the queries already tried.
Write ONE new keyword query. Target the most AT-RISK open claim — most likely false, most harmful if false, most concretely checkable first — and fold in closely related open claims when one query can serve them; put their ids in "targets". Claims marked "paired with" each other (an attribution "X said Y" and Y as its own claim) usually share one search — coverage of the saying also surfaces evidence on Y itself; target both ids. Chase what the GAP notes point to: a primary/official record, a named person or document, a specific date, an exact figure. Make the query meaningfully different from every prior query — do not reword one. For time-sensitive claims anchor the query to the post's date — never guess a period the post does not state. Compact keyword query, 4-10 words, no operators.
Output JSON: {"targets": [<claim ids>], "query": "<the search query>"}"""

READ_SYSTEM = """You are an evidence pointer for a fact-check. You receive a news article whose sentences are numbered [S1], [S2], ..., plus a social-media POST and the CLAIMS extracted from it.
For each claim that the article speaks to, output the sentence numbers that bear on it and a page-local stance.
Stance definitions (judge ONLY what THIS article says; the claim's EXACT proposition is what counts):
- "supports": the article affirms the claim's exact proposition — every load-bearing part of it.
- "partially-supports": the article affirms part of the claim (one conjunct, an adjacent or weaker fact) but not the full exact proposition. If the claim's exact figure, superlative, quoted words, or causal/agency attribution is absent from this article, the stance is at most partially-supports, even when the surrounding event is fully confirmed.
- "neutral": the article contextualizes the claim — discusses its subject without affirming or contradicting any part of it.
- "partially-refutes": the article contradicts part of the claim, or undercuts it without directly contradicting the full proposition.
- "refutes": the article contradicts the claim's exact proposition. For a quote/attribution claim, refutes requires this article to contradict the SAYING (a denial, a correction, or a different verbatim record of the same statement); the quote merely not appearing here is neutral, NOT refutation.
Pair rule: when a bare claim Y also appears among the claims as reported speech ("X said Y"), this article merely REPORTING that X said Y bears on the attribution claim only — for the bare claim Y it is "neutral", however many outlets repeat the saying. "supports"/"refutes" for Y require the article's OWN voice (its own reporting, records, or data) to affirm or contradict Y itself.
Rules:
- Cite the MINIMAL set of sentences that carry the stance you assign — the specific sentences a fact-checker would highlight as the evidence itself. Do NOT cite surrounding or merely topical context; context is added mechanically later from your pointers.
- Prefer contiguous spans where natural.
- OMIT claims the article says nothing about. Do not paraphrase or quote text. Do not judge overall truth.
- Set "republication" true if this page is substantially the POSTING OUTLET'S OWN reporting relayed — a reprint/mirror of that outlet's article, or a page that merely restates that outlet's report without independent reporting. A page that discusses, investigates, or disputes the outlet's reporting with its own sources is NOT a republication.
Output strictly valid JSON, no other keys:
{"docs": [{"doc": "D1", "republication": <bool>, "evidence": [{"claim_id": <n>, "segs": [<sentence numbers>], "stance": "supports|partially-supports|neutral|partially-refutes|refutes"}]}]}
Cover exactly the doc ids given. A doc with no evidence gets "evidence": []."""

RESOLVE_SYSTEM = """You are the claim resolver of a fact-checking loop, judging the claims of one social-media post by a news outlet. You receive the POST (with its date), the queries already run, and one DOSSIER per claim. An open claim's dossier gives: its status, search budget ("tried n/2"), claim type, a code-computed bar check, and its evidence entries — each with source id, domain, a reliability tag, publication date if known, an advisory reader stance, and source text in which the reader's cited sentences are marked like <<this>>; unmarked sentences are surrounding context. A closed claim's dossier is a one-line resolution (its full evidence is retained outside this prompt) — leave closed claims alone UNLESS their dossier says new contrary evidence arrived, in which case re-judge them.
Reliability tags: PRIMARY (produces the record itself — government, court, statistics agency, intergovernmental org, journal of record); RELIABLE(NG=n) (rated news outlet, higher n = more reliable); UNRATED (no rating — may inform your reasoning and leads but NEVER counts toward closing a claim); UNRELIABLE(NG<60) (never a basis for any status — treat its assertions as unverified). An entry marked OPINION piece is commentary: it can confirm an attribution claim (that someone said something) but never establishes the truth of an assertion.

Re-judge each open (or contrary-flagged) claim YOURSELF from the evidence text. The advisory stance is a hint and may be imperfect — the marked sentences and their context are the ground truth.
Judging rules:
- SAME-EVENT rule: evidence must concern the same event or period the post describes — use the post's date. A similar event at a different date (an earlier incident, a later statement, another month's statistic) neither supports nor refutes the claim; a source published before the claimed event cannot confirm it. If all evidence is off-period, the claim is still "open". The REFUTE direction is time-anchored too: a claim about the state of things at post time — a schedule ("launching soon"), a standing condition ("remains barred"), the existence of reports or rumors, or a superlative over a trailing window ("biggest in 13 months") — is judged AS OF THE POST DATE; evidence marked PUBLISHED AFTER THE POST describing a later development, ruling, confirmation, or data release does not refute what was true when posted.
- ATTRIBUTION claims ("X said/claims Y", "the report states Y"): the proposition is the SAYING, not Y. If credible evidence confirms X said it, the claim is supported — even if Y itself is dubious. This applies ONLY to third parties: the posting outlet's own assertions ARE the post — judge them on substance, never as "the outlet said it".
- PAIR rule: a dossier marked "paired with claim k" is one member of an extraction pair — the attribution "X said Y" and Y as its own bare claim. Judge the members INDEPENDENTLY; their statuses may differ. Evidence that merely reports the saying counts for the attribution member only: for the bare content member it is neither support nor refutation, however many outlets repeat it — the content member closes only on evidence in a source's own voice affirming or contradicting Y itself.
- Do not substitute a weaker proposition: "supported" requires the FULL exact proposition, including any causal or agency attribution ("under pressure from X", "because of Y"). Evidence affirming the event but not the attribution does NOT make the claim supported.
- "refuted" requires credible evidence DIRECTLY contradicting the exact proposition. A slightly different figure, an adjacent time window, a hedged version of the same fact, or a minor label discrepancy is looseness, NOT refutation — such a claim is still supported if its substance holds. For an ATTRIBUTION claim, refutation requires evidence about the SAYING itself — a denial, a correction, or a record of the same statement with materially different substance; the exact words being absent from the evidence gathered is NOT refutation, and a paraphrase carrying the same substance supports rather than contradicts.
Statuses:
- "supported": at least one clearly credible source affirms the claim's EXACT proposition (prefer two independent ones) and nothing credible contradicts it.
- "refuted": credible evidence contradicts the exact proposition.
- "conflicting": credible sources assert INCOMPATIBLE versions of the SAME proposition about the SAME event and period. Evidence about a different period or event, a partial confirmation, or debate about the merits of an attributed statement ("a study claims X" vs critics of the study) is NOT conflict.
- "unsupported": the searches that targeted this claim surfaced no evidence bearing on it (or only evidence that neither affirms nor contradicts it) and further searching is unlikely to help. A claim no query has targeted stays "open".
- "open": evidence is partial or missing AND another, differently-targeted search could plausibly resolve it.
Closure bar (ENFORCED IN CODE — a close below it is refused and the claim stays open; each open dossier shows its own bar check): "supported"/"refuted" requires one READ RELIABLE source with NG>=90, or two independent voices (PRIMARY/institutional and reliable sources both count) on distinct domains of which at least one is READ — a SNIPPET ONLY entry from a PRIMARY or NG>=90 source may serve as the second voice, but snippets can NEVER close a claim by themselves, and a single PRIMARY/institutional source alone does not close a claim. UNRATED, UNRELIABLE, republication-flagged entries, and OPINION pieces (except for attribution claims) do not count toward the bar. If the evidence does not clear it, keep the claim "open".
Independence rule: a source that merely republishes or mirrors the originating outlet's own reporting (same text, wire-style reprint, or explicitly sourced to that outlet) is NOT independent corroboration — weigh it as the outlet's own voice. The same applies to ANY shared upstream voice: the same wire story, the same media group, or the same press release/statement relayed across domains is ONE voice — count independent voices, not domains.
Snippet rule: entries marked SNIPPET ONLY are search-result excerpts, not read sources — useful leads and corroboration hints, never sufficient on their own; many snippets repeating the same wording across domains are syndication of ONE report, one voice.
For every claim you leave "open", write a one-line "gap": the specific missing evidence the next search should chase (a primary record, a named person or document, a date, an exact figure) — not a restatement of the claim, and never a copy of evidence text already in the dossier.
Output strictly valid JSON, no other keys. Emit a row ONLY for a claim whose status you are CHANGING, or an open claim whose gap you are updating in light of this round's evidence. A claim with nothing new — closed and standing, or open with no new evidence and an unchanged gap — gets NO row; the system keeps its state. Output length costs latency; say only what changed:
{"ledger": [{"claim_id": <n>, "status": "supported|refuted|conflicting|unsupported|open", "gap": <string, only for status "open"> }]}"""

RESOLVE_FINAL_NOTE = "\n\nFINAL ROUND: no further searches will run. Resolve what this evidence supports; any claim you leave open will be recorded as unsupported."

TRIAGE_SYSTEM = """You are the triage step of a fact-checking loop. You receive a social-media POST, its OPEN claims, and up to 10 search results (id, domain, date, snippet). Full pages are read only for the results you pick — reading is the expensive step, snippets are free.
Decide two things:
1. "read": which results to read IN FULL, as an ordered priority list (best first). Pick as many as this round needs: a claim can only be closed as supported by credible FULL sources (prefer 2 independent ones) — snippets never suffice — so pick enough to potentially close the open claims. Prefer primary/institutional sources and reliable outlets; skip results that are clearly the same wire story twice (one voice); skip results whose snippet shows the page is not actually about the claims. If NOTHING is relevant (the query missed), return an empty list — that is a valid, useful answer.
2. "snippet_evidence": results whose snippet ALREADY bears on a specific claim (states, contradicts, or gives a key figure for it), whether or not you also picked them to read.
3. "why": ONE short sentence explaining the selection — what made the picked results the right ones (and, if notable, why the rest were skipped). Plain language; this is shown to human reviewers, it does not affect the loop.
Output strictly valid JSON, nothing else:
{"read": [<result ids, best first>], "snippet_evidence": [{"result": <id>, "claim_id": <n>}], "why": "<one sentence>"}"""

EXA_QUERY_SYSTEM = """You compose ONE query for Exa, a NEURAL search engine, as the final escalation of a fact-check whose keyword searches left some claims unresolved. Exa does next-link prediction: it returns the page that would most plausibly be LINKED right after your sentence. So write the sentence a careful researcher would type immediately before pasting the ideal link — a content-rich DECLARATIVE sentence describing the evidence page itself, ending with a colon.
Rules:
- Describe the ideal source, naming its ARCHETYPE: primary reporting, official record/statement, court filing, peer-reviewed study, or fact-check.
- Pack in semantic surface area: the concrete entities, figures, dates, and places from the OPEN claims. Long and specific beats short.
- The prior keyword queries failed — fold their angle in as context, but do not reuse them as keywords.
- No question form, no search operators, no keyword lists.
- EXCEPTION: if an open claim hinges on an EXACT string (a verbatim quote, rare proper noun, document ID, exact figure), include that string verbatim in double quotes inside the sentence.
Example: "Here is the official court record and primary news reporting that confirms or refutes whether [entity] [specific contested fact], including the [figure/date] and the statement \"[exact quote]\":"
Output JSON: {"query": "<one declarative sentence ending with a colon>"}"""

REQUERY_SYSTEM = """The previous Google searches (listed) did not resolve this fact-check's open claims, and the next proposed query merely rewords them — running it would waste a round. Write ONE NEW keyword query that attacks the open claims from a DIFFERENT ANGLE than every prior query: change the entities you name, the phrasing, or the KIND of source you chase (switch toward a primary/official record, a named person or document, a specific date, or an exact figure). Do NOT just add or reorder words from the prior queries. Anchor time-sensitive claims to the post's date. Compact keyword query, 4-10 words, no operators, no quotes.
Output JSON: {"query": "<the new search query>"}"""


async def _diversify_query(pools, cfg, post_hdr, claims, ledger, n, avoid, only=None):
    """One forced, genuinely-different Serper query when STEP's follow-up only reworded a prior
    one (or aimed at budget-exhausted claims). Fed every prior query (`avoid`) so it goes a NEW
    way. `only` restricts the open-claims block to a target subset (v5 rotation)."""
    open_claims = [f"{i}. {claims[i-1]['c']}" for i in range(1, n + 1)
                   if ledger.get(i) == "open" and (only is None or i in only)] \
        or [f"1. {claims[0]['c']}"]
    obj = await pools.llm.chat_json(
        [{"role": "system", "content": REQUERY_SYSTEM},
         {"role": "user", "content": f"{post_hdr}\n\nOPEN CLAIMS:\n" + "\n".join(open_claims)
          + "\n\nPRIOR QUERIES (all failed — go a DIFFERENT way, do not rehash these):\n"
          + "\n".join(f"- {q}" for q in avoid) + "\n\nWrite the new query."}],
        max_tokens=cfg.requery_max_tokens, label="requery")
    return (obj.get("query") or "").strip()

# =============================================================================
# Evidence resolution (pointers -> marked context windows)
# =============================================================================

def resolve_windows(sents: list[str], segs: list[int], ctx: int) -> str:
    """Cited segs -> merged ±ctx windows; cited sentences marked <<...>>, ellipses between gaps."""
    segs = sorted({s for s in segs if 1 <= s <= len(sents)})
    if not segs:
        return ""
    cited = set(segs)
    windows: list[tuple[int, int]] = []
    for s in segs:
        lo, hi = max(1, s - ctx), min(len(sents), s + ctx)
        if windows and lo <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], max(windows[-1][1], hi))
        else:
            windows.append((lo, hi))
    parts = []
    for lo, hi in windows:
        chunk = " ".join(f"<<{sents[i - 1]}>>" if i in cited else sents[i - 1]
                         for i in range(lo, hi + 1))
        parts.append(chunk)
    return " [...] ".join(parts)


def _rel_tag(e: dict) -> str:
    """Render an entry's reliability tag for the evidence table. Empty for entries that
    predate the tagging (old run records replayed through newer code)."""
    r = e.get("rel")
    if not r:
        return ""
    if r == "RELIABLE":
        r = f"RELIABLE(NG={e['ng']:.0f})" if e.get("ng") is not None else "RELIABLE"
    elif r == "UNRELIABLE":
        r = f"UNRELIABLE(NG<{NG_RELIABLE:.0f})"
    return f" | {r}" + (" | OPINION piece" if e.get("opinion") else "")


def evidence_table(entries: list[dict]) -> str:
    if not entries:
        return "(no evidence gathered yet)"
    # Render-side dedupe (v5): a snippet whose domain was later READ in full for the same claim
    # is superseded by the full-read window (95% of measured near-dup clutter, forensics
    # 2026-07-13). Prompt diet only — the guards keep counting over the full evidence list,
    # where _meets_bar's distinct-domain logic already collapses these.
    full_pairs = {(e["domain"], e["claim_id"]) for e in entries if not e.get("snippet_only")}
    entries = [e for e in entries
               if not (e.get("snippet_only") and (e["domain"], e["claim_id"]) in full_pairs)]
    lines = []
    for e in entries:
        date = e.get("date") or "date unknown"
        rel = _rel_tag(e)
        repub = " | FLAGGED: likely republication of the posting outlet's own reporting" \
            if e.get("republication") else ""
        if e.get("snippet_only"):
            srep = " | FLAGGED: mentions the posting outlet — likely its own reporting relayed" \
                if e.get("republication") else ""
            lines.append(f"[{e['src']} | {e['domain']}{rel} | {date} | round {e['round']} | "
                         f"SNIPPET ONLY — search-result excerpt, not a read source{srep}] "
                         f"claim {e['claim_id']}\n{e['text']}")
            continue
        lines.append(f"[{e['src']} | {e['domain']}{rel} | {date} | round {e['round']}{repub}] "
                     f"claim {e['claim_id']} | reader stance (advisory): {e['stance']}\n{e['text']}")
    return "\n\n".join(lines)

# =============================================================================
# Claim dossiers (v6) — the state RESOLVE and QUERY actually see
# =============================================================================
# The prompt diet is filtered; the guards always count over the FULL evidence list.
# Measured basis (ledger study 2026-07-13): closed-claim compression −37% final table,
# + snippet-supersede and the direction-aware cap −56%; re-judging closed claims changed
# the outcome 1/50 posts, so closed claims freeze unless contrary evidence arrives.

_REL_RANK = {"PRIMARY": 0, "RELIABLE": 1, "UNRATED": 2, "UNRELIABLE": 3}
_FINAL_STATUSES = ("supported", "refuted", "conflicting", "unsupported")


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _after_post(entry_date: str | None, post_date: str | None) -> bool:
    """True when the entry's publication date is AFTER the post's date (both ISO-parseable).
    v7: surfaced in the dossier so RESOLVE can apply the as-of-post-date rule in the refute
    direction (2026-07-15 audit: post-dated evidence flipping as-of-date claims was the
    largest wrong-refutation class)."""
    d, p = (entry_date or "")[:10], (post_date or "")[:10]
    return bool(_ISO_DATE.match(d) and _ISO_DATE.match(p) and d > p)


def _entry_line(e: dict, current_round: int = 0, post_date: str | None = None) -> str:
    date = e.get("date") or "date unknown"
    if _after_post(e.get("date"), post_date):
        date += " | PUBLISHED AFTER THE POST"
    rel = _rel_tag(e)
    new = " | NEW this round" if e.get("round") == current_round else ""
    if e.get("snippet_only"):
        srep = " | FLAGGED: mentions the posting outlet — likely its own reporting relayed" \
            if e.get("republication") else ""
        return (f"[{e['src']} | {e['domain']}{rel} | {date} | round {e['round']}{new} | "
                f"SNIPPET ONLY — search-result excerpt, not a read source{srep}]\n{e['text']}")
    repub = " | FLAGGED: likely republication of the posting outlet's own reporting" \
        if e.get("republication") else ""
    return (f"[{e['src']} | {e['domain']}{rel} | {date} | round {e['round']}{new}{repub}] "
            f"reader stance (advisory): {e['stance']}\n{e['text']}")


def _diet(entries: list[dict], k: int) -> list[dict]:
    """Direction-aware per-claim cap for the PROMPT: every refute-direction entry is kept
    (safety-critical, and the ±2/±3 study showed refutation content is the easiest to lose);
    remaining slots go to full-reads over snippets, higher reliability first. A snippet
    superseded by a full read of the same domain is dropped outright."""
    full_doms = {e["domain"] for e in entries if not e.get("snippet_only")}
    entries = [e for e in entries
               if not (e.get("snippet_only") and e["domain"] in full_doms)]
    keep = [e for e in entries if (e.get("stance") or "") in REFUTE_STANCES]
    rest = sorted((e for e in entries if e not in keep),
                  key=lambda e: (bool(e.get("snippet_only")),
                                 _REL_RANK.get(e.get("rel") or "UNRATED", 2),
                                 -(e.get("ng") or 0)))
    keep.extend(rest[:max(0, k - len(keep))])
    order = {id(e): i for i, e in enumerate(entries)}
    return sorted(keep, key=lambda e: order.get(id(e), 0))   # original (round) order


def _pair_note(claims: list[dict], i: int) -> str:
    """' | paired with claim k (type)' when the extractor grouped claim i with siblings
    (attribution + bare content two-axis pairs, v7.4). Empty for ungrouped claims and
    when every sibling was filtered before the loop."""
    g = claims[i - 1].get("g")
    if g is None:
        return ""
    sibs = [j for j, c in enumerate(claims, 1) if j != i and c.get("g") == g]
    if not sibs:
        return ""
    return " | paired with " + ", ".join(
        f"claim {j} ({claims[j - 1].get('t') or 'assertion'})" for j in sibs)


def _bar_line(evidence: list[dict], cid: int, claims: list[dict]) -> str:
    okS, whyS, _ = _meets_bar(_qualifying(evidence, cid, "supported", claims), "supports")
    okR, whyR, _ = _meets_bar(_qualifying(evidence, cid, "refuted", claims), "refutes")
    return (f"bar check — as supported: {'MET — ' if okS else 'not met — '}{whyS}; "
            f"as refuted: {'MET — ' if okR else 'not met — '}{whyR}")


def claim_dossiers(claims: list[dict], ledger: dict[int, str], tries: dict[int, int],
                   cap: int, evidence: list[dict], resolutions: dict[int, str],
                   close_round: dict[int, int], gaps: dict[int, str], k: int,
                   current_round: int = 0, post_date: str | None = None) -> str:
    """Per-claim dossiers for RESOLVE: open claims expanded (bar check + dieted evidence),
    closed claims frozen to their one-line resolution — auto-re-expanded when contrary-
    direction evidence arrives after the close (measured risk of freezing: 1/50)."""
    blocks = []
    for i, c in enumerate(claims, 1):
        st = ledger.get(i, "open")
        typ = c.get("t") or "assertion"
        ents = [e for e in evidence if e.get("claim_id") == i]
        if st in _FINAL_STATUSES and i in resolutions:
            # what counts as contrary: for supported, refute-direction; for refuted,
            # support-direction; for unsupported/conflicting, ANY directional evidence
            want = REFUTE_STANCES if st == "supported" else \
                SUPPORT_STANCES if st == "refuted" else SUPPORT_STANCES | REFUTE_STANCES
            contrary = [e for e in ents if e.get("round", 0) > close_round.get(i, 0)
                        and (e.get("stance") or "") in want]
            if not contrary:
                blocks.append(f"CLAIM {i} [{st} — closed round {close_round.get(i, '?')} | {typ}]: {c['c']}\n"
                              f"resolution: {resolutions[i]}")
                continue
            body = "\n\n".join(_entry_line(e, current_round, post_date) for e in _diet(ents, k))
            blocks.append(f"CLAIM {i} [{st} — closed round {close_round.get(i, '?')} | {typ} | "
                          f"NEW CONTRARY EVIDENCE arrived after the close — re-judge this claim]: {c['c']}\n"
                          f"{_bar_line(evidence, i, claims)}\n{body}")
            continue
        hdr = f"CLAIM {i} [{st} | tried {min(tries.get(i, 0), cap)}/{cap} | {typ}{_pair_note(claims, i)}]: {c['c']}"
        gap = gaps.get(i)
        gline = f"\ngap (your prior note): {gap}" if gap else ""
        if not ents:
            blocks.append(f"{hdr}{gline}\n(no evidence gathered for this claim yet)")
            continue
        body = "\n\n".join(_entry_line(e, current_round, post_date) for e in _diet(ents, k))
        blocks.append(f"{hdr}{gline}\n{_bar_line(evidence, i, claims)}\n{body}")
    return "\n\n".join(blocks)


def _open_claims_block(claims: list[dict], ledger: dict[int, str], tries: dict[int, int],
                       cap: int, gaps: dict[int, str], only: set[int] | None = None) -> str:
    """Compact open-claim headers for QUERY: budget + type + gap, no evidence windows
    (quote-level detail produced echo-shaped queries, not better ones — ledger study)."""
    lines = []
    for i, c in enumerate(claims, 1):
        if ledger.get(i) != "open" or (only is not None and i not in only):
            continue
        lines.append(f"{i}. [tried {min(tries.get(i, 0), cap)}/{cap} | {c.get('t') or 'assertion'}"
                     f"{_pair_note(claims, i)}] {c['c']}\n"
                     f"   gap: {gaps.get(i) or 'no evidence found yet'}")
    return "\n".join(lines) or f"1. {claims[0]['c']}"


def code_verdict(ledger: dict[int, str], claims: list[dict], evidence: list[dict],
                 resolutions: dict[int, str], coerced_open: list[int],
                 cfg: PostVerifyConfig) -> dict:
    """v6: the post verdict is a CODE RULE over claim labels — no verdict LLM call
    (Daniel 2026-07-13: 'if we have the verdicts for the claims, we don't really need a
    verdict step'). nudge iff >= nudge_min_refuted refuted claims; conflicting/unsupported
    concentration thresholds are TBD — the counts ship in the record so the rule can be
    tuned offline without re-running. NOTE: misleading-framing detection is consciously
    NOT covered by this rule (claim labels can't see framing); parked per the Arm-C decision."""
    counts: dict[str, int] = {}
    for st in ledger.values():
        counts[st] = counts.get(st, 0) + 1
    refuted = [cid for cid, st in ledger.items() if st == "refuted"]
    nudge = len(refuted) >= cfg.nudge_min_refuted
    if refuted:
        mt = "FALSE"
    elif counts.get("conflicting"):
        mt = "CONFLICTING"
    elif counts.get("unsupported"):
        mt = "UNSUPPORTED"
    else:
        mt = "NONE"
    parts = []
    for cid in refuted:
        res = resolutions.get(cid) or "refuted (see evidence)"
        parts.append(f"claim {cid} ('{claims[cid - 1]['c'][:90]}') {res}")
    just = "; ".join(parts) if parts else \
        f"no claim refuted ({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))})"
    return {"nudge": nudge, "veracity": None, "misinfo_type": mt, "counts": counts,
            "coerced_open": coerced_open, "justification": just,
            "rule": f"code-v6: nudge iff refuted >= {cfg.nudge_min_refuted}; "
                    f"conflicting/unsupported thresholds TBD"}


# =============================================================================
# The loop
# =============================================================================

def _claims_block(claims: list[dict]) -> str:
    return "\n".join(f"{i}. {c['c']}" for i, c in enumerate(claims, 1))


def _ledger_block(claims: list[dict], ledger: dict[int, str], tries: dict[int, int] | None = None,
                  cap: int = 2) -> str:
    if tries is None:
        return "\n".join(f"{i}. [{ledger.get(i, 'open')}] {c['c']}" for i, c in enumerate(claims, 1))
    return "\n".join(f"{i}. [{ledger.get(i, 'open')} | tried {min(tries.get(i, 0), cap)}/{cap}] {c['c']}"
                     for i, c in enumerate(claims, 1))


async def _read_doc(pools: Pools, post_text: str, claims: list[dict],
                    doc_id: str, url: str, sents: list[str], cfg: PostVerifyConfig,
                    kws: set[str] | None = None, *, outlet: str = "") -> tuple[list[dict], dict]:
    block, _last, rstats = numbered_block(sents, cfg.cap_tok, kws)
    user = "\n\n".join([
        "CLAIMS (for orientation while reading; full task below):\n" + _claims_block(claims),
        f"ARTICLE {doc_id} ({url}):\n{block}",
        f"POST (posted by {outlet or 'unknown outlet'}):\n{post_text}",
        "CLAIMS:\n" + _claims_block(claims),
        f"Output the evidence pointers for each doc. Posting outlet for the republication "
        f"flag: {outlet or 'unknown'}."])
    obj = await pools.llm.chat_json(
        [{"role": "system", "content": READ_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=cfg.read_max_tokens, label=f"read-{doc_id}")
    out = []
    for dd in (obj.get("docs") or []):
        repub = bool(dd.get("republication"))
        for ev in (dd.get("evidence") or []):
            cid = _as_cid(ev.get("claim_id"), len(claims))
            segs = sorted({s for s in (ev.get("segs") or []) if isinstance(s, int)})
            if len(segs) > cfg.max_segs:
                # whole-article marks defeat the pointer design; keep the head (news puts
                # the load-bearing facts first) and log the truncation
                log.info("read %s claim %s: %d segs truncated to %d",
                         doc_id, cid, len(segs), cfg.max_segs)
                segs = segs[:cfg.max_segs]
            if cid is not None and segs:
                out.append({"claim_id": cid, "segs": segs, "republication": repub,
                            "stance": (ev.get("stance") or "neutral").lower()})
    return out, rstats


async def verify_post(post: dict, pools: Pools, cfg: PostVerifyConfig = CFG) -> dict:
    """post: {id?, url?, handle, domain, date?, text, claims: [{c, t?, cw?, g?, cat?}]} — claims
    should be the checkworthy ones (plus "unresolved" ones the CONTEXT step may recover).
    Returns the full record: rounds, evidence, ledger, verdict."""
    # CONTEXT (v7.4): pin referents from the linked article, adjudicate self-sourced claims.
    # Runs BEFORE the cw filter so recovered "unresolved" claims enter the loop.
    context_rec = None
    if cfg.link_context_enabled:
        try:
            context_rec = await _context_step(post, pools, cfg)
        except Exception as e:
            log.warning("context step failed for %s: %s — proceeding without", post.get("url"), e)
            context_rec = {"failed": f"error: {e}"}
    claims = [c for c in post["claims"] if c.get("cw", True)]
    n = len(claims)
    ledger: dict[int, str] = {i: "open" for i in range(1, n + 1)}
    evidence: list[dict] = []
    rounds: list[dict] = []
    guard_events: list[dict] = []
    ng_scores = newsguard_score_map()
    seen_urls: set[str] = set()
    pool_shingles: list[set] = []
    origin = (post.get("domain") or "").strip().lower()
    origins = _origin_domains(origin)   # rebrands/sister brands count as the origin too
    o_toks = sorted({t for o in origins for t in origin_tokens(o, post.get("handle") or "")})
    # the linked article is never evidence: its exact URL never gets read, and a third-party
    # linked DOMAIN is excluded like the origin (Daniel 2026-07-20 — the outlet's chosen
    # source must not corroborate the outlet)
    linked_dom = ""
    if context_rec and context_rec.get("url"):
        seen_urls.add(context_rec["url"])
        d = context_rec.get("domain") or ""
        if d and not context_rec.get("origin_link"):
            linked_dom = d
    excl_domains = [x for x in (*sorted(origins), linked_dom) if x] or None
    post_hdr = f"POST (@{post.get('handle', '?')}, {post.get('date', 'date unknown')}):\n{post['text']}"

    verdict = None
    provider = "serper"
    exa_done = False
    query = ""
    # per-claim budget: tries[cid] = targeted Serper queries spent on that claim
    tries: dict[int, int] = {i: 0 for i in range(1, n + 1)}
    targeted: set[int] = set()
    query_targets: set[int] = set()
    gaps: dict[int, str] = {}          # RESOLVE's per-claim "what's missing" notes -> QUERY
    resolutions: dict[int, str] = {}   # one-line close records -> frozen dossiers + verdict
    resolution_status: dict[int, str] = {}
    close_round: dict[int, int] = {}
    # self-sourced closes (v7.4): the linked page on the outlet's own domain IS the event
    # (their interview/broadcast/publication act) — trivially true, no search spent. A later
    # round's contrary evidence can still reopen it via the frozen-dossier rule.
    for i, c in enumerate(claims, 1):
        if c.get("self_sourced"):
            ledger[i] = "supported"
            resolutions[i] = ("supported — self-sourced: the linked page on the outlet's own "
                              "domain is the venue of this saying/act")
            resolution_status[i] = "supported"
            close_round[i] = 0
    rnd = 0
    stopped = "resolved"
    while True:
        rnd += 1
        if rnd > 2 * n + 3:  # unreachable given the tries accounting; hard backstop
            log.warning("round backstop hit for %s — concluding", post.get("url"))
            stopped = "backstop"
            break
        under = [cid for cid in range(1, n + 1)
                 if ledger.get(cid) == "open" and tries.get(cid, 0) < cfg.tries_per_claim]
        unresolved = any(ledger.get(cid) in ("open", "unsupported") for cid in range(1, n + 1))
        if under:
            # QUERY — one keyword query for the most at-risk under-tried open claim(s)
            prior = [r["query"] for r in rounds]
            q_user = "\n\n".join([
                post_hdr,
                "OPEN CLAIMS (budget, type, and what is missing):\n"
                + _open_claims_block(claims, ledger, tries, cfg.tries_per_claim, gaps, set(under)),
                ("PRIOR QUERIES:\n" + "\n".join(f"- {q}" for q in prior)) if prior else
                "PRIOR QUERIES: none yet — this is the opening query.",
                "Write the next query."])
            q_obj = await pools.llm.chat_json(
                [{"role": "system", "content": QUERY_SYSTEM}, {"role": "user", "content": q_user}],
                max_tokens=cfg.open_max_tokens, label=f"query-r{rnd}")
            query = (q_obj.get("query") or "").strip()
            q_targets = [c for c in (_as_cid(t, n) for t in (q_obj.get("targets") or []))
                         if c is not None and c in under]
            if not q_targets:      # QUERY aimed at exhausted/closed claims: charge the under-tried
                q_targets = under
            if not query:
                query = " ".join(claims[q_targets[0] - 1]["c"].split()[:8])
                log.warning("QUERY returned no query for %s — claim-head fallback", post.get("url"))
            elif prior and _redundant_query(query, prior):
                # a rewording wastes the targets' tries: force one diversified attempt; if the
                # model STILL can't diversify, run it anyway — tries accounting bounds the loop
                alt = await _diversify_query(pools, cfg, post_hdr, claims, ledger, n,
                                             prior + [query], only=set(q_targets))
                if alt and not _redundant_query(alt, prior + [query]):
                    log.info("redundant query (%r) -> forced diversified query (%r)",
                             query[:50], alt[:50])
                    query = alt
            query_targets = set(q_targets)
            for cid in query_targets:  # a targeted keyword query spends one of the claim's tries
                tries[cid] = tries.get(cid, 0) + 1
            try:
                hits = await pools.serper.search_(query, cfg.serper_k,
                                                  date_ceiling=post.get("date") if cfg.date_ceiling else None,
                                                  exclude_domains=excl_domains)
            except Exception:
                hits = []
        elif cfg.exa_enabled and not exa_done and unresolved:
            # ONE escalation for ALL unresolved claims (open or budget-exhausted unsupported)
            provider = "exa"
            query_targets = {i for i in range(1, n + 1)
                             if ledger.get(i) in ("open", "unsupported")}
            open_claims = [f"{i}. {c['c']}" for i, c in enumerate(claims, 1)
                           if i in query_targets] or [f"1. {claims[0]['c']}"]
            exa_obj = await pools.llm.chat_json(
                [{"role": "system", "content": EXA_QUERY_SYSTEM},
                 {"role": "user", "content": f"{post_hdr}\n\nOPEN CLAIMS:\n" + "\n".join(open_claims)
                  + "\n\nPRIOR KEYWORD QUERIES:\n" + "\n".join(f"- {r['query']}" for r in rounds)
                  + "\n\nCompose the neural query."}],
                max_tokens=cfg.exa_query_max_tokens, label="exa-query")
            query = (exa_obj.get("query") or query).strip()
            try:
                hits = await pools.exa.search_(query, cfg.exa_k,
                                               date_ceiling=post.get("date") if cfg.date_ceiling else None,
                                               exclude_domains=excl_domains)
            except Exception as e:
                log.warning("exa search failed (%s) — concluding on current evidence", e)
                hits = []
            exa_done = True
        else:
            # every open claim had its tries and Exa is done/off -> conclude
            unresolved_now = any(ledger.get(cid) in ("open", "unsupported") for cid in range(1, n + 1))
            stopped = "budget-exhausted" if unresolved_now else "resolved"
            break
        targeted.update(query_targets)
        ranked = rank_hits(hits, ng_scores)
        # claim+query terms drive the junk/off-topic gates and READ's sentence selection
        kws = claim_keywords(claims, query)
        # TRIAGE — snippet-informed: which hits to read, how many (0..cap), + snippet evidence
        triage_rec, snippet_ev, read_target = None, [], cfg.pages_per_round
        walk = [(h, "rank") for h in ranked]
        if cfg.triage_enabled and ranked:
            open_claims = [i for i in range(1, n + 1) if ledger.get(i) == "open"] or list(range(1, n + 1))
            res_lines = "\n".join(
                f"[{i}] {_domain_of(h.get('url') or '')} | {h.get('date') or 'date unknown'} | "
                f"{(h.get('snippet') or '(no snippet)')[:260]}"
                for i, h in enumerate(ranked, 1))
            tri = await pools.llm.chat_json(
                [{"role": "system", "content": TRIAGE_SYSTEM},
                 {"role": "user", "content": "\n\n".join([
                     post_hdr,
                     "OPEN CLAIMS:\n" + "\n".join(f"{i}. {claims[i-1]['c']}" for i in open_claims),
                     f"SEARCH RESULTS:\n{res_lines}",
                     "Output the triage decision."])}],
                max_tokens=cfg.triage_max_tokens, label=f"triage-r{rnd}")
            picks = [c for c in (_as_cid(i, len(ranked)) for i in (tri.get("read") or []))
                     if c is not None]
            if picks or (tri.get("read") == []):        # trust an explicit empty list
                # backfill (v2): if a pick dies (junk page, failed scrape, dup), the next
                # ranked hit fills its read slot instead of the round silently shrinking.
                # Provenance travels with each hit — backfill fired in 68/176 audited rounds
                # and an UNTRIAGED hit taking a pick's slot was invisible in the trace.
                ps = set(picks)
                walk = ([(ranked[i - 1], "pick") for i in picks]
                        + ([(h, "backfill") for i, h in enumerate(ranked, 1) if i not in ps]
                           if picks else []))
                read_target = len(picks)  # v5: no cap — triage decides how many reads it needs
            seen_snip: list[set] = []
            for se in (tri.get("snippet_evidence") or []):
                ri = _as_cid(se.get("result"), len(ranked))
                cid = _as_cid(se.get("claim_id"), n)
                if ri is not None and cid is not None and (ranked[ri - 1].get("snippet") or "").strip():
                    h = ranked[ri - 1]
                    snip = h["snippet"].strip()
                    # circularity for snippets: mirrors share ledes — near-identical snippets on
                    # different domains are ONE voice; and an origin-token mention = outlet's voice
                    sh = _shingles(snip, n=5)
                    if any(near_dup(sh, s, thresh=0.6) for s in seen_snip):
                        continue
                    seen_snip.append(sh)
                    repub = any(t in snip.lower() for t in o_toks)
                    dom = _domain_of(h.get("url") or "")
                    rel, ng = _rel_info(dom, h.get("url") or "", ng_scores)
                    # src namespaced by round: bare R2 collided across rounds (audit-confirmed)
                    snippet_ev.append({"src": f"R{rnd}.{ri}", "domain": dom,
                                       "rel": rel, "ng": ng,
                                       "opinion": bool(_OPINION_URL.search(h.get("url") or "")),
                                       "date": h.get("date"), "round": rnd, "claim_id": cid,
                                       "stance": "snippet", "segs": [], "republication": repub,
                                       "snippet_only": True, "text": snip})
            triage_rec = {"read": picks, "snippet_ev": len(snippet_ev),
                          "why": (tri.get("why") or "").strip() or None}
        docs, dom_counts, dropped, fallback = [], {}, [], []
        for h, hsrc in walk:
            if len(docs) >= read_target:
                break
            url = h.get("url") or ""
            dom = _domain_of(url)
            # every discarded candidate is logged with a reason (v2: audit found silent
            # losses shrank rounds to single-voice closes with no trace of why)
            # PRIMARY domains may contribute TWO docs per round (audit 2026-07-13: dup-domain
            # killed 7 primary docs — different records on one .gov can bear on different
            # claims; independence is unaffected, guards count distinct domains)
            dom_limit = 2 if is_primary_source(dom) else 1
            if not url or url in seen_urls or dom_counts.get(dom, 0) >= dom_limit:
                dropped.append({"url": url, "reason": "dup-url" if url in seen_urls else "dup-domain"})
                continue
            if any(o in dom for o in origins):
                dropped.append({"url": url, "reason": "origin"})
                continue
            if linked_dom and linked_dom in dom:
                dropped.append({"url": url, "reason": "linked-source"})
                continue
            # Exa bundles page text; Serper hits are scraped
            txt = h.get("content") or await pools.scrape.scrape(url)
            if not txt or len(txt) <= 400:
                dropped.append({"url": url, "reason": "scrape-failed" if not txt else "too-short"})
                # permissive-admission fallback (audit 2026-07-13): the hit still contributes
                # its short text or SERP snippet as a SNIPPET-ONLY entry — guards keep those
                # from ever closing a claim alone, and scrape failures stop being silent
                fb = (txt or "").strip() or (h.get("snippet") or "").strip()
                if fb:
                    fallback.append((h, dom, fb[:400]))
                continue
            cleaned = clean_text(txt)
            # content-aware junk gate: a junk PHRASE alone no longer condemns a page with
            # real substance (a cosmetic JS banner was killing full congress.gov bills —
            # 8/8 gate-eligible junk drops were .gov false positives, scrape audit 2026-07-12)
            if is_junk(cleaned, kws, _JUNK):
                dropped.append({"url": url, "reason": "junk-page"})
                continue
            if is_off_topic(cleaned, kws):
                dropped.append({"url": url, "reason": "off-topic"})
                continue
            # circularity guard: republications of the origin outlet are not evidence
            if is_republication(cleaned, o_toks, origin, dom):
                dropped.append({"url": url, "reason": "republication"})
                continue
            sh = _shingles(cleaned)
            if any(near_dup(sh, d["shingles"]) for d in docs) or \
               any(near_dup(sh, s) for s in pool_shingles):
                dropped.append({"url": url, "reason": "near-dup"})
                continue
            sents = sentences(cleaned)
            if len(sents) < 4:   # headline/video stubs that clear the char floor
                dropped.append({"url": url, "reason": "too-thin"})
                fb = cleaned.strip()
                if fb:
                    fallback.append((h, dom, fb[:400]))
                continue
            seen_urls.add(url)
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
            docs.append({"id": f"D{len(seen_urls)}", "url": url, "domain": dom, "src": hsrc,
                         "date": h.get("date"), "sents": sents, "shingles": sh})
        pool_shingles.extend(d["shingles"] for d in docs)
        # convert fallback candidates into snippet-only entries bound to this query's targets
        # (open ones) — same dedupe + circularity treatment as triage snippets
        fb_targets = [cid for cid in sorted(query_targets) if ledger.get(cid) == "open"] \
            or sorted(query_targets)
        seen_fb = [_shingles(e["text"], n=5) for e in snippet_ev]
        for h, dom, fb in fallback:
            sh = _shingles(fb, n=5)
            if any(near_dup(sh, s, thresh=0.6) for s in seen_fb):
                continue
            seen_fb.append(sh)
            rel, ng = _rel_info(dom, h.get("url") or "", ng_scores)
            repub = any(t in fb.lower() for t in o_toks)
            for cid in fb_targets:
                snippet_ev.append({"src": f"R{rnd}.f{len(snippet_ev) + 1}", "domain": dom,
                                   "rel": rel, "ng": ng,
                                   "opinion": bool(_OPINION_URL.search(h.get("url") or "")),
                                   "date": h.get("date"), "round": rnd, "claim_id": cid,
                                   "stance": "snippet", "segs": [], "republication": repub,
                                   "snippet_only": True, "auto_snippet": True, "text": fb})
        # READ each doc IN PARALLEL (they were serial — with the triage cap removed a 5-doc
        # round cost ~60s of avoidable wall time; the LLM pool's semaphore governs actual
        # concurrency), then resolve pointers into the shared evidence pool in doc order
        read_results = await asyncio.gather(*(
            _read_doc(pools, post["text"], claims, d["id"], d["url"], d["sents"], cfg, kws,
                      outlet=f"@{post.get('handle', '?')} ({origin})") for d in docs))
        new_entries = []
        for d, (evs, rstats) in zip(docs, read_results):
            d["read"] = rstats
            rel, ng = _rel_info(d["domain"], d["url"], ng_scores)
            opinion = bool(_OPINION_URL.search(d["url"]))
            for ev in evs:
                text = resolve_windows(d["sents"], ev["segs"], cfg.ctx_window)
                if text:
                    new_entries.append({"src": d["id"], "domain": d["domain"], "date": d["date"],
                                        "rel": rel, "ng": ng, "opinion": opinion,
                                        "round": rnd, "claim_id": ev["claim_id"],
                                        "stance": ev["stance"], "segs": ev["segs"],
                                        "republication": ev.get("republication", False),
                                        "text": text})
        evidence.extend(snippet_ev)
        evidence.extend(new_entries)
        # A snippet is a stand-in for a page we could not read. Once the SAME voice has been
        # read in full for the SAME claim, its snippet adds nothing and is dropped from the
        # record (Daniel 2026-07-21). The closing bar already ignored these (full-read domains
        # are subtracted from snippet voices in _meets_bar), so no close changes — this keeps
        # the dossiers and the saved evidence honest about what was actually used.
        read_voices = {(e["claim_id"], _voice_key(e["domain"]))
                       for e in evidence if not e.get("snippet_only")}
        before = len(evidence)
        evidence[:] = [e for e in evidence
                       if not (e.get("snippet_only")
                               and (e["claim_id"], _voice_key(e["domain"])) in read_voices)]
        if len(evidence) < before:
            log.info("dropped %d snippet(s) superseded by a full read", before - len(evidence))
        # RESOLVE — re-judge the open dossiers, update the ledger, note gaps
        final = provider == "exa"   # the Exa round is always the last
        resolve_user = "\n\n".join([
            post_hdr,
            "QUERIES RUN SO FAR:\n" + "\n".join(f"- {r['query']}" for r in rounds) + f"\n- {query}",
            "CLAIM DOSSIERS:\n" + claim_dossiers(claims, ledger, tries, cfg.tries_per_claim,
                                                 evidence, resolutions, close_round, gaps,
                                                 cfg.dossier_entries_cap, current_round=rnd,
                                                 post_date=post.get("date")),
            "Re-judge the open claims and output the ledger rows."
            + (RESOLVE_FINAL_NOTE if final else "")])
        step = await pools.llm.chat_json(
            [{"role": "system", "content": RESOLVE_SYSTEM},
             {"role": "user", "content": resolve_user}],
            max_tokens=cfg.step_max_tokens, label=f"resolve-r{rnd}")
        ev_events, _reopened = _apply_step_ledger(step, ledger, evidence, targeted,
                                                  claims, final, rnd)
        guard_events.extend(ev_events)
        # gap notes for claims left open -> QUERY's briefing next round
        for row in (step.get("ledger") or []):
            cid = _as_cid(row.get("claim_id"), n)
            g = row.get("gap")
            if cid is not None and ledger.get(cid) == "open" and isinstance(g, str) and g.strip():
                gaps[cid] = g.strip()[:200]
        if provider != "exa":
            # budget: an open claim that has spent its targeted tries is closed unsupported by
            # CODE (honest — it WAS targeted that many times); the final Exa round gets one shot
            # at every unresolved claim, and RESOLVE may still flip it if evidence lands.
            for cid in range(1, n + 1):
                if ledger.get(cid) == "open" and tries.get(cid, 0) >= cfg.tries_per_claim:
                    ledger[cid] = "unsupported"
                    guard_events.append({"round": rnd, "guard": "budget-exhausted",
                                         "claim_id": cid, "tries": tries[cid]})
        # resolution one-liners: recorded at close time, freeze the dossier, feed the verdict
        for cid in range(1, n + 1):
            st = ledger.get(cid)
            if st in _FINAL_STATUSES and resolution_status.get(cid) != st:
                if st in ("supported", "refuted"):
                    _ok, why, _snip = _meets_bar(_qualifying(evidence, cid, st, claims),
                                                 _EXACT_STANCE[st])
                    resolutions[cid] = f"{st} — {why}"
                elif st == "conflicting":
                    resolutions[cid] = "conflicting — credible sources genuinely disagree"
                else:
                    resolutions[cid] = (f"unsupported — {tries.get(cid, 0)} targeted search(es) "
                                        f"surfaced no qualifying evidence")
                resolution_status[cid] = st
                close_round[cid] = rnd
        rounds.append({"round": rnd, "provider": provider, "query": query,
                       "targets": sorted(query_targets), "tries": dict(tries),
                       "dropped": dropped, "triage": triage_rec,
                       # visibility only (never fed to any LLM step): the FULL ranked result
                       # list, so unpicked/never-walked hits leave a trace too
                       "results": [{"i": i, "url": h.get("url"), "domain": dom,
                                    "date": h.get("date"),
                                    "snippet": (h.get("snippet") or "")[:200],
                                    "rel": rel, "ng": ng}
                                   for i, h in enumerate(ranked, 1)
                                   for dom in [_domain_of(h.get("url") or "")]
                                   for rel, ng in [_rel_info(dom, h.get("url") or "", ng_scores)]],
                       "docs": [{"id": d["id"], "url": d["url"], "domain": d["domain"],
                                 "date": d["date"], "src": d.get("src"),
                                 "read": d.get("read")} for d in docs],
                       "new_evidence": len(new_entries), "ledger": dict(ledger),
                       "gaps": dict(gaps), "guards": len(ev_events)})
        if provider == "exa":                      # the Exa round is always the last
            stopped = "exa-final"
            break
        if cfg.max_rounds and rnd >= cfg.max_rounds:
            stopped = "round-cap"
            break
    # safety: no opens in the final ledger — but record which were coerced, so analysis can
    # distinguish "no evidence exists" from "we stopped looking" (OQ6 interacts with this)
    coerced_open = [cid for cid, st in ledger.items() if st == "open"]
    for cid in coerced_open:
        ledger[cid] = "unsupported"
        resolutions.setdefault(cid, "unsupported — never resolved within budget (coerced)")
    # verdict: a CODE RULE over the claim labels — no verdict LLM call (v6)
    verdict = code_verdict(ledger, claims, evidence, resolutions, coerced_open, cfg)
    # pair outcome matrix (v7.4): the raw material for attribution-aware nudging — the
    # policy itself stays postponed; per-source stats can already split e.g. "true saying,
    # false content" (accurately reported falsehood) from fabricated attributions
    pair_groups: dict[int, list[int]] = {}
    for i, c in enumerate(claims, 1):
        if c.get("g") is not None:
            pair_groups.setdefault(c["g"], []).append(i)
    pairs = [{"group": g, "members": [{"claim_id": i, "type": claims[i - 1].get("t"),
                                       "status": ledger.get(i)} for i in ids]}
             for g, ids in sorted(pair_groups.items()) if len(ids) > 1]
    return {"id": post.get("id") or post.get("url"), "handle": post.get("handle"),
            "domain": post.get("domain"), "date": post.get("date"), "text": post["text"],
            "claims": [{"c": c["c"], "t": c.get("t"), "claim_id": c.get("claim_id"),
                        **({"c_orig": c["c_orig"]} if c.get("c_orig") else {}),
                        **({"g": c["g"]} if c.get("g") is not None else {}),
                        **({"self_sourced": True} if c.get("self_sourced") else {}),
                        **({"recovered_by_link": True} if c.get("recovered_by_link") else {})}
                       for c in claims],
            "context": context_rec, "pairs": pairs,
            "rounds": rounds, "evidence": evidence,
            "ledger": {str(k): v for k, v in ledger.items()},
            "resolutions": {str(k): v for k, v in resolutions.items()},
            "coerced_open": coerced_open, "verdict": verdict, "guard_events": guard_events,
            "loop_version": LOOP_VERSION, "targeted": sorted(targeted),
            "stopped": stopped, "close_round": {str(k): v for k, v in close_round.items()},
            "tries": {str(k): v for k, v in tries.items()}}
