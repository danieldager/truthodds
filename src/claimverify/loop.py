"""Claim-level verify loop — port of pipeline/verify_tweet_claims.verify_post (v7.5) to ONE
claim per call. Everything claim-level is lifted verbatim (query redundancy, ranking,
scrape gauntlet, READ pointers + context windows, dossiers, corroboration bar, closure
guards, per-claim budget, Exa escalation). Removed: the CONTEXT step, pair logic, the
posting-outlet / origin-token republication machinery, and the post nudge verdict.
The claim is carried as a one-element `claims` list (claim_id 1) so the guard functions
are byte-identical to v7.5.

    per round:  QUERY -> Serper (claim-date ceiling, origin excluded) -> rank
                -> read selection (first `pages_per_round` in rank order, or TRIAGE)
                -> scrape gauntlet -> READ per doc (parallel) -> windows
                -> RESOLVE (dossier -> status, guards enforce the bar)
    routing:    tries_per_claim Serper rounds, then code marks unsupported, then ONE Exa
                round if enabled, then done.

verify_claim(claim, pools, cfg, trace) -> record (see the return at the bottom).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata

from claimverify.config import LOOP_VERSION, ClaimVerifyConfig
from claimverify.credibility import is_factcheck_domain, is_primary_source, is_trusted_factchecker
from claimverify.origin import origin_domain
from claimverify.pools import CallFatal, Outcome, Pools
from claimverify.prompts import (
    EXA_QUERY_SYSTEM, QUERY_SYSTEM, READ_SYSTEM, REQUERY_SYSTEM, RESOLVE_FINAL_NOTE,
    RESOLVE_SYSTEM, RESOLVE_SYSTEM_NO_BAR, TRIAGE_SYSTEM,
)
from claimverify.read_select import claim_keywords, is_junk, numbered_block
from claimverify.retrieval import newsguard_score_map
from claimverify.trace import Trace

log = logging.getLogger("claimverify.loop")

RAW_MAP = {"supported": "Supported", "refuted": "Refuted",
           "unsupported": "Not Enough Evidence",
           "conflicting": "Conflicting Evidence/Cherrypicking"}
CC_MAP = {**RAW_MAP, "unsupported": "Refuted"}   # ClaimCheck: no support found => Refuted

# =============================================================================
# Query redundancy (v7.5)
# =============================================================================

_QUERY_STOPWORDS = frozenset(
    "a an the of to in on at for and or nor but vs with without is are was were be been "
    "by from as that this these those how why what who whom when where which new".split())


def _query_tokens(q: str) -> set[str]:
    return {t for t in q.lower().split() if len(t) > 2 and t not in _QUERY_STOPWORDS}


def _redundant_query(q: str, past: list[str]) -> bool:
    """True only when q introduces NO new content token beyond the prior queries.
    NB: v7.5 defined `_content_tokens` twice and the second (regex, len>3, no stopwords)
    shadowed the first, so this is the tokenizer that actually ran there."""
    if not past:
        return False
    seen: set[str] = set().union(*(_content_tokens(p) for p in past))
    return not (_content_tokens(q) - seen)

# =============================================================================
# Preprocessing
# =============================================================================

_UNI = {"‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ", "…": "..."}
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
    out, seen = [], set()
    for para in text.split("\n"):
        for s in _SENT.split(para):
            s = s.strip()
            key = " ".join(s.split()).lower()
            if s and key not in seen:
                seen.add(key)
                out.append(s)
    return out

# =============================================================================
# Ranking
# =============================================================================

def _domain_of(url: str) -> str:
    m = re.findall(r"https?://([^/]+)", url or "")
    return re.sub(r"^www\.", "", m[0]).lower() if m else ""


NG_ALIASES = {
    "ms.now": "msnbc.com",
    "reutersconnect.com": "reuters.com",
    "dailymail.com": "dailymail.co.uk",
    "arabnews.pk": "arabnews.com",
    "chinadailyhk.com": "chinadaily.com.cn",
}
_NG_PREFIXES = ("m.", "amp.", "mobile.", "news.", "www.")


def _ng_of(dom: str, scores: dict) -> float | None:
    dom = (dom or "").lower().removeprefix("www.")
    s = scores.get(dom)
    if s is None and dom in NG_ALIASES:
        s = scores.get(NG_ALIASES[dom])
    if s is None:
        for pre in _NG_PREFIXES:
            if dom.startswith(pre):
                s = scores.get(dom[len(pre):])
                if s is not None:
                    break
    if s is None and dom.count(".") > 1:
        s = scores.get(".".join(dom.split(".")[-2:]))
    return s


def _source_tier(dom: str, url: str, scores: dict) -> int:
    if is_primary_source(dom) or ".gov/" in (url or ""):
        return 0
    s = _ng_of(dom, scores)
    if s is None:
        return 3
    return 1 if s >= 75 else 2 if s >= 60 else 4


def rank_hits(hits: list[dict], scores: dict | None = None) -> list[dict]:
    """Stable sort by source-quality tier, then provider relevance order."""
    if scores is None:
        scores = newsguard_score_map()
    return sorted(hits, key=lambda h: _source_tier(
        _domain_of(h.get("url") or ""), h.get("url") or "", scores))

# =============================================================================
# Closure guards (v7.5, verbatim)
# =============================================================================

NG_STRONG = 90.0
NG_RELIABLE = 60.0
SUPPORT_STANCES = {"supports", "partially-supports"}
REFUTE_STANCES = {"refutes", "partially-refutes"}
_EXACT_STANCE = {"supported": "supports", "refuted": "refutes"}
SNIPPET_MIN_OVERLAP = 0.15


def _content_tokens(text: str | None) -> set:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", (text or "").lower()) if len(w) > 3}


def _on_topic(snippet_text: str | None, claim_toks: set) -> bool:
    if not claim_toks:
        return False
    return len(claim_toks & _content_tokens(snippet_text)) / len(claim_toks) >= SNIPPET_MIN_OVERLAP


_PUB_FAMILIES = [
    {"thesun.co.uk", "thescottishsun.co.uk", "the-sun.com", "thesun.ie"},
    {"foxnews.com", "foxbusiness.com", "nation.foxnews.com", "radio.foxnews.com"},
    {"nypost.com", "pagesix.com"},
    {"dailymail.co.uk", "mailonsunday.co.uk", "thisismoney.co.uk"},
]


def _voice_key(dom: str) -> str:
    d = (dom or "").lower().removeprefix("www.")
    for fam in _PUB_FAMILIES:
        if d in fam or any(d.endswith("." + f) for f in fam):
            return sorted(fam)[0]
    parts = d.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "net", "org", "ac", "gov", "go", "gob"):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else d


_OPINION_URL = re.compile(r"/(opinions?|commentary|comment|op-eds?|opeds?|editorials?|"
                          r"blogs?|columns?|columnists?|voices|perspectives?)/", re.I)
_WIKIPEDIA = "wikipedia.org"


def _rel_info(dom: str, url: str, scores: dict) -> tuple[str, float | None]:
    if is_primary_source(dom) or ".gov/" in (url or ""):
        return "PRIMARY", None
    if is_trusted_factchecker(dom):
        return "RELIABLE", NG_STRONG
    ng = _ng_of(dom, scores)
    if dom == _WIKIPEDIA or dom.endswith("." + _WIKIPEDIA):
        return "RELIABLE", ng if ng is not None else 75.0
    if ng is None:
        return "UNRATED", None
    return ("RELIABLE", ng) if ng >= NG_RELIABLE else ("UNRELIABLE", ng)


def _as_cid(x, n: int) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, str) and x.strip().isdigit():
        x = int(x.strip())
    if isinstance(x, int) and 1 <= x <= n:
        return x
    return None


_SEG_ID = re.compile(r"^s?(\d+)$", re.I)


def _as_seg(x) -> int | None:
    """Segment id -> int. READ sometimes numbers its pointers "S3" instead of 3."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and (m := _SEG_ID.match(x.strip())):
        return int(m.group(1))
    return None


def _qualifying(evidence: list[dict], cid: int, status: str, claims: list[dict]) -> list[dict]:
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
    full = [e for e in entries if not e.get("snippet_only")]
    if exact is not None and not any((e.get("stance") or "").lower() == exact for e in full):
        return False, f"no full-read '{exact}' anchor — partial/snippet evidence only", []
    strong = sorted({e["domain"] for e in full
                     if e.get("rel") == "RELIABLE" and (e.get("ng") or 0) >= NG_STRONG})
    if strong:
        return True, f"strong secondary (NG>={NG_STRONG:.0f}): {strong}", []
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
                       rnd: int, code_bar: bool = True,
                       gaps: dict[int, str] | None = None) -> tuple[list[dict], list[int]]:
    gaps = gaps or {}
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
            events.append({"round": rnd, "guard": "unsupported-untargeted", "claim_id": cid,
                           "refused": st})
            reopened.append(cid)
            continue
        if st in ("supported", "refuted") and ledger.get(cid) != st:
            if not code_bar:
                want = SUPPORT_STANCES if st == "supported" else REFUTE_STANCES
                same = [e for e in evidence
                        if e.get("claim_id") == cid and not e.get("snippet_only")
                        and not e.get("republication")
                        and (e.get("stance") or "").lower() in want]
                if not same and gaps.get(cid):
                    events.append({"round": rnd, "guard": "close-no-direction-evidence",
                                   "claim_id": cid, "refused": st, "gap": gaps[cid]})
                    if ledger.get(cid, "open") == "open":
                        reopened.append(cid)
                    continue
                events.append({"round": rnd, "guard": "bar-off-accepted",
                               "claim_id": cid, "proposed": st})
            else:
                qual = _qualifying(evidence, cid, st, claims)
                ok, why, snip_used = _meets_bar(qual, _EXACT_STANCE[st])
                if not ok:
                    events.append({"round": rnd,
                                   "guard": "close-unbacked" if not qual else "close-below-bar",
                                   "claim_id": cid, "refused": st, "detail": why})
                    if ledger.get(cid, "open") == "open":
                        reopened.append(cid)
                    continue
                if snip_used:
                    events.append({"round": rnd, "guard": "snippet-corroboration",
                                   "claim_id": cid, "status": st, "snippets": snip_used})
        if st == "conflicting" and ledger.get(cid) != st:
            if not code_bar:
                events.append({"round": rnd, "guard": "bar-off-accepted",
                               "claim_id": cid, "proposed": st})
            else:
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


_JUNK = re.compile(r"(not authorized to access|access denied|page not found|"
                   r"problem finding that (article|page)|enable (javascript|cookies)|"
                   r"subscribe to (continue|read)|to continue reading|are you a robot|"
                   r"verify you are (a )?human|tollbit|captcha|log in to continue)", re.I)


def _shingles(text: str, n: int = 8, span: int = 5000) -> set:
    words = re.sub(r"\W+", " ", text[:span].lower()).split()
    return {" ".join(words[i:i + n]) for i in range(0, max(len(words) - n + 1, 0))}


def near_dup(sh_a: set, sh_b: set, thresh: float = 0.5) -> bool:
    if not sh_a or not sh_b:
        return False
    inter = len(sh_a & sh_b)
    return inter / max(min(len(sh_a), len(sh_b)), 1) >= thresh

# =============================================================================
# Evidence resolution + dossiers (v7.5; post_date -> claim_date wording)
# =============================================================================

def resolve_windows(sents: list[str], segs: list[int], ctx: int) -> str:
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
    r = e.get("rel")
    if not r:
        return ""
    if r == "RELIABLE":
        r = f"RELIABLE(NG={e['ng']:.0f})" if e.get("ng") is not None else "RELIABLE"
    elif r == "UNRELIABLE":
        r = f"UNRELIABLE(NG<{NG_RELIABLE:.0f})"
    return f" | {r}" + (" | OPINION piece" if e.get("opinion") else "")


_REL_RANK = {"PRIMARY": 0, "RELIABLE": 1, "UNRATED": 2, "UNRELIABLE": 3}
_FINAL_STATUSES = ("supported", "refuted", "conflicting", "unsupported")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _after_claim(entry_date: str | None, claim_date: str | None) -> bool:
    d, p = (entry_date or "")[:10], (claim_date or "")[:10]
    return bool(_ISO_DATE.match(d) and _ISO_DATE.match(p) and d > p)


def _entry_line(e: dict, current_round: int = 0, claim_date: str | None = None) -> str:
    date = e.get("date") or "date unknown"
    if _after_claim(e.get("date"), claim_date):
        date += " | PUBLISHED AFTER THE CLAIM"
    rel = _rel_tag(e)
    new = " | NEW this round" if e.get("round") == current_round else ""
    if e.get("snippet_only"):
        return (f"[{e['src']} | {e['domain']}{rel} | {date} | round {e['round']}{new} | "
                f"SNIPPET ONLY — search-result excerpt, not a read source]\n{e['text']}")
    return (f"[{e['src']} | {e['domain']}{rel} | {date} | round {e['round']}{new}] "
            f"reader stance (advisory): {e['stance']}\n{e['text']}")


def _diet(entries: list[dict], k: int) -> list[dict]:
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
    return sorted(keep, key=lambda e: order.get(id(e), 0))


def _bar_line(evidence: list[dict], cid: int, claims: list[dict]) -> str:
    okS, whyS, _ = _meets_bar(_qualifying(evidence, cid, "supported", claims), "supports")
    okR, whyR, _ = _meets_bar(_qualifying(evidence, cid, "refuted", claims), "refutes")
    return (f"bar check — as supported: {'MET — ' if okS else 'not met — '}{whyS}; "
            f"as refuted: {'MET — ' if okR else 'not met — '}{whyR}")


def claim_dossiers(claims: list[dict], ledger: dict[int, str], tries: dict[int, int],
                   cap: int, evidence: list[dict], resolutions: dict[int, str],
                   close_round: dict[int, int], gaps: dict[int, str], k: int,
                   current_round: int = 0, claim_date: str | None = None,
                   code_bar: bool = True) -> str:
    blocks = []
    for i, c in enumerate(claims, 1):
        st = ledger.get(i, "open")
        typ = c.get("t") or "assertion"
        ents = [e for e in evidence if e.get("claim_id") == i]
        if st in _FINAL_STATUSES and i in resolutions:
            want = REFUTE_STANCES if st == "supported" else \
                SUPPORT_STANCES if st == "refuted" else SUPPORT_STANCES | REFUTE_STANCES
            contrary = [e for e in ents if e.get("round", 0) > close_round.get(i, 0)
                        and (e.get("stance") or "") in want]
            if not contrary:
                blocks.append(f"CLAIM {i} [{st} — closed round {close_round.get(i, '?')} | {typ}]: {c['c']}\n"
                              f"resolution: {resolutions[i]}")
                continue
            body = "\n\n".join(_entry_line(e, current_round, claim_date) for e in _diet(ents, k))
            blocks.append(f"CLAIM {i} [{st} — closed round {close_round.get(i, '?')} | {typ} | "
                          f"NEW CONTRARY EVIDENCE arrived after the close — re-judge this claim]: {c['c']}\n"
                          f"{_bar_line(evidence, i, claims) if code_bar else ''}\n{body}")
            continue
        hdr = f"CLAIM {i} [{st} | tried {min(tries.get(i, 0), cap)}/{cap} | {typ}]: {c['c']}"
        gap = gaps.get(i)
        gline = f"\ngap (your prior note): {gap}" if gap else ""
        if not ents:
            blocks.append(f"{hdr}{gline}\n(no evidence gathered for this claim yet)")
            continue
        body = "\n\n".join(_entry_line(e, current_round, claim_date) for e in _diet(ents, k))
        blocks.append(f"{hdr}{gline}\n{_bar_line(evidence, i, claims) if code_bar else ''}\n{body}")
    return "\n\n".join(blocks)


def _open_claims_block(claims: list[dict], ledger: dict[int, str], tries: dict[int, int],
                       cap: int, gaps: dict[int, str]) -> str:
    lines = []
    for i, c in enumerate(claims, 1):
        if ledger.get(i) != "open":
            continue
        lines.append(f"{i}. [tried {min(tries.get(i, 0), cap)}/{cap} | {c.get('t') or 'assertion'}] "
                     f"{c['c']}\n   gap: {gaps.get(i) or 'no evidence found yet'}")
    return "\n".join(lines) or f"1. {claims[0]['c']}"


LENGTH_NUDGE = ("\n\nLENGTH: your previous answer was cut off at the token limit. Answer again "
                "in under {n} tokens — only the required JSON keys, the fewest items that carry "
                "the point, no prose.")


def _is_length_cap(e: BaseException) -> bool:
    return isinstance(e, CallFatal) and e.outcome is Outcome.LENGTH_CAP


async def _chat_json_brief(pools, messages, *, max_tokens, temperature, label, trace,
                           events: list[dict], rnd: int) -> dict | None:
    """chat_json that never lets a length cap fail the claim: on a cap, re-ask ONCE with a
    brevity nudge appended to the system message. A second cap returns None and the caller
    falls back (no close for RESOLVE, the previous query for QUERY)."""
    try:
        return await pools.llm.chat_json(messages, max_tokens=max_tokens,
                                         temperature=temperature, label=label, trace=trace)
    except CallFatal as e:
        if not _is_length_cap(e):
            raise
    nudged = [{**messages[0],
               "content": messages[0]["content"] + LENGTH_NUDGE.format(n=max_tokens // 2)},
              *messages[1:]]
    try:
        obj = await pools.llm.chat_json(nudged, max_tokens=max_tokens, temperature=temperature,
                                        label=f"{label}-brief", trace=trace)
    except CallFatal as e:
        if not _is_length_cap(e):
            raise
        events.append({"round": rnd, "guard": "length-cap", "label": label, "action": "fallback"})
        return None
    events.append({"round": rnd, "guard": "length-cap", "label": label, "action": "re-asked"})
    return obj


async def _diversify_query(pools, cfg, hdr, claims, avoid, trace, events, rnd):
    obj = await _chat_json_brief(
        pools,
        [{"role": "system", "content": REQUERY_SYSTEM},
         {"role": "user", "content": f"{hdr}\n\nOPEN CLAIM:\n1. {claims[0]['c']}"
          + "\n\nPRIOR QUERIES (all failed — go a DIFFERENT way, do not rehash these):\n"
          + "\n".join(f"- {q}" for q in avoid) + "\n\nWrite the new query."}],
        max_tokens=cfg.requery_max_tokens, temperature=cfg.temperature, label="requery",
        trace=trace, events=events, rnd=rnd)
    return (obj.get("query") or "").strip() if obj else ""


async def _read_doc(pools: Pools, claims: list[dict], doc_id: str, url: str,
                    sents: list[str], cfg: ClaimVerifyConfig, kws: set[str] | None,
                    trace: Trace) -> tuple[list[dict], dict, list[dict]]:
    block, _last, rstats = numbered_block(sents, cfg.cap_tok, kws)
    user = "\n\n".join([
        "CLAIM (for orientation while reading; full task below):\n1. " + claims[0]["c"],
        f"ARTICLE {doc_id} ({url}):\n{block}",
        "CLAIM:\n1. " + claims[0]["c"],
        "Output the evidence pointers for each doc."])
    trace.reads.append({"doc": doc_id, "url": url, "block": block, "stats": rstats})
    try:
        obj = await pools.llm.chat_json(
            [{"role": "system", "content": READ_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=cfg.read_max_tokens, temperature=cfg.temperature, label=f"read-{doc_id}",
            trace=trace)
    except CallFatal as e:
        # a capped READ is a failed read, not a failed claim: drop the doc and carry on
        if not _is_length_cap(e):
            raise
        rstats["read_status"] = "length_cap"
        return [], rstats, [{"guard": "length-cap", "label": f"read-{doc_id}", "doc": doc_id,
                             "action": "doc-dropped"}]
    out, events = [], []
    for dd in (obj.get("docs") or []):
        for ev in (dd.get("evidence") or []):
            cid = _as_cid(ev.get("claim_id"), len(claims))
            raw_segs = ev.get("segs") or []
            coerced = [s for s in raw_segs if not isinstance(s, int) and _as_seg(s) is not None]
            segs = sorted({s for s in map(_as_seg, raw_segs) if s is not None})
            if coerced:
                events.append({"guard": "seg-coerced", "doc": doc_id, "raw": coerced})
            if len(segs) > cfg.max_segs:
                log.info("read %s: %d segs truncated to %d", doc_id, len(segs), cfg.max_segs)
                segs = segs[:cfg.max_segs]
            if not segs:
                events.append({"guard": "evidence-dropped-no-segs", "doc": doc_id})
            elif cid is not None:
                out.append({"claim_id": cid, "segs": segs,
                            "stance": (ev.get("stance") or "neutral").lower()})
    return out, rstats, events


# =============================================================================
# The loop
# =============================================================================

async def verify_claim(claim: dict, pools: Pools, cfg: ClaimVerifyConfig,
                       trace: Trace | None = None) -> dict:
    """claim: {claim_id, text, date (ISO), origin_url?, type?, context?}. Returns the full record."""
    t_start = time.monotonic()
    trace = trace or Trace(str(claim.get("claim_id")))
    trace.config = cfg.to_dict()
    claims = [{"c": claim["text"], "t": claim.get("type") or "assertion"}]
    n = 1
    ledger: dict[int, str] = {1: "open"}
    evidence: list[dict] = []
    rounds: list[dict] = []
    guard_events: list[dict] = []
    ng_scores = newsguard_score_map()
    seen_urls: set[str] = set()
    pool_shingles: list[set] = []
    claim_date = claim.get("date")
    origin = origin_domain(claim.get("origin_url")) if cfg.origin_exclusion else None
    excl_domains = [origin] if origin else None
    hdr = f"CLAIM (claimed on {claim_date or 'date unknown'}):\n{claim['text']}"
    ctx = (claim.get("context") or "").strip()   # optional; enters the QUERY block only
    ceiling = (cfg.ceiling_date or claim_date) if cfg.date_ceiling else None

    provider = "serper"
    exa_done = False
    query = ""
    tries: dict[int, int] = {1: 0}
    targeted: set[int] = set()
    query_targets: set[int] = set()
    gaps: dict[int, str] = {}
    resolutions: dict[int, str] = {}
    resolution_status: dict[int, str] = {}
    close_round: dict[int, int] = {}
    counts = {"serper_calls": 0, "exa_calls": 0, "scrapes": 0}
    rnd = 0
    stopped = "resolved"
    while True:
        rnd += 1
        if rnd > 2 * n + 3:
            stopped = "backstop"
            break
        under = [cid for cid in range(1, n + 1)
                 if ledger.get(cid) == "open" and tries.get(cid, 0) < cfg.tries_per_claim]
        unresolved = any(ledger.get(cid) in ("open", "unsupported") for cid in range(1, n + 1))
        if under:
            prior = [r["query"] for r in rounds]
            q_user = "\n\n".join([
                hdr,
                *([f"CONTEXT (circulation of the claim, from a neutral description): {ctx}"]
                  if ctx else []),
                "OPEN CLAIM (budget, type, and what is missing):\n"
                + _open_claims_block(claims, ledger, tries, cfg.tries_per_claim, gaps),
                ("PRIOR QUERIES:\n" + "\n".join(f"- {q}" for q in prior)) if prior else
                "PRIOR QUERIES: none yet — this is the opening query.",
                "Write the next query."])
            q_obj = await _chat_json_brief(
                pools,
                [{"role": "system", "content": QUERY_SYSTEM}, {"role": "user", "content": q_user}],
                max_tokens=cfg.open_max_tokens, temperature=cfg.temperature,
                label=f"query-r{rnd}", trace=trace, events=guard_events, rnd=rnd)
            capped_query = q_obj is None
            if capped_query:
                q_obj = {"query": prior[-1] if prior else ""}
                trace.notes.append({"round": rnd, "note": "length-cap on query -> previous query"})
            query = (q_obj.get("query") or "").strip()
            q_targets = under
            if not query:
                query = " ".join(claims[0]["c"].split()[:8])
                trace.notes.append({"round": rnd, "note": "empty query -> claim-head fallback"})
            elif prior and not capped_query and _redundant_query(query, prior):
                alt = await _diversify_query(pools, cfg, hdr, claims, prior + [query], trace,
                                             guard_events, rnd)
                if alt and not _redundant_query(alt, prior + [query]):
                    trace.notes.append({"round": rnd, "note": "redundant query diversified",
                                        "from": query, "to": alt})
                    query = alt
            query_targets = set(q_targets)
            for cid in query_targets:
                tries[cid] = tries.get(cid, 0) + 1
            counts["serper_calls"] += 1
            try:
                hits = await pools.serper.search_(query, cfg.serper_k, date_ceiling=ceiling,
                                                  exclude_domains=excl_domains, trace=trace)
            except Exception as e:
                trace.notes.append({"round": rnd, "note": f"serper failed: {e}"[:200]})
                hits = []
        elif cfg.exa_enabled and not exa_done and unresolved:
            provider = "exa"
            query_targets = {1}
            exa_obj = await _chat_json_brief(
                pools,
                [{"role": "system", "content": EXA_QUERY_SYSTEM},
                 {"role": "user", "content": f"{hdr}\n\nOPEN CLAIM:\n1. {claims[0]['c']}"
                  + "\n\nPRIOR KEYWORD QUERIES:\n" + "\n".join(f"- {r['query']}" for r in rounds)
                  + "\n\nCompose the neural query."}],
                max_tokens=cfg.exa_query_max_tokens, temperature=cfg.temperature,
                label="exa-query", trace=trace, events=guard_events, rnd=rnd)
            query = ((exa_obj or {}).get("query") or query).strip()
            counts["exa_calls"] += 1
            try:
                hits = await pools.exa.search_(query, cfg.exa_k, date_ceiling=ceiling,
                                               exclude_domains=excl_domains, trace=trace)
            except Exception as e:
                trace.notes.append({"round": rnd, "note": f"exa failed: {e}"[:200]})
                hits = []
            exa_done = True
        else:
            unresolved_now = any(ledger.get(cid) in ("open", "unsupported") for cid in range(1, n + 1))
            stopped = "budget-exhausted" if unresolved_now else "resolved"
            break
        targeted.update(query_targets)
        ranked = rank_hits(hits, ng_scores)
        if cfg.fc_undated_drop:
            fc_undated = [h for h in ranked
                          if not (h.get("date") or "").strip() and is_factcheck_domain(_domain_of(h.get("url") or ""))]
            if fc_undated:
                ranked = [h for h in ranked if h not in fc_undated]
                guard_events.append({"round": rnd, "guard": "fc-undated-dropped",
                                     "urls": [h.get("url") for h in fc_undated]})
        kws = claim_keywords(claims, query)
        triage_rec, snippet_ev, read_target = None, [], cfg.pages_per_round
        walk = [(h, "rank") for h in ranked]
        snippet_pick: list[tuple[int, dict]] = []   # (result index, hit) rows for snippet evidence
        if cfg.triage_enabled and ranked:
            res_lines = "\n".join(
                f"[{i}] {_domain_of(h.get('url') or '')} | {h.get('date') or 'date unknown'} | "
                f"{(h.get('snippet') or '(no snippet)')[:260]}"
                for i, h in enumerate(ranked, 1))
            tri = await _chat_json_brief(
                pools,
                [{"role": "system", "content": TRIAGE_SYSTEM},
                 {"role": "user", "content": "\n\n".join([
                     hdr, f"SEARCH RESULTS:\n{res_lines}", "Output the triage decision."])}],
                max_tokens=cfg.triage_max_tokens, temperature=cfg.temperature,
                label=f"triage-r{rnd}", trace=trace, events=guard_events, rnd=rnd) or {}
            picks = [c for c in (_as_cid(i, len(ranked)) for i in (tri.get("read") or []))
                     if c is not None]
            if picks or (tri.get("read") == []):
                ps = set(picks)
                walk = ([(ranked[i - 1], "pick") for i in picks]
                        + ([(h, "backfill") for i, h in enumerate(ranked, 1) if i not in ps]
                           if picks else []))
                read_target = len(picks)
            for se in (tri.get("snippet_evidence") or []):
                ri = _as_cid(se.get("result"), len(ranked))
                if ri is not None:
                    snippet_pick.append((ri, ranked[ri - 1]))
            triage_rec = {"read": picks, "why": (tri.get("why") or "").strip() or None}
        if cfg.snippet_evidence == "all":
            snippet_pick = [(i, h) for i, h in enumerate(ranked, 1)]
        seen_snip: list[set] = []
        for ri, h in snippet_pick:
            if not (h.get("snippet") or "").strip():
                continue
            snip = h["snippet"].strip()
            sh = _shingles(snip, n=5)
            if any(near_dup(sh, s, thresh=0.6) for s in seen_snip):
                continue
            seen_snip.append(sh)
            dom = _domain_of(h.get("url") or "")
            rel, ng = _rel_info(dom, h.get("url") or "", ng_scores)
            snippet_ev.append({"src": f"R{rnd}.{ri}", "domain": dom, "rel": rel, "ng": ng,
                               "opinion": bool(_OPINION_URL.search(h.get("url") or "")),
                               "date": h.get("date"), "round": rnd, "claim_id": 1,
                               "stance": "snippet", "segs": [], "republication": False,
                               "snippet_only": True, "text": snip})
        if triage_rec is not None:
            triage_rec["snippet_ev"] = len(snippet_ev)
        docs, dom_counts, dropped, fallback = [], {}, [], []
        for h, hsrc in walk:
            if len(docs) >= read_target:
                break
            url = h.get("url") or ""
            dom = _domain_of(url)
            dom_limit = 2 if is_primary_source(dom) else 1
            if not url or url in seen_urls or dom_counts.get(dom, 0) >= dom_limit:
                dropped.append({"url": url, "reason": "dup-url" if url in seen_urls else "dup-domain"})
                continue
            if origin and (dom == origin or dom.endswith("." + origin)):
                dropped.append({"url": url, "reason": "origin"})
                continue
            if h.get("content"):
                txt, src = h["content"], "exa"
            else:
                counts["scrapes"] += 1
                d = await pools.scrape.scrape_detail(url, trace=trace)
                txt, src = d.get("text"), d.get("source")
            page_sha = trace.page(url, txt, src or "?") if txt else None
            if not txt or len(txt) <= 400:
                dropped.append({"url": url, "reason": "scrape-failed" if not txt else "too-short",
                                "page": page_sha})
                fb = (txt or "").strip() or (h.get("snippet") or "").strip()
                if fb:
                    fallback.append((h, dom, fb[:400]))
                continue
            cleaned = clean_text(txt)
            if is_junk(cleaned, kws, _JUNK):
                dropped.append({"url": url, "reason": "junk-page", "page": page_sha})
                continue
            sh = _shingles(cleaned)
            if any(near_dup(sh, d["shingles"]) for d in docs) or \
               any(near_dup(sh, s) for s in pool_shingles):
                dropped.append({"url": url, "reason": "near-dup", "page": page_sha})
                continue
            sents = sentences(cleaned)
            if len(sents) < 4:
                dropped.append({"url": url, "reason": "too-thin", "page": page_sha})
                fb = cleaned.strip()
                if fb:
                    fallback.append((h, dom, fb[:400]))
                continue
            seen_urls.add(url)
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
            docs.append({"id": f"D{len(seen_urls)}", "url": url, "domain": dom, "src": hsrc,
                         "date": h.get("date"), "sents": sents, "shingles": sh,
                         "page": page_sha, "provider": h.get("provider")})
        pool_shingles.extend(d["shingles"] for d in docs)
        seen_fb = [_shingles(e["text"], n=5) for e in snippet_ev]
        for h, dom, fb in fallback:
            sh = _shingles(fb, n=5)
            if any(near_dup(sh, s, thresh=0.6) for s in seen_fb):
                continue
            seen_fb.append(sh)
            rel, ng = _rel_info(dom, h.get("url") or "", ng_scores)
            snippet_ev.append({"src": f"R{rnd}.f{len(snippet_ev) + 1}", "domain": dom,
                               "rel": rel, "ng": ng,
                               "opinion": bool(_OPINION_URL.search(h.get("url") or "")),
                               "date": h.get("date"), "round": rnd, "claim_id": 1,
                               "stance": "snippet", "segs": [], "republication": False,
                               "snippet_only": True, "auto_snippet": True, "text": fb})
        read_results = await asyncio.gather(*(
            _read_doc(pools, claims, d["id"], d["url"], d["sents"], cfg, kws, trace) for d in docs))
        new_entries = []
        for d, (evs, rstats, read_events) in zip(docs, read_results):
            d["read"] = rstats
            guard_events.extend({"round": rnd, **g} for g in read_events)
            rel, ng = _rel_info(d["domain"], d["url"], ng_scores)
            opinion = bool(_OPINION_URL.search(d["url"]))
            for ev in evs:
                text = resolve_windows(d["sents"], ev["segs"], cfg.ctx_window)
                if text:
                    new_entries.append({"src": d["id"], "domain": d["domain"], "date": d["date"],
                                        "rel": rel, "ng": ng, "opinion": opinion,
                                        "round": rnd, "claim_id": ev["claim_id"],
                                        "stance": ev["stance"], "segs": ev["segs"],
                                        "republication": False, "text": text})
        evidence.extend(snippet_ev)
        evidence.extend(new_entries)
        read_voices = {(e["claim_id"], _voice_key(e["domain"]))
                       for e in evidence if not e.get("snippet_only")}
        evidence[:] = [e for e in evidence
                       if not (e.get("snippet_only")
                               and (e["claim_id"], _voice_key(e["domain"])) in read_voices)]
        # the Exa round is always the last; with Exa off (the AVeriTeC arms, which v7.5 has
        # no equivalent of) the last Serper try is, so the final note still gets sent
        final = provider == "exa" or (not cfg.exa_enabled
                                      and tries.get(1, 0) >= cfg.tries_per_claim)
        dossier = claim_dossiers(claims, ledger, tries, cfg.tries_per_claim, evidence,
                                 resolutions, close_round, gaps, cfg.dossier_entries_cap,
                                 current_round=rnd, claim_date=claim_date,
                                 code_bar=cfg.code_bar)
        resolve_user = "\n\n".join([
            hdr,
            "QUERIES RUN SO FAR:\n" + "\n".join(f"- {r['query']}" for r in rounds) + f"\n- {query}",
            "CLAIM DOSSIER:\n" + dossier,
            "Re-judge the claim and output the ledger rows."
            + (RESOLVE_FINAL_NOTE if final else "")])
        trace.resolves.append({"round": rnd, "dossier": dossier, "final": final})
        before = dict(ledger)
        step = await _chat_json_brief(
            pools,
            [{"role": "system", "content": RESOLVE_SYSTEM if cfg.code_bar else RESOLVE_SYSTEM_NO_BAR},
             {"role": "user", "content": resolve_user}],
            max_tokens=cfg.step_max_tokens, temperature=cfg.temperature,
            label=f"resolve-r{rnd}", trace=trace, events=guard_events, rnd=rnd)
        if step is None:                      # capped twice: no close this round
            step = {"ledger": []}
            trace.notes.append({"round": rnd, "note": "length-cap on resolve -> no close"})
        ev_events, _reopened = _apply_step_ledger(step, ledger, evidence, targeted,
                                                  claims, final, rnd, cfg.code_bar, gaps)
        guard_events.extend(ev_events)
        for row in (step.get("ledger") or []):
            cid = _as_cid(row.get("claim_id"), n)
            g = row.get("gap")
            if cid is not None and ledger.get(cid) == "open" and isinstance(g, str) and g.strip():
                gaps[cid] = g.strip()[:200]
        if provider != "exa":
            for cid in range(1, n + 1):
                if ledger.get(cid) == "open" and tries.get(cid, 0) >= cfg.tries_per_claim:
                    ledger[cid] = "unsupported"
                    guard_events.append({"round": rnd, "guard": "budget-exhausted",
                                         "claim_id": cid, "tries": tries[cid]})
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
        trace.ledger.append({"round": rnd, "before": before, "proposed": step.get("ledger"),
                             "after": dict(ledger), "guards": ev_events})
        rounds.append({"round": rnd, "provider": provider, "query": query,
                       "tries": dict(tries), "dropped": dropped, "triage": triage_rec,
                       "results": [{"i": i, "url": h.get("url"), "domain": dom,
                                    "date": h.get("date"), "snippet": h.get("snippet"),
                                    "rel": rel, "ng": ng, "provider": h.get("provider")}
                                   for i, h in enumerate(ranked, 1)
                                   for dom in [_domain_of(h.get("url") or "")]
                                   for rel, ng in [_rel_info(dom, h.get("url") or "", ng_scores)]],
                       "docs": [{"id": d["id"], "url": d["url"], "domain": d["domain"],
                                 "date": d["date"], "src": d.get("src"), "page": d.get("page"),
                                 "read": d.get("read")} for d in docs],
                       "new_evidence": len(new_entries), "ledger": dict(ledger),
                       "gaps": dict(gaps), "guards": len(ev_events)})
        if provider == "exa":
            stopped = "exa-final"
            break
        if cfg.max_rounds and rnd >= cfg.max_rounds:
            stopped = "round-cap"
            break
    coerced_open = ledger.get(1) == "open"
    if coerced_open:
        ledger[1] = "unsupported"
        resolutions.setdefault(1, "unsupported — never resolved within budget (coerced)")
    status = ledger[1]
    usage = {"prompt": 0, "cached": 0, "completion": 0}
    cost = 0.0
    for c in trace.llm_calls:
        u = c.get("usage") or {}
        for k in usage:
            usage[k] += int(u.get(k) or 0)
        cost += float(u.get("cost_usd") or 0.0)
    return {"claim_id": claim.get("claim_id"), "text": claim["text"], "date": claim_date,
            "origin_url": claim.get("origin_url"), "origin_excluded": origin,
            "status": status, "verdict_raw": RAW_MAP[status], "verdict_cc": CC_MAP[status],
            "resolution": resolutions.get(1), "coerced_open": coerced_open,
            "rounds": rounds, "evidence": evidence, "guard_events": guard_events,
            "stopped": stopped, "tries": tries[1], "close_round": close_round.get(1),
            "gaps": gaps.get(1),
            "llm_calls": len(trace.llm_calls),
            "llm_cache_hits": sum(1 for c in trace.llm_calls if c.get("cache_hit")),
            "serper_calls": counts["serper_calls"], "exa_calls": counts["exa_calls"],
            "scrapes": counts["scrapes"], "tokens": usage, "cost_usd": round(cost, 6),
            "elapsed_s": round(time.monotonic() - t_start, 3), "loop_version": LOOP_VERSION}
