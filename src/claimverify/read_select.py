"""Relevance-aware evidence reading (copied from pipeline/read_select.py; is_off_topic stub removed): which sentences READ sees, and what counts as junk.

Both concerns are driven by the same signal — the CLAIM KEYWORDS — on two findings from the
scrape audit (2026-07-12, `scripts/scrape_quality_audit.py`):

* **Head-first truncation threw away the useful part of long pages.** 18.9% of read docs exceeded
  the 10k-char READ cap and lost a MEDIAN 56.6% of their body, silently. On a 682,614-char
  congress.gov bill — a PRIMARY source for the very claim being checked — the first 10k chars are
  the table of contents. Keyword-anchored selection reads the relevant sentences instead.

* **The junk filter scanned only `cleaned[:600]` for a phrase**, so the cosmetic banner "Alert: For
  a better experience on Congress.gov, please enable JavaScript" condemned that same bill. 8 of 8
  gate-eligible `junk-page` drops were .gov false positives (6 of them primary sources). A page
  that manifestly talks about the claims is not a CAPTCHA wall: keyword hits prove content.

Primary sources have the highest drop-rate of any tier (23.8%) — the tier the corroboration gate
most wants is the one we destroy most often. This module exists to stop that.
"""
from __future__ import annotations

import re

# Function words carry no retrieval signal. Years are deliberately NOT here — "2026" discriminates.
_STOP = frozenset(
    "a an the of to in on at for and or nor but vs with without is are was were be been being "
    "by from as that this these those it its their his her they them he she we you our your "
    "has have had do does did will would can could should shall may might must "
    "not no than then there here what which who whom whose when where why how "
    "new more most other some any all both each such only own same so too very just "
    "said says say according after before over under about into out up down off again".split())

_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")

SUBSTANCE_HITS = 3   # distinct claim terms that prove a page is genuinely about the claims


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def claim_keywords(claims: list[dict], query: str = "") -> set[str]:
    """Content terms from the post's claims (+ this round's query) — what a page must actually
    talk about to be relevant. Drives BOTH the junk gate and sentence selection."""
    text = " ".join(c.get("c", "") for c in claims)
    if query:
        text += " " + query
    return _tokens(text)


def keyword_hits(text: str, kws: set[str]) -> int:
    """How many DISTINCT claim terms appear in `text`."""
    return len(kws & _tokens(text)) if kws else 0


def is_junk(cleaned: str, kws: set[str], junk_re: re.Pattern) -> bool:
    """True if the page is a block/challenge/paywall wall rather than an article.

    A junk PHRASE alone is NOT enough — it must also lack SUBSTANCE. Otherwise a cosmetic banner
    condemns a full legislative text. Real walls are short and share almost nothing with the claims.
    """
    if not junk_re.search(cleaned[:600]):
        return False
    return keyword_hits(cleaned, kws) < SUBSTANCE_HITS


def numbered_block(sents: list[str], cap_tok: int, kws: set[str] | None = None,
                   lede: int = 8, ctx: int = 2) -> tuple[str, int, dict]:
    """Render sentences for READ as '[S<i>] text', capped at `cap_tok` * 4 chars.

    Indices are ALWAYS the sentence's ORIGINAL position, so READ's [S..] citations still resolve
    against the full `sents` list through resolve_windows() even when sentences are skipped.

    * Fits under the cap  -> everything, in order (unchanged behaviour).
    * Over the cap        -> SELECT by priority, then RENDER in document order:
        1. the LEDE (first `lede` sentences, which carry the topic), but capped at a quarter of
           the budget — on a legislative page the opening sentences are enormous and would
           otherwise eat the whole cap before a single relevant sentence is reached;
        2. keyword-hit sentences, richest first, each with +/-`ctx` sentences of context (a bare
           matching line with no surroundings is unreadable — that is how a related-links
           headline gets mistaken for body prose);
        3. a window that does not fit is SKIPPED, not a stop — a later, cheaper one may fit.
      Skipped spans are marked '[...]'.

    Returns (block, last_index_shown, stats).
    """
    cap_chars = cap_tok * 4
    n = len(sents)
    total = sum(len(s) + 8 for s in sents)

    if total <= cap_chars or not kws:
        out, used, last = [], 0, 0
        for i, s in enumerate(sents, 1):
            if used + len(s) > cap_chars and out:
                break
            out.append(f"[S{i}] {s}")
            used += len(s) + 8
            last = i
        return ("\n".join(out), last,
                {"mode": "head", "kept": last, "total": n, "truncated": last < n})

    keep: set[int] = set()
    used = 0

    # 1. lede, but never more than a quarter of the budget
    lede_cap = cap_chars // 4
    for i in range(1, min(lede, n) + 1):
        cost = len(sents[i - 1]) + 8
        if used + cost > lede_cap:
            break
        keep.add(i)
        used += cost

    # 2. keyword-hit sentences, richest first, each with its context window
    scored = [(len(kws & _tokens(s)), i) for i, s in enumerate(sents, 1)]
    for hits, i in sorted((x for x in scored if x[0]), reverse=True):
        window = [j for j in range(max(1, i - ctx), min(n, i + ctx) + 1) if j not in keep]
        cost = sum(len(sents[j - 1]) + 8 for j in window)
        if used + cost > cap_chars:
            continue                      # too big — a later, cheaper window may still fit
        keep.update(window)
        used += cost

    out, prev = [], 0
    for i in sorted(keep):
        if prev and i > prev + 1:
            out.append("[...]")
        out.append(f"[S{i}] {sents[i - 1]}")
        prev = i
    return ("\n".join(out), prev,
            {"mode": "keyword", "kept": len(keep), "total": n, "truncated": len(keep) < n})
