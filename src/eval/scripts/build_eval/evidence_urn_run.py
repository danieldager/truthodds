"""E1 runner — Truth Odds urn measurement on CAL (program §4, READ paradigm §2).

Inputs are the CONTEXTUALIZED draw (2026-08-04): CAL joined with
claim_context_cal.parquet (x_date / x_context / context_ok) and
claim_resolution.parquet (claim_resolved, resolution_status), gated by
e1_gate_decisions.json (27-row media/demonstrative hand-read). Excluded rows are
RECORDED with an "excluded" reason and empty results — never silenced.

Per claim: ONE query (query-v3: resolved claim + usable context + claim date) →
Serper top-10 under the claim-date ceiling (x_date fallback, review−1 last) with
FC-block + reviewing-publisher exclusion →
read ALL results (scrape → Jina inside scrape() → snippet fallback; no triage) →
per-doc pointer-READ (read-v5, Daniel 2026-08-04: E1 runs v5 directly, no v4
pass): {direction: 5|4|3|2|1|X|I, evidence:[sent ids], reason (I only)} over
sentence-numbered text, plus the flag token's logprob when the provider returns
it (soft per-doc vote, read_v5_prompts docstring).
Code-side per result: NG tier, voice key, mirror flag, provenance, leak flag.
Docs are SAVED sentence-split so pointers reconstruct evidence exactly.

Cost controls: rich constant prefix [system+claim] + prompt_cache_key=claim key
(DeepInfra caching, verified ~18% price on cached tokens); reads are SEQUENTIAL
within a claim (first read primes the cache) and PARALLEL across claims; hard
budget meter from usage.estimated_cost with projection abort.

  uv run python -m eval.scripts.build_eval.evidence_urn_run --smoke            # 25 claims
  uv run python -m eval.scripts.build_eval.evidence_urn_run --budget 10        # E1-lite (GATED)

Output: eval/data/urn_runs/e1_ctx/results-*.jsonl (resumable by review_url)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, VERIFICATION_MODEL
from pipeline.config import FACT_CHECK_DOMAINS
from pipeline.credibility import TRUSTED_FACTCHECKERS
from pipeline.search import SearchError, newsguard_score_map, scrape, search
from pipeline.verify_tweet_claims import _domain_of, _rel_info, _voice_key
from eval.scripts.build_eval.claim_modes import (
    MODE_DEFS, MODE_RULES_QUERY, mode_of)

CAL = Path("eval/data/truthodds_cal.parquet")
CTX = Path("eval/data/claim_context_cal.parquet")
RES = Path("eval/data/claim_resolution.parquet")
GATE = Path("eval/data/e1_gate_decisions.json")
SCREEN = Path("eval/data/claim_screen_e1.parquet")   # TAG only — see run_claim
MODEF = Path("eval/data/claim_mode_final.parquet")   # resolved mode, every claim
# e1/ holds the July pilot (contaminated instrument, comparison baseline only).
# The contextualized Thursday run writes to its own dir so resume can never
# treat pilot rows as done.
OUTDIR = Path("eval/data/urn_runs/e1_ctx")

QUERY_PROMPT_V = "query-v3"  # v3 = resolved claim + context (context_ok only) + date.
                             # v4 adds the claim MODE, sharing one definition block with
                             # READ via claim_modes.py. Cached queries from earlier
                             # versions are INVALID — regeneration is mandatory.
READ_PROMPT_V = "read-v5"   # PRODUCTION reader = read-v5 (READ_SYS_MODE, seven-class
                            # 5/4/3/X/I/2/1, hash 92404a300e14): the FROZEN instrument
                            # (Daniel 2026-09-21, clog/210926.md). read-v6.1 (six-class,
                            # 2026-09-14) stays selectable via --reader read-v6.1.
                            # History of read-v5: v5 + claim MODE (attribution vs assertion), Daniel
                            # 2026-08-04 after the tranche-1 audit: 2 of 9 instrument
                            # bugs were documents that merely RELAYED an assertion
                            # being read as supporting it. v5.2 restates the
                            # attribution rule twice: v5.1 collided with the base
                            # rubric's "I" (X 31->0, AUC .848->.800); v5.2 licensed
                            # absence-as-refutation (refutes on gold-TRUE 2->13,
                            # AUC .878->.692). v5.3 states only what the mode
                            # CHANGES and leaves refutation to the base rubric.
                            # direction-FIRST is still load-bearing (p=0.022).
CLEAN_V = "clean-v2"
PREP_V = "prep-v7"   # BM25-mass-ranked CONTIGUOUS regions; K set by MAX_REGIONS (now 1)

MAX_SENTS = 50          # sentence cap per doc fed to READ
# A unit longer than this is EXCLUDED and recorded (not truncated — see sentences()).
# Raised 320 -> 800 on 2026-07-27: once the splitter began breaking on line breaks,
# 320 no longer separated artifacts from prose — legal and academic sentences run
# 300-500 chars, and exclusion is harsher than the truncation it replaced. At 800 only
# genuine artifacts (minified tables, link farms, JSON blobs) are caught.
MAX_SENT_CHARS = 800
REGION_CHARS = 8000     # ~2k tokens per region (the detection-optimal length, cap sweep)
MAX_REGIONS = 1         # K: read slots per doc. select_regions ranks by BM25 mass, so K=1
                        # reads the single best chunk. Daniel 2026-07-30, reversing the
                        # 07-25 cap-sweep choice of K=3: measured conditionally on document
                        # length, the extra regions on 20k-60k docs add 31% more directional
                        # votes but they are right only 56% of the time, and dropping them
                        # raises alignment on that band from 81.8% to 96.2%. Claim-level AUC
                        # 0.7848 (K=3) vs 0.7842 (K=1) out of fold, for 40% fewer read calls.
MAX_DOC_CHARS = REGION_CHARS  # kept for select_sentences compat (sweep imports)
SNIPPET_MIN_TEXT = 200  # scraped text shorter than this -> snippet fallback

QUERY_SYS = ("You write ONE web search query to check a factual claim. Identify the single "
             "LOAD-BEARING proposition — the thing that makes the claim true or false — and "
             "query THAT alone; ignore secondary details, framing, and other clauses of a "
             "multi-part claim. At most 10 words. The query should surface independent "
             "reporting or primary evidence, not commentary about whether the claim is true. "
             "No quotes unless a distinctive phrase is essential. "
             "Respond JSON only: {\"query\": \"...\"}")

# --- e4 mid-band fix hooks (Daniel 2026-09-11, clog/110926.md). ALL DEFAULT FALSE.
# With every flag false this module's behaviour and every prompt byte are the pinned
# production path (e1_ctx, averitec, fc_gold); keyclaim_urn_run --fix turns them on.
# Diagnosed on 12 worst-scoring mid-band class-1 posts: the date ceiling is sent to
# Serper but never enforced on the result (leak_flag was recorded and ignored), the
# reader is never told a document's date so read-v5's own "mind the dates" rule can
# never fire, and query-v3's "single LOAD-BEARING proposition" rule strips the
# speaker out of an "X said Y" claim so retrieval is about Y.
FIX = {"drop_leaks": False,   # documents dated after the ceiling are not read
       "doc_dates": False,    # the document's publication date reaches READ
       "attr_query": False,   # attribution claims query the utterance, not the content
       "read_asof": False}    # READ_SYS_ASOF replaces the base rubric's one date line

# NEW CONSTANT, NEW NAME: query-v3 is untouched. v3a adds one rule and nothing else.
QUERY_SYS_V3A = QUERY_SYS.replace(
    "No quotes unless a distinctive phrase is essential. ",
    "If the claim is that a named person or body SAID, claimed, wrote, argued, "
    "announced or denied something, the load-bearing proposition is the UTTERANCE: "
    "keep the speaker in the query and target the words they used, NOT whether what "
    "they said is true. Use ONLY names that appear in the claim or its context; never "
    "introduce a name they do not contain. "
    "No quotes unless a distinctive phrase is essential. ")
QUERY_PROMPT_V3A = "query-v3a"

# NOT `states` or `writes`: "the United States" made every US claim an attribution
# claim in the first smoke (12 claims, 1 false positive).
_ATTR_VERB = re.compile(r"\b(said|says|claimed|claims|argued|argues|insisted|insists|"
                        r"announced|announces|stated|wrote|told|tells|"
                        r"denied|denies|vowed|vows|alleged|alleges|accused|accuses|"
                        r"warned|warns|according to)\b", re.I)


def is_attribution(row, claim: str) -> bool:
    """The claim is about an utterance. `claim_type` alone is not enough: the mid-band
    extractor typed "Trump said '<quote>'" as an assertion, so the verb decides too."""
    return ((row.get("claim_type") or "").strip() == "attribution"
            or bool(_ATTR_VERB.search(claim or "")))


from eval.scripts.build_eval.read_v5_prompts import READ_SYS_ASOF  # noqa: E402
from eval.scripts.build_eval.read_v5_prompts import READ_SYS_MODE, READ_SYS_V6_1  # noqa: E402
from eval.prompt_hash import prompt_hash  # noqa: E402

# --reader selects the read prompt. read-v5 (READ_SYS_MODE, seven-class 5/4/3/X/I/2/1)
# is the FROZEN production instrument and the DEFAULT (Daniel 2026-09-21); it is what
# reader_lab --prompt v5 selected for the 17 Sep survey re-read (hash 92404a300e14).
# read-v6.1 (READ_SYS_V6_1, six-class, the 2026-09-14 experiment) stays selectable via
# --reader read-v6.1.
# Neither prompt's text changes here; the switch only picks which one and restamps
# READ_PROMPT_V / PROMPT_HASHES. The seven vs six flag set is carried by the prompt itself
# (v5 offers the "3" button, v6.1 does not) and priced downstream by graded_urn.
READERS = {"read-v6.1": (READ_SYS_V6_1, "read-v6.1"),
           "read-v5": (READ_SYS_MODE, "read-v5")}
READ_SYS = READ_SYS_MODE        # default reader (read-v5, frozen); main() may reassign per --reader

PROMPT_HASHES = {"query": prompt_hash(QUERY_SYS), "read": prompt_hash(READ_SYS)}

DIRECTIONS = {"5", "4", "3", "2", "1", "X", "I"}

# Serper window MEASURED (burst test 2026-07-25): rolling short window (remaining
# recovers within seconds) — NOT the misdiagnosed 500/hour. 1 req/s uses ~12% of the
# observed window; backfill dropped so 1 search = 1 request.
_SERPER_MIN_INTERVAL = 0.02  # 50 q/s ceiling (Daniel 2026-08-26); plan allows it, spend is per-request not per-second
_serper_lock = threading.Lock()
_serper_last = [0.0]


def _serper_gate():
    while True:
        with _serper_lock:
            now = time.time()
            if now - _serper_last[0] >= _SERPER_MIN_INTERVAL:
                _serper_last[0] = now
                return
            wait = _SERPER_MIN_INTERVAL - (now - _serper_last[0])
        time.sleep(wait)

# Splits on sentence punctuation AND on any line break. The old pattern required a
# BLANK line (\n{2,}), so single-newline-separated lines were glued into one numbered
# "sentence": 34.6% of the units shown in the E1 pilot contained embedded newlines,
# and only their first line could be cited (the rest reached the model with no id).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9À-Ý«\"'‘“])|(?<=[。！？])|\n+")
# CJK text carries no spaces, so the token-count prose bar in _clean and the
# [.!?]-only splitter both zero it out (cf_probe ja smoke: 11/20 empty-doc on
# fetched pages). CJK-dense lines are judged by characters instead.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URLISH = re.compile(r"https?://\S+|#\S+")


def _wordlike(tok: str) -> bool:
    # clean-v2 (audit 2026-07-24): >=2 letters AFTER stripping non-letters — the old
    # [A-Za-z]{3} run never matched contractions (it's/we're), so grammatical QUOTES
    # and terse verdict sentences failed the prose bar. That lost probative lines.
    return sum(c.isalpha() for c in tok) >= 2


def _clean(text: str) -> str:
    """Strip markdown/nav junk; clean-v2 keeps quotes, terse verdicts, dialogue turns."""
    t = _MD_IMG.sub(" ", text or "")
    t = _MD_LINK.sub(r"\1", t)
    t = _URLISH.sub(" ", t)
    lines = []
    for ln in t.splitlines():
        w = ln.split()
        if not w:
            continue
        ratio = sum(1 for x in w if _wordlike(x)) / len(w)
        cjk = len(_CJK.findall(ln))
        keep = (len(w) >= 4 and ratio >= 0.6) or \
               (ln.strip()[-1:] in ".!?\u201d\"'" and len(w) >= 2 and ratio >= 0.5) or \
               (cjk >= 10 and cjk / len(ln.strip()) >= 0.3)
        if keep:
            lines.append(ln.strip())
    return "\n".join(lines)


def sentences(text: str, stats: dict | None = None) -> list[str]:
    """Split into numbered units, EXCLUDING any unit over MAX_SENT_CHARS.

    Oversized units used to be silently truncated mid-word (6.8% of units, 64% of docs
    in the E1 pilot) — the reader could not tell it was seeing half a clause. Truncation
    is now dropped entirely: an over-long unit is almost always an extraction artifact
    (a minified table, a link farm, a JSON blob), so it is EXCLUDED and recorded instead
    of being fed to the model as mangled text. `stats` collects them for inspection —
    after the first run we decide whether excluding is right or whether they need a
    dedicated fix (Daniel 2026-07-27)."""
    out, dropped = [], []
    for x in _SENT_SPLIT.split(_clean(text)):
        x = x.strip()
        if not x or len(x) <= 2:
            continue
        if len(x) > MAX_SENT_CHARS:
            dropped.append(x)
            continue
        out.append(x)
    if stats is not None:
        stats["oversize_dropped"] = len(dropped)
        stats["oversize_chars"] = sum(len(d) for d in dropped)
        stats["oversize_samples"] = [d[:160] for d in dropped[:3]]
    return out


_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'-]*")
# generic claim-vocabulary that must never anchor a window (prep-v5)
_ANCHOR_STOP = {"said", "says", "video", "photo", "image", "picture", "claim", "claims",
                "claimed", "people", "president", "government", "shows", "show", "post",
                "posted", "viral", "online", "media", "report", "reports", "reported",
                "news", "year", "years", "time", "times", "week", "state", "country",
                "world", "million", "billion", "percent", "according", "during", "after",
                "before", "about", "would", "could", "there", "their", "which", "where"}


def _anchor_terms(text: str) -> set:
    """prep-v5 anchor vocabulary: keep numbers and short capitalized tokens (acronyms,
    proper-noun initials: 5G, EU, WHO), drop generic claim vocabulary; lowercase keys."""
    out = set()
    for w in _WORD.findall(text or ""):
        lw = w.lower()
        if lw in _ANCHOR_STOP:
            continue
        if any(c.isdigit() for c in w) or (len(w) <= 3 and w.isupper()) or len(lw) > 3:
            out.add(lw)
    return out


def _terms(text: str) -> set:   # kept for READ-side QC uses
    return {w.lower() for w in _WORD.findall(text or "") if len(w) > 3}


def select_sentences(text: str, claim: str, query: str = "") -> tuple[list[int], list[str], dict]:
    """prep-v4 (MAXIMAL): whole doc if it fits; else anchor on claim-term sentences,
    then GROW the neighbor radius (±1, ±2, ±3, ...) around all anchors until the
    char/sentence budget is saturated — fill the context we're paying for, learn the
    right radius from pointer-utilization data later. Lede always kept. ORIGINAL
    indices preserved. Returns (ids, texts, prep_meta) for the processing observatory."""
    parts = sentences(text)
    n = len(parts)
    meta = {"n_sents_doc": n, "doc_chars": sum(len(x) for x in parts)}
    if meta["doc_chars"] <= MAX_DOC_CHARS and n <= MAX_SENTS:
        meta.update({"windowed": False, "anchors": None, "radius": None, "coverage": 1.0})
        return list(range(1, n + 1)), parts, meta
    ct = _anchor_terms(claim) | _anchor_terms(query)     # the query IS the distilled proposition
    # prep-v6: sentence-level BM25 (docs = this page's sentences) — the AVeriTeC-
    # canonical selector; consolidates the old rarity heuristic into principled IDF.
    from collections import Counter as _C
    from math import log as _log
    sent_terms = [_anchor_terms(x) for x in parts]
    df_ = _C()
    for st_ in sent_terms:
        df_.update(st_ & ct)
    avg_len = sum(len(x) for x in sent_terms) / max(n, 1)
    K1, B = 1.2, 0.75
    def _bm25(i):
        sc = 0.0
        for t in (sent_terms[i] & ct):
            idf = _log(1 + (n - df_[t] + 0.5) / (df_[t] + 0.5))
            tf = 1.0  # sets: presence-based tf (sentences are short)
            sc += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * len(sent_terms[i]) / max(avg_len, 1)))
        return sc
    scores = [_bm25(i) for i in range(n)]
    anchors = sorted((i for i in range(n) if scores[i] > 0),
                     key=lambda i: -scores[i])
    keep = set(range(min(3, n))) | set(anchors)

    def size(sel):
        return sum(len(parts[j]) for j in sel)

    radius = 0
    while size(keep) < MAX_DOC_CHARS and len(keep) < MAX_SENTS and len(keep) < n:
        radius += 1
        grown = set(keep)
        for a in anchors:
            for j in (a - radius, a + radius):
                if 0 <= j < n:
                    grown.add(j)
        if grown == keep:      # anchors exhausted -> pad by BM25 score order (then doc order)
            order = sorted((j for j in range(n) if j not in keep),
                           key=lambda j: (-scores[j], j))
            if not order:
                break
            grown.add(order[0])
        keep = grown
    sel, total, out = sorted(keep), 0, []
    for j in sel:
        if total + len(parts[j]) > MAX_DOC_CHARS or len(out) >= MAX_SENTS:
            break
        out.append(j)
        total += len(parts[j])
    meta.update({"windowed": True, "anchors": len(anchors), "radius": radius,
                 "coverage": round(len(out) / max(n, 1), 3)})
    return [j + 1 for j in out], [parts[j] for j in out], meta


_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4,
                                                         pool_maxsize=32, max_retries=3))


def select_regions(text: str, claim: str, query: str = ""):
    """prep-v7: contiguous candidate regions (<= REGION_CHARS each) ranked by summed
    BM25 mass; top MAX_REGIONS returned in DOCUMENT ORDER. Whole doc if it fits."""
    parts = sentences(text, stats := {})
    n = len(parts)
    total = sum(len(x) for x in parts)
    meta = {"n_sents_doc": n, "doc_chars": total, "k": None, **stats}
    if total <= REGION_CHARS and n <= MAX_SENTS:
        ct0 = _anchor_terms(claim) | _anchor_terms(query)
        meta.update({"windowed": False, "regions": 1, "coverage": 1.0,
                     "anchor_vocab": len(ct0),
                     "anchor_sents": sum(1 for x in parts if ct0 & _anchor_terms(x)),
                     "mass_read_share": 1.0, "skipped_tiles": 0})
        return [(list(range(1, n + 1)), parts)], meta
    # score sentences via the same BM25 as select_sentences (inline)
    from collections import Counter as _C
    from math import log as _log
    ct = _anchor_terms(claim) | _anchor_terms(query)
    sent_terms = [_anchor_terms(x) for x in parts]
    df_ = _C()
    for st_ in sent_terms:
        df_.update(st_ & ct)
    avg_len = sum(len(x) for x in sent_terms) / max(n, 1)
    def _bm(i):
        sc = 0.0
        for t in (sent_terms[i] & ct):
            idf = _log(1 + (n - df_[t] + 0.5) / (df_[t] + 0.5))
            sc += idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * len(sent_terms[i]) / max(avg_len, 1)))
        return sc
    scores = [_bm(i) for i in range(n)]
    # partition doc into contiguous regions of <= REGION_CHARS
    regions, i = [], 0
    while i < n:
        j, tot = i, 0
        while j < n and tot + len(parts[j]) <= REGION_CHARS:
            tot += len(parts[j]); j += 1
        if j == i:
            j = i + 1
        regions.append((i, j, sum(scores[k] for k in range(i, j))))
        i = j
    ranked = sorted(regions, key=lambda r: -r[2])
    top = sorted(ranked[:MAX_REGIONS])                 # document order for execution
    skipped = ranked[MAX_REGIONS:]
    lede_ids = list(range(1, min(3, n) + 1))
    out = []
    for a, b, _sc in top:
        ids = list(range(a + 1, b + 1))
        sel = parts[a:b]
        if a > 2:                                      # prepend labeled lede context
            ids = lede_ids + ids
            sel = parts[:min(3, n)] + sel
        out.append((ids, sel))
    total_mass = sum(r[2] for r in regions) or 1.0
    meta.update({
        "windowed": True, "regions": len(out), "tiles_total": len(regions),
        "coverage": round(sum(b - a for a, b, _ in top) / max(n, 1), 3),
        "anchor_vocab": len(ct),
        "anchor_sents": sum(1 for st_ in sent_terms if st_ & ct),
        "read_regions": [{"span": [a + 1, b], "chars": sum(len(parts[k]) for k in range(a, b)),
                          "mass": round(sc, 3)} for a, b, sc in top],
        "mass_read_share": round(sum(sc for _, _, sc in top) / total_mass, 3),
        "skipped_top_mass": [round(sc, 3) for _, _, sc in skipped[:5]],
        "skipped_tiles": len(skipped),
    })
    return out, meta


def _is_leak(raw_date, ceil):
    """Serper dates are human-format ('Apr 8, 2026'); the old lexical compare vs ISO
    flagged 86% of results (letters sort above digits). Parse properly; unparseable
    or missing dates are NOT flagged (unknown ≠ leak)."""
    if not (raw_date and ceil):
        return False
    import datetime as _dt
    t = str(raw_date).strip()
    for fmt in ("%b %d, %Y", "%d %b %Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return _dt.datetime.strptime(t, fmt).date().isoformat() > ceil
        except ValueError:
            continue
    return False


def aggregate_reads(reads: list) -> dict:
    """read-v5 doc-level aggregation (deterministic; one doc = one voice).

    K=1 makes this near-trivial (one region per doc); the multi-region rule
    mirrors v4's: support-side and refute-side regions together → "3" with a
    region-conflict flag; otherwise the strongest flag on the winning side."""
    flags = [r["direction"] for r in reads if r]
    ev = sorted({i for r in reads if r for i in r["evidence"]})
    sup = [f for f in flags if f in ("5", "4")]
    ref = [f for f in flags if f in ("1", "2")]
    flag = ""
    if sup and ref:
        d, flag = "3", "region-conflict"
    elif sup:
        d = "5" if "5" in sup else "4"
    elif ref:
        d = "1" if "1" in ref else "2"
    elif "3" in flags:
        d = "3"
    elif "X" in flags:
        d = "X"
    else:
        d = "I"
    qcs = [r.get("qc_flag") for r in reads if r and r.get("qc_flag")]
    if qcs and not flag:
        flag = qcs[0]
    reasons = [r.get("reason") for r in reads if r and r.get("reason")]
    return {"direction": d, "evidence": ev, "reason": reasons[0] if reasons else "",
            "qc_flag": flag, "n_regions": len(reads)}


def _norm_evidence(ev, idset):
    """Flatten the evidence pointers a reader emits into sentence ids. The schema asks for a
    flat id list; Kimi / V4-Pro also answer with [start, end] pairs, nested id lists and
    "a-b" strings (41 of the 63 requestion "read-failed" reads were valid reads in one of
    these shapes, 2026-09-18). Returns (ids, normalised?) or (None, _) when unparseable. Shared with reader_lab."""
    out, normed = set(), False
    for e in ev:
        if isinstance(e, bool):
            return None, normed
        if isinstance(e, int):
            out.add(e)
        elif isinstance(e, str) and re.fullmatch(r"\s*\d+\s*(-\s*\d+\s*)?", e):
            a, _, b = e.partition("-"); normed = True
            out.update(range(int(a), int(b or a) + 1))
        elif isinstance(e, list) and e and all(isinstance(x, int) and not isinstance(x, bool) for x in e):
            normed = True
            out.update(range(e[0], e[1] + 1) if len(e) == 2 and e[0] < e[1] else e)
        else:
            return None, normed
    return sorted(out & idset), normed



def llm(messages, cache_key=None, timeout=60, max_tokens=400, want_logprobs=False,
        want_stats=False):
    # want_stats appends {"latency_s", "completion_tok"} as the LAST element; opt-in
    # because five other scripts unpack the 4-tuple positionally.
    body = {"model": VERIFICATION_MODEL, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}, "messages": messages}
    if cache_key:
        body["prompt_cache_key"] = cache_key
    if want_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 4
    t0 = time.time()
    r = _SESSION.post(f"{EXTRACTION_BASE_URL}/chat/completions",
                      headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
                      json=body, timeout=timeout)
    latency = time.time() - t0
    r.raise_for_status()
    j = r.json()
    usage = j.get("usage", {})
    base = (json.loads(j["choices"][0]["message"]["content"] or "{}"),
            usage.get("estimated_cost") or 0.0,
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            usage.get("prompt_tokens") or 0)
    if want_logprobs:
        base += (((j["choices"][0].get("logprobs") or {}).get("content")) or None,)
    if want_stats:
        base += ({"latency_s": round(latency, 3),
                  "completion_tok": usage.get("completion_tokens") or 0},)
    return base


def _flag_logprob(lp_content, flag: str):
    """Logprob of the direction token (soft per-doc vote, read_v5_prompts docstring).

    direction is the FIRST field, so its value token sits in the first few tokens;
    fail-open (None) when the provider returns no logprobs."""
    if not lp_content:
        return None
    for t in lp_content[:12]:
        if (t.get("token") or "").strip().strip('"') == flag:
            return {"logprob": t.get("logprob"),
                    "top": [{"t": a.get("token"), "lp": a.get("logprob")}
                            for a in (t.get("top_logprobs") or [])[:4]]}
    return None


def read_doc(claim_block: str, ids: list[int], sents: list[str], key: str,
             stats: dict | None = None, doc_date: str | None = None):
    doc = "\n".join(f"[{i}] {s}" for i, s in zip(ids, sents))
    # doc_date is None on every pinned path, so the user block is byte-identical there.
    head = f"DOCUMENT (published {doc_date}):" if doc_date else "DOCUMENT:"
    obj, cost, cached, ptok, lp, st = llm(
        [{"role": "system", "content": READ_SYS_ASOF if FIX["read_asof"] else READ_SYS},
         {"role": "user", "content": f"{claim_block}\n\n{head}\n{doc}"}],
        cache_key=key, want_logprobs=True, want_stats=True)
    st = {**st, "prompt_tok": ptok, "cached_tok": cached}
    if stats is not None:     # timing/usage reaches the caller even when the read is rejected
        stats.update(st)
    d = obj.get("direction")
    if isinstance(d, int):          # json_object mode may emit the flag unquoted
        d = str(d)
    ev, normed = _norm_evidence(obj.get("evidence") or [], set(ids))
    if d not in DIRECTIONS or ev is None:
        return None, cost, cached, ptok
    lp_obj = _flag_logprob(lp, d)   # look up the EMITTED flag, before any coercion
    flag = "evidence-normalised" if normed else ""
    # A directional vote MUST carry a citation (Daniel 2026-07-27, carried to v5:
    # the schema allows an empty bucket only with "I"). An uncited non-I flag is
    # not evidence — coerce to on-claim context rather than let it vote.
    if d != "I" and not ev:
        d, flag = "X", "empty-directional"
    # MARKER, not coercion (smoke audit 2026-08-04): a directional flag citing
    # >=90% of a >=15-sentence doc is the signature of silence-as-refutation
    # (country profiles flagged "1" with every sentence cited) — but ~40% of
    # such reads are genuine anchor refutations, so the direction stands and
    # the flag makes the class stratifiable at fit/validation time.
    elif d in ("5", "4", "3", "2", "1") and len(ids) >= 15 and len(ev) / len(ids) >= 0.9:
        flag = "blanket-citation"
    return ({"direction": d, "evidence": ev, "reason": (obj.get("reason") or "")[:80],
             "qc_flag": flag, "flag_logprob": lp_obj, **st},
            cost, cached, ptok)


_YMD = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?")


def _xdate_ceiling(x) -> str | None:
    """Expand a (possibly partial) extracted utterance date to its LATEST day.

    x_date is verbatim from the article ("2021-05-11", "2020-06", "1985"), so a
    partial date expands to the end of its period — the utterance happened at some
    point inside it, and the review−1 clamp bounds the late side regardless."""
    import calendar
    import datetime
    m = _YMD.match(str(x or "").strip())
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2), m.group(3)
    try:
        if d:
            return datetime.date(int(y), int(mo), int(d)).isoformat()
        if mo:
            return datetime.date(int(y), int(mo),
                                 calendar.monthrange(int(y), int(mo))[1]).isoformat()
        return f"{y}-12-31"
    except ValueError:
        return None


def ceiling_for(row) -> tuple[str, str]:
    """Latest publication date evidence may carry, and where that date came from.

    Chain (2026-08-04 rewiring): native claim_date → extracted utterance x_date
    (partial dates expand to their latest day) → review−1. The per-publisher lag
    fallback is retired: the contextualization pass now dates the rows the lag
    was approximating (draw: 2,436 native / 1,622 x_date / 99 review−1).

    HARD CLAMP at review_date − 1 day (Daniel 2026-07-27). 14.5% of split rows had a
    claim_date at or after their OWN review_date, so the retrieval window included the
    fact-check that produced the gold label — the system could find the answer key.
    Clamping makes that leak structurally impossible instead of something we detect
    afterwards and argue about.

    DISTRUST native claim_date when it is >= review_date (smoke audit 2026-08-04):
    that class is review-scrape timestamps (BOOM) and day/month swaps (AFP), not
    utterance dates — 257/4,157 draw rows, 213 with a good x_date. A claim cannot
    be stated on or after its own review; same-day fact-checks lose nothing (x_date
    or review−1 land on the same day)."""
    import datetime
    cd, rd = row.get("claim_date"), row.get("review_date")
    if cd and rd and cd[:10] >= rd[:10]:
        cd = None
    ceil, src = "", "none"
    if cd:
        ceil, src = cd[:10], "claim_date"
    else:
        xc = _xdate_ceiling(row.get("x_date"))
        if xc:
            ceil, src = xc, "x_date"
        elif rd:
            ceil = (datetime.date.fromisoformat(rd[:10]) - datetime.timedelta(days=1)).isoformat()
            src = "review-1"
    if ceil and rd:
        cap = (datetime.date.fromisoformat(rd[:10]) - datetime.timedelta(days=1)).isoformat()
        if ceil >= cap:
            ceil = cap
            if src != "review-1":
                src += "+clamped"
    return ceil, src


def exclusion_for(row, gate: dict) -> str | None:
    """E1 exclusion gate (roadmap 2026-08-04). Returns a reason or None.

    Excluded rows are RECORDED (empty results + reason), never silenced —
    fit_urn's load() already skips zero-result records."""
    # NOTE: the 8 "EXCLUDE-artifact" rows are NOT excluded — the v5 baseline was
    # run without them and uniformity within one measurement outranks 8 rows.
    # They remain tagged in claim_mode_final.parquet for post-Thursday work.
    g = gate.get(row["review_url"])
    if g and g["decision"] == "exclude":
        return f"gate-{g['kind'].lower()}"          # hand-read media/demonstrative call
    if row["resolution_status"] == "unresolvable-list":
        return "unresolvable-list"
    if row["resolution_status"] == "needs-article" and not row["context_ok"]:
        return "needs-article-no-context"
    return None


def run_claim(row, ng_scores, budget, gate):
    t_claim = time.time()
    key = row["review_url"]
    # Resolution (resolve_claims.py) is a zero-leak code-only prefix; the resolved
    # text IS the claim for both QUERY and READ ("ill-posed for READ and for
    # retrieval until the speaker is resolved"). Raw for 97% of the draw.
    claim = row.get("claim_resolved") or row["claim_text"]
    # Mode = the proposition the FACT-CHECKER graded (claim_mode_final.parquet), so
    # READ assesses what the gold label actually applies to. Every draw claim has a
    # resolved value with recorded provenance (both arms agree / no-attribution-verb
    # / hand-read); mode_of() is only a fallback for rows outside that file.
    mode = (row.get("_mode_override") or row.get("claim_mode")
            or mode_of(row.get("claim_type")))
    ceil, ceil_src = ceiling_for(row)
    ctx_used = bool(row.get("context_ok") and (row.get("x_context") or "").strip())
    # Shown date follows the same distrust rule as the ceiling (a date at or after
    # the review is a scrape artifact or extraction error, not the utterance date);
    # rather than show the reader an impossible timeline, show "unknown date".
    rd = (row.get("review_date") or "")[:10]
    claim_date = ""
    for cand in ((row.get("claim_date") or "")[:10], (row.get("x_date") or "").strip()):
        if cand and not (rd and cand[:10] >= rd):
            claim_date = cand
            break
    rec = {"review_url": key, "claim_text": row["claim_text"], "publisher_site": row["publisher_site"],
           "veracity": row["veracity"], "rating_subtype": row["rating_subtype"],
           "claim_type": row["claim_type"], "topic": row["topic"], "yr": row.get("yr"),
           "claim_resolved": claim, "resolution_status": row["resolution_status"],
           "claim_mode": mode, "mode_source": row.get("mode_source"),
           # screen_verdict is a TAG, never a skip (Daniel's judged_axis precedent:
           # gold-misaligned claims are tagged and dropped from the HEADLINE, not
           # from the run). The screen is ~25% precise where it contradicts the
           # hand-read gate, so letting it silently delete claims would cost more
           # good rows than bad; fit_urn strata do the work instead.
           "screen_verdict": row.get("screen_verdict") or "unscreened",
           "screen_leak": bool(row.get("screen_leak")),
           "context_used": ctx_used, "claim_date_shown": claim_date,
           "ceiling": ceil, "ceiling_src": ceil_src,
           "prompts": {"query": QUERY_PROMPT_V, "read": READ_PROMPT_V, "clean": CLEAN_V, "prep": PREP_V},
           "prompt_hash": PROMPT_HASHES,
           "results": [], "cost": 0.0,
           "llm_calls": 0, "prompt_tok": 0, "cached_tok": 0, "completion_tok": 0,
           "search_calls": 0, "querygen_latency_s": 0.0}

    excl = exclusion_for(row, gate)
    if excl:
        rec["excluded"] = excl
        rec["wall_s"] = round(time.time() - t_claim, 3)
        return rec

    # query-v3 user block: resolved claim + usable context + claim date. Context
    # enters ONLY here (the READ block stays claim + date; read-v5 unchanged).
    qblock = [f"CLAIM: {claim}"]
    if ctx_used:
        qblock.append(f"CONTEXT (circulation of the claim, from a neutral description): "
                      f"{row['x_context'].strip()}")
    if claim_date:
        qblock.append(f"CLAIM DATE: {claim_date}")
    qsys = QUERY_SYS
    if FIX["attr_query"] and is_attribution(row, claim):
        qsys = QUERY_SYS_V3A
        rec["prompts"]["query"] = QUERY_PROMPT_V3A
    query = None
    for _ in range(3):   # NEVER fall back to raw claim text (Daniel 2026-07-24):
        try:             # a raw-claim query is a different query distribution and
            q, cost, cached, ptok, st = llm(                                  # contaminates the urn.
                [{"role": "system", "content": qsys},
                 {"role": "user", "content": "\n".join(qblock)}], max_tokens=80, want_stats=True)
            rec["cost"] += cost
            rec["llm_calls"] += 1
            rec["prompt_tok"] += ptok
            rec["cached_tok"] += cached
            rec["completion_tok"] += st["completion_tok"]
            rec["querygen_latency_s"] += st["latency_s"]
            query = (q.get("query") or "").strip()[:300] or None
        except Exception:
            time.sleep(3)
        if query:
            break
    if not query:
        raise RuntimeError("query-gen failed after retries")  # claim skipped, reruns on resume
    rec["query"] = query

    # Policy 2026-07-25 (Daniel): the date ceiling does the fresh-claim work; FC
    # domains are TAGGED (fc_domain), not blocked — measured: FC = 1% of raw
    # ceiling-filtered top-10. Only the origin publisher (the gold source) is
    # excluded. min_results/backfill dropped (vestigial): urn = Google's
    # date-filtered top-10 minus origin. One request per search.
    # `exclude_extra`: a claim can have more than one origin. An outlet's own tweet
    # originates on BOTH x.com and the outlet's site, and letting apnews.com verify an
    # AP claim is the origin-source error with extra steps (outlet_urn_run.py).
    # A None/empty publisher_site (AVeriTeC rows with no original_url) must not
    # become a `-site:None` operator or crash the cache key: drop falsy entries.
    xd = [d for d in [row["publisher_site"]] + list(row.get("exclude_extra") or []) if d]
    stats = {}
    hits = None
    for attempt in range(2):
        _serper_gate()   # global pacing: stay under the plan's request window
        rec["search_calls"] += 1
        try:
            hits = search(query, 10, date_ceiling=ceil or None, exclude_domains=xd,
                          min_results=0, stats=stats, provider="serper")
            break
        except SearchError as e:
            if attempt == 0:
                print("  [serper-cooldown 300s]", flush=True)
                time.sleep(300)
            else:
                raise RuntimeError(f"search failed: {e}")   # claim skipped, reruns on resume
    
    rec["search_stats"] = {k: v for k, v in stats.items() if isinstance(v, (int, bool, str))}

    claim_block = (f"CLAIM: {claim}\n"
                   f"(claimed on {claim_date or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    FC_DOMS = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}
    seen_voices = {}
    for rank, h in enumerate(hits[:10], 1):
        url = h.get("url") or ""
        dom = _domain_of(url)
        rel, ng = _rel_info(dom, url, ng_scores)
        voice = _voice_key(dom)
        mirror = voice in seen_voices
        seen_voices.setdefault(voice, rank)
        fc_domain = any(dom == f or dom.endswith("." + f) for f in FC_DOMS)
        leak = _is_leak(h.get("date"), ceil)
        text = h.get("content") or scrape(url)
        prov = "scrape"
        if not text or len(text) < SNIPPET_MIN_TEXT:
            text, prov = h.get("snippet") or "", "snippet"
        regions, prep_meta = select_regions(text, claim, query)
        prep_meta["raw_chars"] = len(text or "")
        all_ids = [i for ids, _ in regions for i in ids]
        all_sents_map = {}
        for ids, sel in regions:
            for i, sn in zip(ids, sel):
                all_sents_map[i] = sn
        entry = {"rank": rank, "url": url, "domain": dom, "voice": voice,
                 "mirror": mirror, "rel": rel, "ng": ng, "provenance": prov,
                 "fc_domain": fc_domain, "date": h.get("date"), "leak_flag": leak,
                 "n_sents": len(all_sents_map),
                 "sent_ids": sorted(all_sents_map), 
                 "sents": [all_sents_map[i] for i in sorted(all_sents_map)],
                 "prep": prep_meta}
        if FIX["drop_leaks"] and leak:
            # The ceiling is sent to Serper as tbs cd_max, but Google returns pages it
            # has re-dated or updated, so documents PUBLISHED after the post still come
            # back (6 of 10 on one diagnosed claim). Recording leak_flag and reading the
            # document anyway let a July outcome refute an accurate 9 July report.
            # Not read, no direction -> the slot becomes a pad-to-10 silence in load_urn.
            entry["read_status"] = "post-ceiling-leak"
            entry["region_reads"] = []
            rec["results"].append(entry)
            continue
        if fc_domain and not (h.get("date") or "").strip():
            # undated fact-check page: date-unjudgeable, verdict-bearing worst case —
            # never read; excluded from N (observation-refused, not silence).
            # Code-side placeholder, not a model read: read_status is the marker.
            entry["read"] = {"direction": "I", "evidence": [], "reason": "",
                             "qc_flag": "fc-undated-leak-risk"}
            entry["region_reads"] = []
            entry["read_status"] = "fc-undated"
            rec["results"].append(entry)
            continue
        if not all_sents_map:
            entry["read"] = {"direction": "I", "evidence": [], "reason": "",
                             "qc_flag": "empty-doc-code"}
            entry["region_reads"] = []
            entry["read_status"] = "empty-doc"
        else:
            region_reads = []
            for ids, sel in regions:                       # document order, cache-shared prefix
                out = None
                for _ in range(2):
                    rstat = {}
                    try:
                        out, cost, cached, ptok = read_doc(
                            claim_block, ids, sel, key, stats=rstat,
                            doc_date=(h.get("date") if FIX["doc_dates"] else None))
                    except Exception:
                        time.sleep(2)
                        continue
                    rec["cost"] += cost
                    rec["llm_calls"] += 1
                    rec["prompt_tok"] += ptok
                    rec["cached_tok"] += cached
                    rec["completion_tok"] += rstat.get("completion_tok", 0)
                    budget["cached_tok"] += cached
                    budget["prompt_tok"] += ptok
                    if out:
                        break
                region_reads.append(out or {"direction": "I", "evidence": [],
                                            "reason": "", "qc_flag": "read-failed"})
            entry["region_reads"] = region_reads
            entry["read"] = aggregate_reads(region_reads)
            # read_status makes a dead read a first-class non-observation: "failed"
            # must be excluded from N at fit time, never counted as silence.
            entry["read_status"] = ("failed" if all(
                r.get("qc_flag") == "read-failed" for r in region_reads) else "ok")
        rec["results"].append(entry)
    rec["wall_s"] = round(time.time() - t_claim, 3)
    return rec


def main():
    global READ_SYS, READ_PROMPT_V, PROMPT_HASHES
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="25 stratified CAL claims")
    ap.add_argument("--budget", type=float, default=10.0, help="hard USD cap")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reader", choices=list(READERS), default=READ_PROMPT_V,
                    help="read prompt: read-v6.1 (production six-class, DEFAULT) or read-v5 "
                         "(frozen seven-class survey instrument). Prompt text is unchanged.")
    args = ap.parse_args()
    READ_SYS, READ_PROMPT_V = READERS[args.reader]
    PROMPT_HASHES = {"query": prompt_hash(QUERY_SYS), "read": prompt_hash(READ_SYS)}

    # The run set IS the contextualized draw: claim_context_cal.parquet holds exactly
    # the E1 composition (all CAL trues + 2000 F + 600 M, seeds 505 drawn after
    # .sort("review_url") — same procedure this runner used to draw inline). Joining
    # on it guarantees every row has x_date/x_context/context_ok; the smoke samples
    # from the SAME frame so smoke rows are contextualized too.
    ctx = pl.read_parquet(CTX)
    res = pl.read_parquet(RES).select(["review_url", "resolution_status", "claim_resolved"])
    df = (pl.read_parquet(CAL).join(ctx, on="review_url", how="inner")
          .join(res, on="review_url", how="inner").sort("review_url"))
    assert df["resolution_status"].null_count() == 0
    if MODEF.exists():
        df = df.join(pl.read_parquet(MODEF).select(["review_url", "claim_mode", "mode_source"]),
                     on="review_url", how="left")
        missing = df.filter(pl.col("claim_mode").is_null()).height
        assert missing == 0, f"{missing} claims have no resolved mode"
    if SCREEN.exists():
        scr = pl.read_parquet(SCREEN).select(
            pl.col("review_url"),
            pl.col("verdict").alias("screen_verdict"),
            pl.col("context_leak").alias("screen_leak"))
        df = df.join(scr, on="review_url", how="left")
    gate = {d["review_url"]: d for d in
            json.loads(GATE.read_text())["decisions"]}
    if args.smoke:
        parts = [df.filter(pl.col("veracity") >= 4).sample(8, seed=505),
                 df.filter(pl.col("veracity") <= 2).sample(12, seed=505),
                 df.filter(pl.col("veracity") == 3).sample(5, seed=505)]
        df = pl.concat(parts)
    else:
        assert df.height == ctx.height, f"draw drift: {df.height} != {ctx.height}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / ("smoke.jsonl" if args.smoke else "results-00.jsonl")
    seen = set()
    if out.exists():
        for l in open(out):
            try:
                seen.add(json.loads(l)["review_url"])
            except Exception:
                pass
    rows = list(df.iter_rows(named=True))
    import random
    # ---------------------------------------------------------------------------
    # REPRODUCIBLE SAMPLING — DO NOT REMOVE THE SORT.
    # random.shuffle permutes the list it is GIVEN, so seeding alone does not make a
    # subset reproducible: the result is a function of the parquet's physical row
    # order, which changes whenever the splits are rebuilt. This was live for weeks
    # and was only caught on 2026-07-27, when re-running "seed 707, first 420" against
    # a rebuilt CAL returned just 21 of the original 420 claims — so the E1 pilot could
    # not be replayed as the same set. Sorting by a stable key first makes the draw
    # depend only on the seed and the membership of the split.
    # The drawn subset is also written to <out>.sample.json so any run can be replayed
    # exactly even if the splits are later rebuilt.
    # ---------------------------------------------------------------------------
    rows.sort(key=lambda r: r["review_url"])
    random.Random(707).shuffle(rows)   # any prefix = stratified random subset
    if args.limit:
        rows = rows[:args.limit]       # slice BEFORE seen-filter: resume-stable set
    # Rewrite when the drawn set GROWS (staged runs: a --limit tranche first, then
    # the rest). Keeping the first tranche's file would misdescribe the finished
    # run as 1/10 its actual size.
    sample_path = out.with_suffix(".sample.json")
    prev_n = 0
    if sample_path.exists():
        prev_n = json.loads(sample_path.read_text()).get("n", 0)
    if len(rows) > prev_n:
        sample_path.write_text(json.dumps(
            {"seed": 707, "limit": args.limit or None, "n": len(rows),
             "source": [str(CAL), str(CTX), str(RES), str(GATE), str(SCREEN)],
             "review_urls": [r["review_url"] for r in rows]}, indent=1))
    todo = [r for r in rows if r["review_url"] not in seen]
    print(f"{len(todo)} claims to run -> {out.name} | budget ${args.budget} | "
          f"{args.workers} workers | prompts {QUERY_PROMPT_V}/{READ_PROMPT_V}", flush=True)

    ng_scores = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    t0 = time.time()
    fh = open(out, "a")

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = run_claim(row, ng_scores, budget, gate)
        except Exception as e:
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url'][:60]}: {type(e).__name__}", flush=True)
            return
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n = budget["done"]
            if n % 25 == 0 or n == len(todo):
                fh.flush()
                proj = budget["spent"] / n * len(todo)
                cr = budget["cached_tok"] / max(budget["prompt_tok"], 1)
                print(f"  {n}/{len(todo)} | spent ${budget['spent']:.2f} | "
                      f"projected ${proj:.2f} | cache-hit {cr:.0%} | "
                      f"{(time.time()-t0)/60:.0f}m", flush=True)
                if proj > args.budget and n >= max(25, len(todo) // 4):
                    print(f"  BUDGET ABORT: projection ${proj:.2f} > cap ${args.budget}", flush=True)
                    budget["stop"] = True

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"\ndone {budget['done']} claims | ${budget['spent']:.3f} | "
          f"cache-hit {budget['cached_tok']/max(budget['prompt_tok'],1):.0%} | "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
