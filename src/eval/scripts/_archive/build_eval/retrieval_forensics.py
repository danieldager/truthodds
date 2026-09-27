"""Why did the all-irrelevant dossiers miss the evidence the reviewer cited?

    uv run python -m eval.scripts.build_eval.retrieval_forensics

$0 -- saved reads plus two artifacts already on disk:
  eval/data/retrieval_forensics/fc_review_links.jsonl   outbound links scraped from
      each fc-gold review_url (direct requests, then keyless r.jina.ai for the
      publishers that 403). Regenerable, no paid API.
  eval/data/urn_runs/c2_false/match{,_ext}.jsonl        the CN note text, which
      carries the note author's source URLs inline.

An "all-irrelevant" claim is one whose every retrieved document read I. Each such
miss is bucketed against the reviewer's own sources:

  A  the reviewer's domain never appeared in ANY result for that claim
  B  the source existed but could not be reached -- it sits on the UGC blocklist,
     or on the excluded origin publisher
  C  we surfaced the domain (or the exact URL), read it, and still flagged I
  U  unclassifiable: no source link recovered

MEASURABILITY LIMIT. Bucket B is only PARTLY measurable here and the number must
never be quoted as if it were complete. `evidence_urn_run.run_claim` applies NO
rerank and NO keep-K -- it reads all ten of Google's results -- so there is no
keep-K cut for anything to be lost at. The UGC drop happens inside
`search.serper_finalize` BEFORE the disk cache writes, so a URL Google returned
and we dropped leaves no trace in any artifact. What IS measurable is whether the
reviewer's source SITS on a blocked domain. Scrape failures are not losses: the
runner falls back to the snippet and reads the document anyway.

CONTROLS ARE THE POINT. The same measurement runs on claims from the same corpus
that were NOT all-irrelevant. Bucket A is high everywhere -- working retrieval
routinely finds different-but-adequate evidence -- so the A share alone says
nothing. The discriminating variable is C.
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from pipeline.config import SCRAPE_BLOCKLIST  # noqa: E402
from eval.scripts.build_eval import fit_urn  # noqa: E402

FLAGS = set("54321XI")
E1 = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
C2 = SRC / "eval/data/urn_runs/c2_false"
FCL = SRC / "eval/data/retrieval_forensics/fc_review_links.jsonl"
CTRL_N, CTRL_SEED = 500, 828

# A fact-check article's <a> set is mostly not evidence. Four filters, in order:
# self-domain, share/consent/donate widgets, archive mirrors of the offending post,
# and PER-PUBLISHER CHROME (any external domain in >=30% of that publisher's
# articles is template). What survives splits into UGC and evidence.
WIDGET = re.compile(r'(addtoany|api\.whatsapp|share\.flipboard|/sharer|/intent/|/share\?|'
                    r'l\.join1440|ifcncodeofprinciples|poynter\.org/ifcn|/cdn-cgi/|'
                    r'googletagmanager|doubleclick|paypal|patreon|cookielaw|cookiepedia|'
                    r'onetrust|apple\.news|giving\.aws|digitalcourses|/newsletter|'
                    r'/donate|/subscribe)', re.I)
ARCHIVE = {"archive.is", "archive.ph", "archive.md", "archive.li", "archive.vn",
           "archive.today", "web.archive.org", "perma.cc", "ghostarchive.org",
           "megalodon.jp", "archive.org"}
NAV = {"google.com", "news.google.com", "translate.google.com", "play.google.com",
       "apps.apple.com", "itunes.apple.com", "wa.me", "whatsapp.com", "t.me", "gravatar.com"}
CHROME_DF = 0.30
URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')


def dom(u: str) -> str:
    return urlsplit(u).netloc.lower().removeprefix("www.").removeprefix("m.").removeprefix("amp.")


def blocked(d: str) -> bool:
    return d in SCRAPE_BLOCKLIST or any(d.endswith("." + b) for b in SCRAPE_BLOCKLIST)


def dmatch(a: str, b: str) -> bool:
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def norm(u: str) -> tuple[str, str]:
    sp = urlsplit(u.split("#")[0])
    return sp.netloc.lower().removeprefix("www."), sp.path.rstrip("/").lower()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def bucket(src_urls, got_urls, got_doms, exclude_dom=None) -> tuple[str, str]:
    src_doms = list(dict.fromkeys(d for d in (dom(u) for u in src_urls) if d))
    if not src_doms:
        return "U", "no_source_link"
    gn = {norm(u) for u in got_urls}
    if any(norm(u) in gn for u in src_urls):
        return "C", "exact_url"
    if any(any(dmatch(g, d) for g in got_doms) for d in src_doms):
        return "C", "domain_only"
    blk = [d for d in src_doms if blocked(d)]
    xcl = [d for d in src_doms if exclude_dom and dmatch(d, exclude_dom)]
    if not [d for d in src_doms if d not in blk and d not in xcl]:
        return "B", ("blocklist" if blk else "publisher_excluded")
    return "A", ("some_blocked" if blk or xcl else "clean")


def allirr(r: dict) -> bool:
    ds = [d["read"]["direction"] for d in r["results"] if d["read"]["direction"] in FLAGS]
    return bool(ds) and all(d == "I" for d in ds)


def well_indexed(d: str) -> bool:
    """Would a normal search engine have this domain. NewsGuard-rated at any score, or
    government / academic / primary / science-publisher by the production high-value test."""
    from pipeline.search import _is_high_value, newsguard_score_map
    ng = newsguard_score_map()
    d = d.lower().removeprefix("www.")
    return _is_high_value(d) or d in ng or any(d.endswith("." + x) for x in ng)


def report(rows: list[dict], label: str) -> None:
    n = len(rows)
    c = collections.Counter(x["bucket"] for x in rows)
    print(f"\n=== {label}  n={n} ===")
    for b in "ABCU":
        p, lo, hi = wilson(c[b], n)
        print(f"  {b}: {c[b]:4d}  {p * 100:5.1f}%  [{lo * 100:4.1f}, {hi * 100:4.1f}]")
    print("  detail:", collections.Counter((x["bucket"], x["detail"]) for x in rows).most_common())
    cl = [x for x in rows if x["bucket"] != "U"]
    if not cl:
        return
    print(f"  -- classifiable n={len(cl)} ({len(cl) / n * 100:.0f}%)")
    cc = collections.Counter(x["bucket"] for x in cl)
    for b in "ABC":
        p, lo, hi = wilson(cc[b], len(cl))
        print(f"    {b}: {cc[b]:4d}  {p * 100:5.1f}%  [{lo * 100:4.1f}, {hi * 100:4.1f}]")


# --------------------------------------------------------------------------- CN
def cn(verbose: bool = True) -> dict:
    notes = {}
    for p in ("match.jsonl", "match_ext.jsonl"):
        for line in (C2 / p).open():
            r = json.loads(line)
            notes[r["post_id"]] = r.get("note") or ""
    excl = {e["claim"][:80] for e in json.loads((C2 / "fit_exclusions.json").read_text())}
    hit, ctl = [], []
    for p in ("scores.jsonl", "scores_ext.jsonl"):
        for line in (C2 / p).open():
            r = json.loads(line)
            if ((r.get("claim_resolved") or r.get("claim_text") or "")[:80]) in excl:
                continue
            if not [d for d in r["results"] if d["read"]["direction"] in FLAGS]:
                continue
            urls = [u.rstrip('.,;:)') for u in URL_RE.findall(notes.get(r["post_id"], ""))]
            b, det = bucket(urls, [d["url"] for d in r["results"]],
                            [d["domain"] for d in r["results"]], r.get("publisher_site"))
            nd = len([d for d in r["results"] if d["read"]["direction"] in FLAGS])
            (hit if allirr(r) else ctl).append({"bucket": b, "detail": det, "ndocs": nd})
    if verbose:
        report(hit, "CN false, all-irrelevant")
        report(ctl, "CN false, control (not all-irrelevant)")
    return {"hit": hit, "ctl": ctl, "n_pool": len(ctl)}


# ----------------------------------------------------------------------- fc-gold
def gold(verbose: bool = True) -> dict:
    links = {}
    for line in FCL.open():
        r = json.loads(line)
        links[r["review_url"]] = r["links"]

    def survives(ru, ls):
        self_d = dom(ru)
        out = []
        for l in ls:
            u = l["url"]
            d = dom(u)
            if not d or (d == self_d or dmatch(d, self_d)):
                continue
            if WIDGET.search(u) or d in NAV:
                continue
            if d in ARCHIVE or any(d.endswith("." + a) for a in ARCHIVE):
                continue
            out.append((d, u))
        return out

    surv = {ru: survives(ru, ls) for ru, ls in links.items()}
    pub_docs, pub_dom = collections.Counter(), collections.Counter()
    for ru, s in surv.items():
        pub = dom(ru)
        pub_docs[pub] += 1
        for d in {d for d, _ in s}:
            pub_dom[(pub, d)] += 1
    chrome = {(p, d) for (p, d), k in pub_dom.items()
              if pub_docs[p] >= 5 and k / pub_docs[p] >= CHROME_DF}

    axis = fit_urn.load_judged_axis()
    hit, pool = [], []
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3):
            continue
        if (r.get("rating_subtype") or "?") == "mixed":
            continue
        if (axis.get(r["review_url"]) or "untagged") == fit_urn.MEDIA_AXIS:
            continue
        if not [d for d in r["results"] if d["read"]["direction"] in FLAGS]:
            continue
        ru = r["review_url"]
        nd = len([d for d in r["results"] if d["read"]["direction"] in FLAGS])
        if ru not in surv:
            rec = {"bucket": "U", "detail": "article_not_scraped", "pub": r["publisher_site"]}
        else:
            ev, ugc = [], []
            for d, u in surv[ru]:
                if (dom(ru), d) in chrome:
                    continue
                (ugc if blocked(d) else ev).append(u)
            if ev:
                b, det = bucket(ev, [d["url"] for d in r["results"]],
                                [d["domain"] for d in r["results"]], r.get("publisher_site"))
            elif ugc:
                b, det = "B", "only_ugc_sources"
            else:
                b, det = "U", "no_source_links_in_article"
            rec = {"bucket": b, "detail": det, "pub": r["publisher_site"]}
            if b == "A":
                rec["wi"] = any(well_indexed(d) for d in {dom(u) for u in ev})
        rec["ndocs"] = nd
        (hit if allirr(r) else pool).append(rec)

    import random
    random.Random(CTRL_SEED).shuffle(pool)
    ctl = pool[:CTRL_N]
    if not verbose:
        return {"hit": hit, "ctl": ctl, "n_pool": len(pool)}
    report(hit, "fc-gold FALSE, all-irrelevant")
    report(ctl, f"fc-gold FALSE, control ({CTRL_N} of {len(pool)}, seed {CTRL_SEED})")
    nA = [x for x in hit if x["bucket"] == "A"]
    wi = sum(1 for x in nA if x["wi"])
    p, lo, hi = wilson(wi, len(nA))
    print(f"\n  bucket-A claims citing a well-indexed source: {wi}/{len(nA)} "
          f"{p * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]")

    # composition check: the A>C / C>A flip must hold inside each publisher
    print("\n  publisher-stratified (classifiable only, both arms >=10):")
    th = collections.defaultdict(collections.Counter)
    tc = collections.defaultdict(collections.Counter)
    for x in hit:
        th[x["pub"]][x["bucket"]] += 1
    for x in ctl:
        tc[x["pub"]][x["bucket"]] += 1
    for p in sorted(set(th) | set(tc), key=lambda p: -sum(th[p].values())):
        A, C = th[p], tc[p]
        na, nc = A["A"] + A["B"] + A["C"], C["A"] + C["B"] + C["C"]
        if na < 10 or nc < 10:
            continue
        print(f"    {p:22s} all-irr n={na:3d} A{A['A'] / na * 100:5.1f}% C{A['C'] / na * 100:5.1f}%"
              f"  |  ctrl n={nc:3d} A{C['A'] / nc * 100:5.1f}% C{C['C'] / nc * 100:5.1f}%")
    return {"hit": hit, "ctl": ctl, "n_pool": len(pool)}


if __name__ == "__main__":
    cn()
    gold()
