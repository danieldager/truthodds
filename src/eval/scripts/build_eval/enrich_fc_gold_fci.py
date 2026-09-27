"""fc-gold v3 — enrich the v2 admitted set from Fact Check Insights (Duke Reporters' Lab).

FCI is a ClaimReview dump (257,877 claimReviews + 2,986 mediaReviews, snapshot
2025-09-08). It carries three things the Google Fact Check Tools API never returns:
`itemReviewed.appearance[].url` (the ORIGINAL post that made the claim),
`reviewRating.ratingExplanation` (the fact-checker's own rationale), and a claim date
for publishers GFC leaves blank. This script joins those onto fc-gold and nothing else.

THE KEY IS NOT THE URL. Both corpora carry multi-claim articles (debate wrap-ups,
speech checks), so a shared review_url does NOT mean a shared claim. The validated key
is (normalised review_url, normalised claim_text); rows whose claim text disagrees at a
shared URL are REJECTED, not guessed at. Validation + the numbers behind that decision:
eval/data/fc_gold_lineage.md §v3, audit page eval/data/fci_enrichment_audit.html.

URL normalisation keeps the query string minus tracking params — stripping the whole
query collapses publishers that carry the article id in the query (globes.co.il: 330
distinct reviews onto one key).

FILLS ONLY, NEVER OVERWRITES. claim_date and veracity are written only where the v2
value is null, so the enrichment cannot silently move an existing gold label. Every
enriched row keeps `fci_match_status` + `fci_id` so any addition is traceable to one
FCI record and the whole enrichment is reversible by dropping the fci_* columns.

DELIBERATELY NOT TAKEN: `reviewRating.ratingValue`. The numeric scale's direction is
per-publisher and unreliable (PolitiFact declares worst=0/best=9 but emits 0="True",
4="False", 5="Pants on Fire"; dpa declares worst=5/best=1; 15,654 rows declare
worst=2/best=2), and every row that has one already has a harmonised veracity.

LEAKAGE: `fc_rationale` states the verdict AND the reasoning. It is EVAL-ONLY and must
never reach the verifier (same rule as IDEA-014). Appearance URLs pointing at a
fact-checker's own domain are dropped for the same reason.

  # stage 1 — flatten the raw dump to parquet (once; needs the 313 MB json)
  uv run python -m eval.scripts.build_eval.enrich_fc_gold_fci --flatten \
      --fci-json ~/Downloads/fact_check_insights.json
  # stage 2 — build v3
  uv run python -m eval.scripts.build_eval.enrich_fc_gold_fci

Outputs:
  eval/data/fci/fci_claimreviews.parquet   flattened dump (regenerable; 58 MB)
  eval/data/fci/fci_mediareviews.parquet   media-authenticity rows (Phase-1b seed)
  eval/data/fc_gold_v3.parquet             v2 admitted + fci_* columns + filled nulls
  eval/data/fc_gold_v3_provenance.json     inputs (w/ sha256), key, per-status counts
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import html
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, unquote

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.harmonize import harmonise_veracity, judged_axis_from_rating
from pipeline.credibility import TRUSTED_FACTCHECKERS

DATA = Path("eval/data")
FCI_DIR = DATA / "fci"
FCI_CR = FCI_DIR / "fci_claimreviews.parquet"
FCI_MR = FCI_DIR / "fci_mediareviews.parquet"
# Built on the CLUSTERED file, not the admitted one: fc_claim_cluster.py adds
# cluster_id/cluster_size/cluster_pubs/q_flags to the same 47,111 rows, and
# build_calval_splits.py reads those. Sourcing from admitted would fork the lineage and
# strip the cluster columns off v3.
SRC = DATA / "fc_gold_v2_clustered.parquet"
OUT = DATA / "fc_gold_v3.parquet"
PROV = DATA / "fc_gold_v3_provenance.json"

# Query params that never identify the article. Everything else is kept: some
# publishers put the article id in the query and stripping it merges distinct reviews.
_TRACK = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|source|_ga|igshid|s|amp|"
                    r"__twitter.*|cmpid|ncid|smid|partner|sh|share|via|feature|si)$", re.I)

_SOCIAL = re.compile(r"(facebook\.com|fb\.watch|twitter\.com|x\.com|instagram\.com|tiktok\.com|"
                     r"threads\.net|t\.me|youtube\.com|youtu\.be|reddit\.com|vk\.com|weibo)", re.I)

# Embed/tracking params to strip from POST urls. Twitter's oEmbed cruft is not just
# noise: `ref_url` carries the FACT-CHECK's own URL, so an uncleaned post URL leaks the
# verdict source. Keys that identify the post (story_fbid, fbid, id, v, …) are kept.
_POST_TRACK = re.compile(r"^(ref_src|ref_url|ref|twterm|twgr|twcon|twcamp|utm_.*|fbclid|gclid|"
                         r"igshid|__tn__|__cft__.*|eid|mibextid|rdid|share_url|si|feature|"
                         r"source|s|t|sfnsn|extid|app|_rdr|_rdc)$", re.I)


def clean_post_url(u: str) -> str:
    """Drop embed/tracking params from an appearance URL, keep post-identifying ones."""
    try:
        p = urlsplit(u)
    except ValueError:
        return u
    pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False)
             if not _POST_TRACK.match(k)]
    q = urlencode(pairs)
    return f"{p.scheme}://{p.netloc}{p.path}" + (f"?{q}" if q else "")

NORM_VERSION = "url-v2/text-v1"  # bump if either normaliser changes; recorded in provenance


def norm_url(u) -> str | None:
    if not isinstance(u, str) or not u.strip():
        return None
    u = u.strip()
    if not re.match(r"^https?://", u, re.I):
        u = "http://" + u
    try:
        p = urlsplit(u)
    except ValueError:
        return None
    host = (p.netloc or "").lower().split("@")[-1].split(":")[0]
    host = re.sub(r"^(www|m|amp)\.", "", host)
    path = re.sub(r"/+$", "", re.sub(r"/(amp|amp\.html)$", "", unquote(p.path or ""))).lower()
    pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if not _TRACK.match(k)]
    q = urlencode(sorted(pairs))
    return f"{host}{path}" + (f"?{q}" if q else "") or None


def norm_text(t) -> str:
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKC", html.unescape(html.unescape(t)))
    for a, b in [("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("\xa0", " ")]:
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip().strip('"\'. ').lower()


def _toks(t):
    return set(re.findall(r"\w+", t, re.UNICODE))


def jaccard(a: str, b: str) -> float:
    A, B = _toks(a), _toks(b)
    return len(A & B) / len(A | B) if A | B else 0.0


def _s(x) -> str:
    return "" if x is None else str(x)


def _first(x):
    """FCI emits a few fields as either a scalar or a list; take the scalar."""
    while isinstance(x, list):
        x = x[0] if x else None
    return x


def _host(u: str) -> str:
    m = re.search(r"^https?://([^/?#]+)", u or "", re.I)
    return re.sub(r"^(www|m|web|l|mobile|vt)\.", "", m.group(1).lower()) if m else ""


# ---------------------------------------------------------------- stage 1: flatten
def flatten(src: Path) -> dict:
    print(f"reading {src} ...", flush=True)
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    D = json.loads(raw)
    del raw
    meta = D.get("meta", {})
    print(f"  sha256 {sha[:16]}…  meta={meta}", flush=True)

    rows = []
    for r in D["claimReviews"]:
        it = r.get("itemReviewed") or {}
        a = _first(r.get("author")) or {}
        rr = _first(r.get("reviewRating")) or {}
        ia = _first(it.get("author")) or {}
        fa = _first(it.get("firstAppearance")) or {}
        ap = it.get("appearance") or []
        if isinstance(ap, dict):
            ap = [ap]
        apps = []
        for x in ap:
            x = _first(x)
            u = x.get("url") if isinstance(x, dict) else x
            if isinstance(u, str) and u.startswith("http"):
                apps.append(u)
        au = _s(_first(a.get("url")))
        rows.append(dict(
            fci_id=_s(r.get("id")),
            review_url=_s(r.get("url")),
            site=_host(au) or None,
            publisher_name=_s(_first(a.get("name"))) or None,
            claim_text=_s(_first(r.get("claimReviewed"))) or None,
            claimant=_s(_first(ia.get("name"))) if isinstance(ia, dict) else None,
            review_date=_s(r.get("datePublished"))[:10] or None,
            claim_date=_s(it.get("datePublished"))[:10] or None,
            rating=_s(_first(rr.get("alternateName"))) or None,
            rating_value=_s(_first(rr.get("ratingValue"))) or None,
            best=_s(_first(rr.get("bestRating"))) or None,
            worst=_s(_first(rr.get("worstRating"))) or None,
            rating_explanation=_s(_first(rr.get("ratingExplanation"))) or None,
            appearance_urls="|".join(apps) or None,
            first_appearance=_s(_first(fa.get("url"))) if isinstance(fa, dict) else None,
        ))
    FCI_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(FCI_CR)
    print(f"wrote {FCI_CR}: {len(rows)} claimReviews")

    m = [dict(fci_id=_s(r.get("id")),
              publisher=_s(_first((r.get("author") or {}).get("name"))) or None,
              review_url=_s(r.get("url")), review_date=_s(r.get("datePublished"))[:10] or None,
              media_url=_s((r.get("itemReviewed") or {}).get("contentUrl")) or None,
              media_type=_s((r.get("itemReviewed") or {}).get("@type")) or None,
              authenticity_category=_s(r.get("mediaAuthenticityCategory")) or None,
              original_media_link=_s(r.get("originalMediaLink")) or None,
              original_media_context=_s(r.get("originalMediaContextDescription")) or None,
              start_time=_s((r.get("itemReviewed") or {}).get("startTime")) or None,
              end_time=_s((r.get("itemReviewed") or {}).get("endTime")) or None)
         for r in D["mediaReviews"]]
    pl.DataFrame(m).write_parquet(FCI_MR)
    print(f"wrote {FCI_MR}: {len(m)} mediaReviews")
    return dict(path=str(src), sha256=sha, bytes=len(json.dumps(meta)) and src.stat().st_size,
                retrieved_at=meta.get("retrievedAt"),
                claim_review_count=meta.get("claimReviewCount"),
                media_review_count=meta.get("mediaReviewCount"))


# ---------------------------------------------------------------- stage 2: join
def build(source: Path) -> None:
    if not FCI_CR.exists():
        sys.exit(f"missing {FCI_CR} — run with --flatten --fci-json <path> first")
    g = pl.read_parquet(source)
    F = pl.read_parquet(FCI_CR)
    print(f"{source.name}: {len(g)} rows | {FCI_CR.name}: {len(F)} rows")

    # Fact-checker domains whose pages must never be handed back as "the original post".
    # CURATED ONLY — do NOT derive this from the dump's author.url: FCI lists facebook.com
    # as a publisher site (fact-checks published via a FB page), and folding that in
    # blocklists every genuine Facebook post URL (2,353 of them, i.e. most of the payload).
    fc_domains = ({d.lower() for d in TRUSTED_FACTCHECKERS if "/" not in d}
                  | {s.lower() for s in g["publisher_site"].drop_nulls().to_list()})
    fc_domains = {d for d in fc_domains if not _SOCIAL.search(d)}  # belt and braces

    fidx = collections.defaultdict(list)
    for r in F.iter_rows(named=True):
        k = norm_url(r["review_url"])
        if k:
            r["_nt"] = norm_text(r["claim_text"])
            fidx[k].append(r)

    status, jac, fid, frat, fexpl = [], [], [], [], []
    posts, socials, npost, nsocial = [], [], [], []
    fill_date, date_src = [], []
    n_selfref = 0

    for row in g.iter_rows(named=True):
        k = norm_url(row["review_url"])
        nt = norm_text(row["claim_text"])
        cands = fidx.get(k, []) if k else []
        best, j, st = None, 0.0, "no_url_match"
        if cands:
            exact = [c for c in cands if c["_nt"] == nt and nt]
            if exact:
                best, j = exact[0], 1.0
                st = "exact" if len(cands) == 1 else "exact_multiclaim"
            else:
                best, j = max(((c, jaccard(nt, c["_nt"])) for c in cands), key=lambda t: t[1])
                st = "near" if j >= 0.9 else ("ambiguous" if len(cands) > 1 else "claim_mismatch")
        accept = st in ("exact", "exact_multiclaim", "near")
        status.append(st)
        jac.append(round(j, 4))
        if not accept:
            best = None
        fid.append(best["fci_id"] if best else None)
        frat.append(best["rating"] if best else None)
        ex = _s(best["rating_explanation"]) if best else ""
        fexpl.append(ex if len(ex.strip()) > 20 else None)

        urls = [u for u in _s(best["appearance_urls"] if best else "").split("|") if u]
        keep = []
        for u in urls:
            # suffix match, not equality: the fact-checkers host archived copies of the
            # original post on cdn./static. subdomains (cdn.factcheck.org, static.politifact.com)
            hst = _host(u)
            if any(hst == d or hst.endswith("." + d) for d in fc_domains):
                n_selfref += 1
                continue
            keep.append(clean_post_url(u))
        soc = [u for u in keep if _SOCIAL.search(u)]
        posts.append("|".join(keep) or None)
        socials.append("|".join(soc) or None)
        npost.append(len(keep))
        nsocial.append(len(soc))

        cd = _s(row["claim_date"])
        if re.match(r"^\d{4}-\d{2}-\d{2}", cd):
            fill_date.append(row["claim_date"])
            date_src.append("gfc")
        elif best and re.match(r"^\d{4}-\d{2}-\d{2}$", _s(best["claim_date"])):
            fill_date.append(best["claim_date"])
            date_src.append("fci")
        else:
            fill_date.append(row["claim_date"])
            date_src.append("")

    v3 = g.with_columns(
        pl.Series("fci_match_status", status),
        pl.Series("fci_match_jaccard", jac, dtype=pl.Float64),
        pl.Series("fci_id", fid),
        pl.Series("fci_rating", frat),
        pl.Series("fc_rationale", fexpl),          # EVAL ONLY — never feed to the verifier
        pl.Series("post_urls", posts),
        pl.Series("post_urls_social", socials),
        pl.Series("n_post_urls", npost, dtype=pl.Int64),
        pl.Series("n_post_urls_social", nsocial, dtype=pl.Int64),
        pl.Series("claim_date_source", date_src),
    )
    n_date_filled = sum(1 for s in date_src if s == "fci")
    v3 = v3.with_columns(pl.Series("claim_date", fill_date))

    # veracity FILL (never overwrite): harmonise the FCI rating with the SAME rules v2 used
    ver = v3["veracity"].to_list()
    sub = v3["rating_subtype"].to_list()
    src_ = v3["harmonisation_source"].to_list()
    ax = v3["judged_axis"].to_list()
    n_ver_filled = 0
    for i, (v, r) in enumerate(zip(ver, frat)):
        if v is not None or not r:
            continue
        nv, ns = harmonise_veracity(r)          # text-only: numerics deliberately not used
        if nv is None:
            continue
        ver[i], sub[i], src_[i] = nv, ns, "fci_rule"
        ax[i] = ax[i] or judged_axis_from_rating(r)
        n_ver_filled += 1
    v3 = v3.with_columns(
        pl.Series("veracity", ver, dtype=pl.Float64),
        pl.Series("rating_subtype", sub),
        pl.Series("harmonisation_source", src_),
        pl.Series("judged_axis", ax),
    )
    # keep the v2 invariants consistent with the newly filled rows
    v3 = v3.with_columns(
        (pl.col("rating_subtype") == "unprovable").alias("nee"),
        ((pl.col("veracity") == 3) & (pl.col("rating_subtype") == "mixed")).alias("contested"),
    )
    # ---- acceptance test: v3 must be v2 + columns + null-fills, nothing else. A silent
    # overwrite of a gold label is the one failure mode that would not show up downstream.
    # v2 uses "" (not null) as the unresolved sentinel on rating_subtype /
    # harmonisation_source / judged_axis, so "" -> value is a FILL, not an overwrite.
    def _absent(x):
        return x is None or x == "" or (isinstance(x, float) and x != x)

    for col in ("veracity", "claim_date", "rating_subtype", "harmonisation_source",
                "judged_axis", "original_rating", "claim_text", "review_url"):
        before, after = g[col].to_list(), v3[col].to_list()
        bad = [i for i, (b, a) in enumerate(zip(before, after)) if not _absent(b) and b != a]
        if bad:
            sys.exit(f"ABORT: enrichment overwrote {len(bad)} non-null {col} values "
                     f"(first at row {bad[0]}: {before[bad[0]]!r} -> {after[bad[0]]!r})")
    if len(v3) != len(g):
        sys.exit(f"ABORT: row count changed {len(g)} -> {len(v3)}")
    leaked = v3.filter(~pl.col("fci_match_status").is_in(["exact", "exact_multiclaim", "near"])
                       & (pl.col("fci_id").is_not_null() | pl.col("post_urls").is_not_null()
                          | pl.col("fc_rationale").is_not_null()))
    if len(leaked):
        sys.exit(f"ABORT: payload written onto {len(leaked)} non-accepted rows")
    print("acceptance test: no overwrites, row count stable, no payload on rejected rows")

    v3.write_parquet(OUT)

    counts = collections.Counter(status)
    acc = sum(counts[s] for s in ("exact", "exact_multiclaim", "near"))
    url_matched = len(g) - counts["no_url_match"]
    prov = dict(
        dataset="fc_gold_v3",
        built=datetime.date.today().isoformat(),
        built_by="eval/scripts/build_eval/enrich_fc_gold_fci.py",
        git_commit=_git_head(),
        source_dataset=dict(path=str(source), rows=len(g),
                            sha256=_sha256_file(source)),
        enrichment_source=_fci_meta(),
        match_key="(norm_url, norm_claim_text)",
        normaliser_version=NORM_VERSION,
        accepted_statuses=["exact", "exact_multiclaim", "near"],
        near_threshold_jaccard=0.9,
        status_counts=dict(counts),
        matched=acc,
        matched_pct_of_url_matched=round(acc / url_matched * 100, 2) if url_matched else None,
        payload=dict(
            rows_with_post_url=int(v3.filter(pl.col("n_post_urls") > 0).height),
            rows_with_social_post_url=int(v3.filter(pl.col("n_post_urls_social") > 0).height),
            post_urls_total=int(v3["n_post_urls"].sum()),
            self_referential_urls_dropped=n_selfref,
            rows_with_fc_rationale=int(v3.filter(pl.col("fc_rationale").is_not_null()).height),
            claim_dates_filled=n_date_filled,
            veracity_filled=n_ver_filled,
        ),
        not_taken=dict(rating_value="per-publisher direction unreliable; veracity already harmonised",
                       claim_text_only_fallback="recovers 6.0% of unmatched, 79 post URLs; collision risk"),
        audit_page="eval/data/fci_enrichment_audit.html",
        lineage="eval/data/fc_gold_lineage.md",
    )
    PROV.write_text(json.dumps(prov, indent=1, ensure_ascii=False))

    print(f"\nwrote {OUT}: {len(v3)} rows, {len(v3.columns)} cols")
    for s, n in counts.most_common():
        print(f"  {s:<18} {n:>7}")
    print(f"  ACCEPTED {acc} ({acc/url_matched*100:.1f}% of URL-matched)")
    p = prov["payload"]
    print(f"\npayload: post_url rows {p['rows_with_post_url']} "
          f"(social {p['rows_with_social_post_url']}) | rationales {p['rows_with_fc_rationale']} "
          f"| claim_dates +{p['claim_dates_filled']} | veracity +{p['veracity_filled']} "
          f"| self-ref dropped {p['self_referential_urls_dropped']}")
    print(f"veracity now: {dict(sorted(collections.Counter(v3['veracity'].to_list()).items(), key=lambda t: (t[0] is None, t[0])))}")
    print(f"wrote {PROV}")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _sha256_file(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


_FCI_META_CACHE = FCI_DIR / "fci_source.json"


def _fci_meta() -> dict:
    if _FCI_META_CACHE.exists():
        return json.loads(_FCI_META_CACHE.read_text())
    return dict(note="raw-dump metadata unavailable; re-run --flatten to record it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flatten", action="store_true", help="rebuild the flattened FCI parquets")
    ap.add_argument("--fci-json", type=Path,
                    default=Path.home() / "Downloads" / "fact_check_insights.json")
    ap.add_argument("--source", type=Path, default=SRC, help="gold set to enrich")
    args = ap.parse_args()

    if args.flatten:
        meta = flatten(args.fci_json)
        FCI_DIR.mkdir(parents=True, exist_ok=True)
        _FCI_META_CACHE.write_text(json.dumps(meta, indent=1))
        print(f"wrote {_FCI_META_CACHE}")
    build(args.source)


if __name__ == "__main__":
    main()
