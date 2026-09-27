"""Unrated-pool census — every evidence result across all datasets/runs (2026-07-23).

Sweeps ALL distinct evidence-bearing run dirs (Truth-Odds round-1 profiles = the raw
10-slot universe; verify runs v2..v7.5 = the engaged multi-round pile), re-classifies
every result mention with TODAY's reliability rules (NG aliases/prefixes/parent,
institutional, trusted fact-checkers, wikipedia), and profiles what the UNRATED bin
actually contains: domain frequency + concentration, category mix, fates, and how many
open/unsupported claims sat on discarded unrated support.

  cd src && uv run python eval/scripts/claim_sourcing/unrated_pool_census.py \
      [-o eval/data/survey_claims/unrated_pool_census.html] [--csv <domains.csv>]

Profile runs expose the full pre-triage result list; verify runs only record what was
read/dropped/cited (not-picked slots are invisible) — composition shares are therefore
reported per source-kind, never pooled across the two.

Current-filter simulation (Daniel 2026-07-23): every mention is first passed through
TODAY's production search filters — the SCRAPE_BLOCKLIST (socials + UGC, server-side)
and origin/sister-brand exclusion — so the census answers "what would currently go as
unrated", not what historical runs happened to retrieve.

Double-count guard (Daniel 2026-07-23): the same post appears in several runs (v6 + v7
reran the same datasets; dev50 dirs overlap dev500) and near-identical queries return
near-identical result lists, so raw mention counts would re-count the same retrieval
draw. All counts are deduped on (dataset, post_id, url) [snippet-voice entries carry no
URL and dedup on (post_id, domain)]; domain rankings also report DISTINCT POSTS; the
starved-claims analysis uses only the newest loop version per dataset (v7 runs).
"""
import argparse, glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[3]))

from pipeline.config import SCRAPE_BLOCKLIST
from pipeline.credibility import is_primary_source, is_trusted_factchecker
from pipeline.search import newsguard_score_map
from pipeline.verify_tweet_claims import (_domain_of, _ng_of, _origin_domains,
                                          _rel_info, _voice_key)

SUP = {"supports", "partially-supports"}
REF = {"refutes", "partially-refutes"}

BASE = _P(__file__).resolve().parents[2] / "data" / "survey_claims"

# (dir, dataset, kind). Profile v1 dirs are strict subsets of v2 (resumed in place) and
# excluded; smoke/replay/snapshot dirs are excluded (tiny or re-draws of the same runs).
RUNS = [
    ("truthodds_profile_averitec_v2", "averitec", "profile"),
    ("truthodds_profile_fcgold_v2", "fcgold", "profile"),
    ("truthodds_profile_dev1000_v2", "tweets", "profile"),
    ("verify_v6_averitec", "averitec", "verify"),
    ("verify_v7_averitec", "averitec", "verify"),
    ("verify_v6_dev500", "tweets", "verify"),
    ("verify_v7_dev500", "tweets", "verify"),
    ("verify_v7_dev500b", "tweets", "verify"),
    ("dev50_verify_trace", "tweets", "verify"),
    ("runs/dev50_main_v2", "tweets", "verify"),
    ("runs/dev50_main_v3", "tweets", "verify"),
    ("runs/dev50_main_v3_rerun", "tweets", "verify"),
    ("runs/dev50_lowng_v2", "tweets", "verify"),
    ("runs/dev50_lowng_v3", "tweets", "verify"),
]

OPEN_STATES = {"open", "unsupported", None, ""}

# Newest loop version per dataset — the only runs the starved-claims analysis uses
# (v6/dev50 dirs re-ran the same claims; counting them again would double-count).
STARVED_RUNS = {"verify_v7_averitec", "verify_v7_dev500", "verify_v7_dev500b"}


# ---------------------------------------------------------------------------
# Current-filter simulation (Daniel 2026-07-23): older runs predate the search-level
# UGC block and server-side origin exclusion, so their piles contain results that
# TODAY would never be retrieved. Every mention is passed through the CURRENT filters
# and the census reports only survivors — "what would currently go as unrated".
# ---------------------------------------------------------------------------
def _is_blocked(dom: str) -> bool:
    d = (dom or "").lower().removeprefix("www.")
    return d in SCRAPE_BLOCKLIST or any(d.endswith("." + b) for b in SCRAPE_BLOCKLIST)


def current_filter(dom: str, origin_keys: set) -> str | None:
    """Return the filter that would remove this result today, or None if it survives.
    Mirrors production search: SCRAPE_BLOCKLIST server-side excludes (socials + UGC)
    and origin/sister-brand exclusion (server-side since 2026-07-21)."""
    if _is_blocked(dom):
        return "ugc-social-blocklist"
    if origin_keys and _voice_key(dom) in origin_keys:
        return "origin-exclusion"
    return None

# ---------------------------------------------------------------------------
# Category map for the head of the unrated pile (hand-labeled from the ranked
# census, 2026-07-23; covers every domain with >=6 mentions). The tail is
# reported as "unlabeled tail". Labels are DESCRIPTIVE families, not
# reliability judgments — the remediation column is where the action is.
# ---------------------------------------------------------------------------
CATEGORIES = {
    # UGC / platforms (social, forums, upload, podcast/video hosts, newsletters)
    "ugc-platform": [
        "threads.com", "reddit.com", "quora.com", "scribd.com", "medium.com",
        "bsky.app", "patreon.com", "nairaland.com", "usmessageboard.com", "dokumen.pub",
        "podcasts.apple.com", "open.spotify.com", "music.amazon.com", "audioboom.com",
        "feeds.acast.com", "dailymotion.com", "pod.wave.co", "substack.com",
        "github.com", "huggingface.co", "localguidesconnect.com",
    ],
    # aggregators / mirrors / AI answer engines — content originates elsewhere
    "aggregator-mirror": [
        "ground.news", "inkl.com", "qoshe.com", "newsbreak.com", "article.wn.com",
        "muckrack.com", "web.archive.org", "archive.org", "archive.ph",
        "grokipedia.com", "perplexity.ai", "indexmundi.com", "indpaedia.com",
    ],
    # genuine news organizations NewsGuard does not cover (international press)
    "intl-press": [
        # India
        "timesofindia.indiatimes.com", "indiatoday.in", "ndtv.com", "theprint.in",
        "thequint.com", "thehindu.com", "m.economictimes.com",
        "economictimes.indiatimes.com", "news18.com", "zeenews.india.com",
        "firstpost.com", "aajtak.in", "deccanherald.com", "dnaindia.com",
        "bhaskar.com", "m.thewire.in", "scroll.in", "thenewsminute.com",
        "wionews.com", "newslaundry.com", "thelogicalindian.com", "news.abplive.com",
        "india.com", "theweek.in", "newindianexpress.com", "thestatesman.com",
        "ibtimes.co.in", "moneycontrol.com", "vccircle.com",
        # Nigeria / Africa
        "guardian.ng", "businessday.ng", "premiumtimesng.com", "dailytrust.com",
        "thisdaylive.com", "punchng.com", "vanguardngr.com", "thecable.ng",
        "channelstv.com", "tribuneonlineng.com", "saharareporters.com",
        "gazettengr.com", "thewhistler.ng", "thenigerialawyer.com", "thesun.ng",
        "icirnigeria.org", "twocircles.net",
        # Pakistan / SE Asia / East Asia
        "dawn.com", "tribune.com.pk", "straitstimes.com", "philstar.com",
        "abs-cbn.com", "gmanetwork.com", "thestar.com.my",
        # Middle East / Caucasus / other
        "aa.com.tr", "ynetnews.com", "jns.org", "arabnews.jp",
        "today.lorientlejour.com", "civil.ge", "oc-media.org", "israel.com",
        "misbar.com", "factcheckhub.com",
    ],
    # US local / student / trade / specialist press
    "us-local-trade-press": [
        "capitolnewsillinois.com", "lapublicpress.org", "gvwire.com",
        "wausaupilotandreview.com", "coloradopolitics.com", "yaledailynews.com",
        "wyomingpublicmedia.org", "kosu.org", "abc45.com", "ktsa.com", "wfmd.com",
        "thebanner.com", "chelseanewsny.com", "hstoday.us", "eenews.net",
        "spacepolicyonline.com", "airandspaceforces.com", "flyingmag.com",
        "medicaleconomics.com", "pharmacytimes.com", "ajmc.com",
        "respiratory-therapy.com", "meritalk.com", "institutionalinvestor.com",
        "jurist.org", "news.bloomberglaw.com", "subscriber.politicopro.com",
        "thediplomat.com", "devex.com", "thecascadian.com", "audacy.com",
        "iheart.com", "thefader.com", "musicrow.com", "news.pollstar.com",
        "awfulannouncing.com", "hitsdailydouble.com", "extratv.com",
        "countrysplash.com", "bugcountry.com", "independentnewsroom.com",
    ],
    # peer-reviewed publishers not on the scholarly allowlist
    "academic-publisher": [
        "tandfonline.com", "mdpi.com", "academic.oup.com", "onlinelibrary.wiley.com",
        "obgyn.onlinelibrary.wiley.com", "pubs.acs.org", "journals.sagepub.com",
        "cambridge.org", "emerald.com", "jstor.org", "ebsco.com",
        "harvardlawreview.org", "frontiersin.org",
    ],
    # preprint servers / upload repositories (deliberately never institutional)
    "preprint-repository": [
        "researchgate.net", "academia.edu", "papers.ssrn.com", "arxiv.org",
        "medrxiv.org", "amelica.org", "macrothink.org", "journals.npsa-se.org.ng",
        "publichealthpolicyjournal.com",
    ],
    # data / reference databases (stats, sports, entertainment, real estate, stock media)
    "data-reference": [
        "statista.com", "macrotrends.net", "tradingeconomics.com",
        "worldpopulationreview.com", "census2011.co.in", "censusreporter.org",
        "usafacts.org", "top500.org", "tools.myfooddata.com", "wallethub.com",
        "zillow.com", "redfin.com", "homes.com", "realtor.com", "polymarket.com",
        "imdb.com", "cricinfo.com", "mlb.com", "nba.com", "nhl.com", "olympics.com",
        "fox.com", "gettyimages.com", "alamy.com", "shutterstock.com",
        "newsflare.com", "ticketmaster.com", "legacy.com", "investing.com",
        "marketbeat.com", "alphaspread.com", "247wallst.com", "nasdaq.com",
        "stocktitan.net", "electproject.github.io", "dashboards.sdgindex.org",
        "global-monitoring.com", "intelpoint.co",
    ],
    # official records / gov-adjacent / courts / transcripts / legislative trackers
    "official-record": [
        "opcw.org", "manhattanda.org", "dawnsturgess.independent-inquiry.uk",
        "news.devon-cornwall.police.uk", "sec.state.ma.us", "nyse.com",
        "everycrsreport.com", "govtrack.us", "legiscan.com", "law.justia.com",
        "supreme.justia.com", "ncsl.org", "federalreservehistory.org",
        "americanrhetoric.com", "rev.com", "iea.org", "nsidc.org",
        "rubinobservatory.org", "ama-assn.org", "apha.org", "americanbar.org",
        "ap.org", "newsroom.ap.org", "nar.realtor", "iranprimer.usip.org",
        "whitehousehistory.org",
    ],
    # think tanks / NGOs / advocacy (both directions)
    "think-tank-ngo": [
        "brennancenter.org", "taxfoundation.org", "amnesty.org", "amnestyusa.org",
        "crisisgroup.org", "chathamhouse.org", "vera.org", "cbpp.org",
        "americanimmigrationcouncil.org", "fdd.org", "jamestown.org", "csis.org",
        "crfb.org", "isdglobal.org", "longwarjournal.org", "isis-online.org",
        "urban.org", "milkeninstitute.org", "naacp.org", "naacpldf.org",
        "becketfund.org", "americanoversight.org", "knightcolumbia.org",
        "arabcenterdc.org", "beyondpesticides.org", "iaomt.org", "yaf.org",
        "action.yaf.org", "aclukansas.org", "freedom250.org", "america250.org",
        "unitehere11.org", "nlcng.org", "potomacriverkeepernetwork.org",
        "americanrivers.org", "wagingnonviolence.org", "coveringclimatenow.org",
        "resist.bot", "multistate.us", "bti-project.org", "terrorism-info.org.il",
        "israel-alma.org", "conference-board.org", "nautilus.org",
        "newyork.edtrust.org", "eqca.org", "drishtiias.com", "futures.issafrica.org",
        "brookings.edu",
    ],
    # corporate IR / PR / company sites
    "corporate-pr-ir": [
        "prnewswire.com", "ir.applieddigital.com", "investors.staar.com",
        "investor.lilly.com", "investors.coreweave.com", "coreweave.com",
        "blog.23andme.com", "twosigma.com", "a16z.com", "sirion.ai",
        "provisioservices.com", "midstatemedical.org", "oattravel.com",
        "market-scope.com", "testbook.com", "byjus.com",
    ],
    # partisan / alternative media and religious press
    "alt-partisan-media": [
        "blackagendareport.com", "mronline.org", "popularresistance.org",
        "palestinechronicle.com", "orinocotribune.com", "countercurrents.org",
        "americanfreakshow.news", "newsmaxbalkans.com", "muslimmirror.com",
        "milligazette.com", "indiatomorrow.net", "ewtnnews.com", "osvnews.com",
        "johnmenadue.com", "mindingthecampus.org", "airwars.org",
        "lindaikejisblog.com",
    ],
    # entertainment / lifestyle / listicle content
    "entertainment-lifestyle": [
        "mensxp.com", "storypick.com", "thebetterindia.com", "bollywoodshaadis.com",
        "complex.com", "buzzfeed.com", "consequence.net",
    ],
}

_CAT_OF = {d: c for c, ds in CATEGORIES.items() for d in ds}


def category(dom: str) -> str:
    d = (dom or "").lower().removeprefix("www.")
    if d in _CAT_OF:
        return _CAT_OF[d]
    if d.endswith(".substack.com"):
        return "ugc-platform"
    if d.startswith(("investors.", "investor.", "ir.")):
        return "corporate-pr-ir"
    return "tail-unlabeled"


# Remediation lever per category — what closing this slice of the gap looks like.
REMEDIATION = {
    "intl-press": "third-party ratings (MBFC / Wikipedia perennial / Lin et al.)",
    "us-local-trade-press": "third-party ratings; many are credible niche outlets",
    "academic-publisher": "extend the scholarly allowlist (PRIMARY_SOURCES already has springer/sciencedirect/nature)",
    "official-record": "extend institutional allowlist + TLD patterns (state .us, police.uk, inquiry.uk) + NG aliases (ap.org)",
    "data-reference": "decide a fixed mid-tier score (wikipedia precedent) for curated aggregators; rest stays unrated",
    "think-tank-ngo": "mid-tier voice at most; leaning risk — needs a deliberate policy",
    "ugc-platform": "blocked at search as of 2026-07-23; deliberate leftovers: substack, github, huggingface (can carry primary content)",
    "aggregator-mirror": "treat as mirror, never a voice (circularity machinery)",
    "preprint-repository": "keep excluded (unvetted uploads)",
    "corporate-pr-ir": "keep unrated; primary only for claims ABOUT the company (needs care)",
    "alt-partisan-media": "keep unrated / unreliable",
    "entertainment-lifestyle": "keep unrated",
    "tail-unlabeled": "agreement-based trust once runs accumulate; lists won't reach it",
}


def load(dirpath):
    for f in sorted(glob.glob(str(BASE / dirpath / "results-*.jsonl"))):
        for l in open(f):
            d = json.loads(l)
            if d.get("ok") and d.get("result"):
                yield d["result"]


def rel_bin(dom, url, scores):
    """Current-rules bin for a domain: institutional / fact-checker / NG tiers / unrated."""
    rel, ng = _rel_info(dom, url, scores)
    if rel == "PRIMARY":
        return "institutional"
    if is_trusted_factchecker(dom):
        return "fact-checker"
    if ng is None:
        return "unrated"
    return ("NG 90-100" if ng >= 90 else "NG 80-89" if ng >= 80 else
            "NG 70-79" if ng >= 70 else "NG 60-69" if ng >= 60 else "NG <60")


BINS = ["institutional", "fact-checker", "NG 90-100", "NG 80-89", "NG 70-79",
        "NG 60-69", "NG <60", "unrated"]


def profile_mentions(rec):
    """Profile run: every raw round-1 slot -> (url, domain, recorded_ng, fate, on_claim)."""
    if not rec.get("rounds"):
        return
    rd = rec["rounds"][0]
    targets = set(rd.get("targets") or [])
    ev = rec.get("evidence") or []
    doc_by_url = {d["url"]: d["id"] for d in rd.get("docs") or []}
    drops = {d["url"]: d.get("reason", "") for d in rd.get("dropped") or []}
    sc = {s["i"]: s for s in (rec.get("profile") or {}).get("source_classes") or []}
    for r in rd.get("results") or []:
        i, url = r.get("i"), r.get("url") or ""
        dom = r.get("domain") or _domain_of(url)
        if url in doc_by_url:
            did = doc_by_url[url]
            ents = [e for e in ev if e.get("src") == did and e.get("claim_id") in targets]
            stances = {e.get("stance") for e in ents}
            fate = ("evidence-support" if stances & SUP else
                    "evidence-refute" if stances & REF else
                    "read-other" if ents else "read-no-evidence")
        elif any(e.get("src") == f"R1.{i}" and e.get("snippet_only") for e in ev):
            fate = "snippet-voice"
        elif url in drops:
            fate = "dropped"
        else:
            fate = "not-picked"
        yield url, dom, r.get("ng"), fate, (sc.get(i) or {}).get("on_claim")


def verify_mentions(rec):
    """Verify run: docs read + dropped + snippet voices, all rounds.
    Yields (url, domain, recorded_ng, fate, None)."""
    ev = rec.get("evidence") or []
    ng_by_dom = {e.get("domain"): e.get("ng") for e in ev if e.get("domain")}
    doc_stance = defaultdict(set)
    for e in ev:
        doc_stance[e.get("src")].add(e.get("stance"))
    for rd in rec.get("rounds") or []:
        for d in rd.get("docs") or []:
            did, url = d.get("id"), d.get("url") or ""
            st = doc_stance.get(did, set())
            fate = ("evidence-support" if st & SUP else
                    "evidence-refute" if st & REF else
                    "read-other" if st else "read-no-evidence")
            yield url, d.get("domain") or _domain_of(url), ng_by_dom.get(d.get("domain")), fate, None
        for d in rd.get("dropped") or []:
            url = d.get("url") or ""
            yield url, _domain_of(url), None, "dropped", None
    for e in ev:
        if e.get("snippet_only"):
            yield "", e.get("domain") or "", e.get("ng"), "snippet-voice", None


def final_states(rec):
    """claim_id -> final ledger state (verify runs)."""
    led = rec.get("ledger") or {}
    if isinstance(led, dict):
        return {str(k): v for k, v in led.items()}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="write the full unrated domain frequency table")
    ap.add_argument("-o", "--out", default=None, help="HTML report path")
    args = ap.parse_args()

    scores = newsguard_score_map()
    mentions = []           # (dataset, kind, run, post_id, url, dom, recorded_ng, fate, on_claim)
    removed = []            # mentions the CURRENT filters would have kept out of retrieval
    open_claims = []        # (run, dataset, post_id, claim_text, [supporting full-read domains])

    for run, dataset, kind in RUNS:
        n = 0
        for rec in load(run):
            pid = rec.get("id") or rec.get("post_id")
            origin_keys = {_voice_key(x) for x in _origin_domains(rec.get("domain") or "")}
            fn = profile_mentions if kind == "profile" else verify_mentions
            for url, dom, ng, fate, oc in fn(rec):
                if not dom:
                    continue
                flt = current_filter(dom, origin_keys)
                if flt:
                    removed.append((dataset, pid, url or f"snip::{dom}", dom, flt))
                    continue
                mentions.append((dataset, kind, run, pid, url, dom, ng, fate, oc))
                n += 1
            if run in STARVED_RUNS:
                states = final_states(rec)
                ev = rec.get("evidence") or []
                claims = {str(i + 1): c.get("c") for i, c in enumerate(rec.get("claims") or [])}
                doc_url = {d.get("id"): d.get("url") or "" for rd in rec.get("rounds") or []
                           for d in rd.get("docs") or []}
                for cid, st in states.items():
                    if (st or "open") not in OPEN_STATES:
                        continue
                    sup = {}  # domain -> url of the supporting full-read doc
                    for e in ev:
                        if (str(e.get("claim_id")) == cid and e.get("stance") in SUP
                                and not e.get("snippet_only") and e.get("domain")
                                and current_filter(e.get("domain"), origin_keys) is None):
                            sup.setdefault(e.get("domain"), doc_url.get(e.get("src"), ""))
                    open_claims.append((run, dataset, pid, claims.get(cid, cid),
                                        sorted(sup.items())))
        print(f"loaded {run}: {n} mentions", file=sys.stderr)

    removed_u = {(ds, pid, key): (dom, flt) for ds, pid, key, dom, flt in removed}
    rm_ct = Counter(flt for dom, flt in removed_u.values())
    rm_doms = defaultdict(Counter)
    for dom, flt in removed_u.values():
        rm_doms[flt][dom] += 1
    print("\n== Removed by CURRENT filters (unique (dataset,post,url); would not be retrieved today) ==")
    for flt, nn in rm_ct.most_common():
        top = ", ".join(f"{d} {c}" for d, c in rm_doms[flt].most_common(8))
        print(f"  {flt}: {nn}   top: {top}")

    # ---- dedup (Daniel 2026-07-23): same post across runs draws near-identical result
    # lists. One record per (dataset, post, url) — per kind for composition (profile and
    # verify observe different universes), cross-kind for the pooled census. Keep the
    # most-engaged fate and any on-claim classification seen for the pair.
    FATE_PRI = {f: i for i, f in enumerate(
        ["evidence-support", "evidence-refute", "read-other", "read-no-evidence",
         "snippet-voice", "dropped", "not-picked"])}

    def _dedup(rows, with_kind):
        best = {}
        for ds, kind, run, pid, url, dom, ng, fate, oc in rows:
            key = (ds, kind, pid, url or f"snip::{dom}") if with_kind else \
                  (ds, pid, url or f"snip::{dom}")
            cur = best.get(key)
            if cur is None or FATE_PRI[fate] < FATE_PRI[cur[7]]:
                best[key] = (ds, kind, run, pid, url, dom, ng, fate,
                             oc if oc is not None else (cur[8] if cur else None))
            elif oc is not None and cur[8] is None:
                best[key] = cur[:8] + (oc,)
        return list(best.values())

    comp_view = _dedup(mentions, with_kind=True)     # composition / fates per ds+kind
    census_view = _dedup(mentions, with_kind=False)  # pooled domain census
    print(f"dedup: {len(mentions)} raw mentions -> {len(comp_view)} per-kind unique "
          f"-> {len(census_view)} cross-run unique (dataset,post,url)", file=sys.stderr)

    # ---- classify every domain with current rules ----
    bin_of = {}
    for ds, kind, run, pid, url, dom, ng, fate, oc in census_view:
        if dom not in bin_of:
            bin_of[dom] = rel_bin(dom, url, scores)

    # composition per dataset x kind (deduped per kind)
    comp = defaultdict(Counter)
    for ds, kind, run, pid, url, dom, ng, fate, oc in comp_view:
        comp[(ds, kind)][bin_of[dom]] += 1

    print("\n== Composition under CURRENT rules (share of unique (post,url) records) ==")
    hdr = "dataset/kind".ljust(22) + "".join(b.rjust(14) for b in BINS) + "     n"
    print(hdr)
    for key in sorted(comp):
        c = comp[key]
        tot = sum(c.values())
        row = f"{key[0]}/{key[1]}".ljust(22)
        row += "".join(f"{c[b]/tot:13.1%} " for b in BINS) + f"{tot:6d}"
        print(row)

    # ---- re-rating delta: recorded UNRATED at runtime, rated now (profile runs only —
    # verify 'dropped' mentions never recorded an ng, so they'd inflate the denominator) ----
    rerated = Counter()
    n_rec_unrated = 0
    for ds, kind, run, pid, url, dom, ng, fate, oc in comp_view:
        if kind == "profile" and ng is None:
            n_rec_unrated += 1
            if bin_of[dom] != "unrated":
                rerated[bin_of[dom]] += 1
    print(f"\n== Profile records recorded UNRATED at runtime, resolving under current rules ==")
    print(f"recorded unrated at runtime: {n_rec_unrated}; now rated: {sum(rerated.values())}")
    for b, n in rerated.most_common():
        print(f"  -> {b}: {n}")

    # ---- the unrated pile (cross-run deduped census) ----
    unr = [m for m in census_view if bin_of[m[5]] == "unrated"]
    dom_ct = Counter(m[5] for m in unr)
    dom_posts = defaultdict(set)
    dom_ds = defaultdict(set)
    dom_fates = defaultdict(Counter)
    dom_url = {}
    dom_onclaim = defaultdict(lambda: [0, 0])  # [on_claim true, classified]
    for ds, kind, run, pid, url, dom, ng, fate, oc in unr:
        dom_posts[dom].add((ds, pid))
        dom_ds[dom].add(ds)
        dom_fates[dom][fate] += 1
        dom_url.setdefault(dom, url)
        if oc is not None:
            dom_onclaim[dom][1] += 1
            dom_onclaim[dom][0] += bool(oc)

    tot = len(unr)
    # rank by distinct posts (robust to one post pulling the same domain many times),
    # then by unique records
    ranked = sorted(dom_ct, key=lambda d: (-len(dom_posts[d]), -dom_ct[d]))
    print(f"\n== Unrated pile: {tot} unique (post,url) records, {len(dom_ct)} distinct domains ==")
    csum = 0
    marks = {10, 25, 50, 100, 200}
    for i, d in enumerate(ranked, 1):
        csum += dom_ct[d]
        if i in marks:
            print(f"  top {i:>3} domains cover {csum/tot:5.1%} of unrated records")
    once = sum(1 for d in dom_ct if dom_ct[d] == 1)
    print(f"  domains appearing once: {once} ({once/len(dom_ct):.0%} of domains)")

    # category masses
    cat_ct, cat_ev, cat_posts = Counter(), Counter(), defaultdict(set)
    for ds, kind, run, pid, url, dom, ng, fate, oc in unr:
        c = category(dom)
        cat_ct[c] += 1
        cat_posts[c].add((ds, pid))
        if fate in ("evidence-support", "evidence-refute"):
            cat_ev[c] += 1
    print("\n== Unrated pile by category (unique records; labeled head + tail) ==")
    for c, n in cat_ct.most_common():
        print(f"  {c:24s} {n:5d} ({n/tot:5.1%})  ev:{cat_ev[c]:<4d} posts:{len(cat_posts[c]):<5d} "
              f"-> {REMEDIATION.get(c, '')}")

    print(f"\n== Top 150 unrated domains (ranked by distinct posts) ==")
    for d in ranked[:150]:
        f = dom_fates[d]
        n = dom_ct[d]
        evn = f["evidence-support"] + f["evidence-refute"]
        oc = dom_onclaim[d]
        ocs = f"{oc[0]}/{oc[1]}" if oc[1] else "-"
        print(f"  {d:42s} {n:4d} posts:{len(dom_posts[d]):<4d} ev:{evn:<3d} "
              f"snip:{f['snippet-voice']:<3d} not-picked:{f['not-picked']:<4d} on-claim:{ocs:>7s} "
              f" {category(d):22s} [{','.join(sorted(dom_ds[d]))}]")

    # ---- fates rated vs unrated (profile runs only — full slot visibility) ----
    fr, fu = Counter(), Counter()
    ocr, ocu = [0, 0], [0, 0]
    for ds, kind, run, pid, url, dom, ng, fate, oc in comp_view:
        if kind != "profile":
            continue
        (fu if bin_of[dom] == "unrated" else fr)[fate] += 1
        tgt = ocu if bin_of[dom] == "unrated" else ocr
        if oc is not None:
            tgt[1] += 1
            tgt[0] += bool(oc)
    print("\n== Fates, rated vs unrated (profile runs, raw round-1 slots) ==")
    fates = ["evidence-support", "evidence-refute", "read-other", "read-no-evidence",
             "snippet-voice", "dropped", "not-picked"]
    tr, tu = sum(fr.values()), sum(fu.values())
    print("fate".ljust(20) + "rated".rjust(10) + "unrated".rjust(10))
    for f in fates:
        print(f.ljust(20) + f"{fr[f]/tr:9.1%} " + f"{fu[f]/tu:9.1%}")
    print(f"on-claim (classified slots): rated {ocr[0]/max(ocr[1],1):.1%} ({ocr[1]}), "
          f"unrated {ocu[0]/max(ocu[1],1):.1%} ({ocu[1]})")

    # ---- starved claims (newest verify run per dataset only) under CURRENT rules ----
    starved = defaultdict(Counter)   # dataset -> {0,1,2+}
    starved_ex = []
    for run, ds, pid, claim, sup in open_claims:
        upairs = [(d, u) for d, u in sup
                  if bin_of.get(d, rel_bin(d, u, scores)) == "unrated"]
        starved[ds][min(len(upairs), 2)] += 1
        if len(upairs) >= 2 and len(starved_ex) < 40:
            starved_ex.append((run, pid, claim, upairs))
    print(f"\n== Open/unsupported claims at end of the NEWEST verify runs ({', '.join(sorted(STARVED_RUNS))}) ==")
    for ds in sorted(starved):
        s = starved[ds]
        t = sum(s.values())
        print(f"  {ds}: {t} open; 0 unrated supporting voices: {s[0]}, 1: {s[1]}, "
              f">=2 (closeable if unrated counted): {s[2]} ({s[2]/t:.0%})")
    print("\n  examples (>=2 unrated full-read supporting voices, claim stayed open):")
    for run, pid, claim, pairs in starved_ex[:15]:
        print(f"  [{run}] {pid}  {str(claim)[:90]}")
        print(f"      unrated support: {', '.join(d for d, u in pairs)}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["domain", "records", "posts", "category", "evidence", "read_no_ev",
                        "snippet_voice", "not_picked", "dropped", "on_claim", "classified",
                        "datasets", "example_url"])
            for d in ranked:
                f = dom_fates[d]
                oc = dom_onclaim[d]
                w.writerow([d, dom_ct[d], len(dom_posts[d]), category(d),
                            f["evidence-support"] + f["evidence-refute"],
                            f["read-no-evidence"], f["snippet-voice"], f["not-picked"],
                            f["dropped"], oc[0], oc[1], "|".join(sorted(dom_ds[d])), dom_url[d]])
        print(f"\nwrote {args.csv}")

    if args.out:
        import html as _h

        def tr(cells, tag="td"):
            return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

        comp_rows = ""
        for key in sorted(comp):
            c = comp[key]
            t = sum(c.values())
            comp_rows += tr([f"{key[0]}/{key[1]}"] + [f"{c[b]/t:.1%}" for b in BINS] + [t])

        cat_rows = ""
        for c, n in cat_ct.most_common():
            cat_rows += tr([c, n, f"{n/tot:.1%}", cat_ev[c], len(cat_posts[c]),
                            _h.escape(REMEDIATION.get(c, ""))])

        dom_rows = ""
        for d in ranked[:100]:
            f = dom_fates[d]
            dom_rows += tr([_h.escape(d), dom_ct[d], len(dom_posts[d]),
                            f["evidence-support"] + f["evidence-refute"],
                            f["snippet-voice"], f["not-picked"], category(d),
                            ",".join(sorted(dom_ds[d]))])

        fate_rows = ""
        for f in fates:
            fate_rows += tr([f, f"{fr[f]/tr_ if (tr_:=sum(fr.values())) else 0:.1%}",
                             f"{fu[f]/tu_ if (tu_:=sum(fu.values())) else 0:.1%}"])

        starv_rows = ""
        for ds in sorted(starved):
            s = starved[ds]
            t = sum(s.values())
            starv_rows += tr([ds, t, s[0], s[1], f"{s[2]} ({s[2]/t:.0%})"])

        def _sup_links(pairs):
            out = []
            for d, u in pairs:
                out.append(f'<a href="{_h.escape(u)}" target="_blank">{_h.escape(d)}</a>'
                           if u else _h.escape(d))
            return ", ".join(out)

        ex_html = "".join(
            f"<li><code>{_h.escape(str(pid))}</code> {_h.escape(str(claim)[:140])}"
            f"<br><span class='sub'>unrated support: {_sup_links(pairs)}</span></li>"
            for run, pid, claim, pairs in starved_ex[:20])

        rer_rows = "".join(tr([b, n]) for b, n in rerated.most_common())

        rm_rows = ""
        for flt, nn in rm_ct.most_common():
            top = ", ".join(f"{_h.escape(d)} ({c})" for d, c in rm_doms[flt].most_common(6))
            rm_rows += tr([flt, nn, top])

        conc = ""
        csum2 = 0
        for i, d in enumerate(ranked, 1):
            csum2 += dom_ct[d]
            if i in marks:
                conc += tr([f"top {i}", f"{csum2/tot:.1%}"])

        doc = f"""<!doctype html><meta charset="utf-8">
<title>Unrated-pool census — all datasets (2026-07-23)</title>
<style>
body{{font:14px/1.45 -apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}}
table{{border-collapse:collapse;margin:.6em 0 1.4em}} td,th{{border:1px solid #ccc;padding:3px 9px;text-align:right}}
th{{background:#f2f2f2}} td:first-child,th:first-child{{text-align:left}}
h2{{margin-top:1.6em;border-bottom:1px solid #ddd;padding-bottom:.2em}}
.note{{color:#555;font-size:13px;max-width:75ch}} .sub{{color:#777;font-size:12px}}
code{{background:#f5f5f5;padding:0 3px}} li{{margin:.4em 0}}
</style>
<h1>Unrated-pool census — every evidence result, all datasets</h1>
<p class="note">Built by <code>unrated_pool_census.py</code> (2026-07-23) over {len(RUNS)} run dirs
(Truth-Odds round-1 profiles + verify v2–v7 runs; profile v1 dirs and smokes excluded as re-draws).
All counts deduped on (dataset, post, url) — the same post re-run under v6/v7 draws near-identical
result lists, so raw mentions would double-count (33,713 raw → 25,054 unique records). Every domain
re-classified with TODAY's rules (NG aliases/prefixes/parent, institutional, fact-checkers, wikipedia).
Profile runs expose the full pre-triage 10-slot list; verify runs only record what was engaged
(read/dropped/cited) — shares are never pooled across the two kinds.</p>

<h2>0 · Removed first by TODAY's search filters</h2>
<table><tr><th>filter</th><th>unique records removed</th><th>top domains</th></tr>{rm_rows}</table>
<p class="note">Historical runs predate the search-level UGC block (2026-07-21) and server-side
origin exclusion. These records would never be retrieved today and are excluded from EVERYTHING
below — the census reads as "what would currently go as unrated". Remaining production filters
that shape the pile but don't remove domains up front: the eval-only fact-check block and date
ceiling, the credibility rerank (ordering only), the scrape gauntlet (dup/near-dup, junk,
off-topic, too-short, republication/circularity guard), and the bar-side rules (opinion-URL
pieces and UNRATED/UNRELIABLE voices never count toward closure).</p>

<h2>1 · Composition under current rules</h2>
<table><tr><th>dataset/kind</th>{"".join(f"<th>{b}</th>" for b in BINS)}<th>n</th></tr>{comp_rows}</table>
<p class="note">Unrated is ~30–38% everywhere — the single largest bin after the NG-rated mass.
Verify-run universes look slightly better than profile universes because triage already avoids
unrated sources when picking what to engage.</p>

<h2>2 · Already reclaimed by the alias/institutional work</h2>
<table><tr><th>now resolves to</th><th>profile records</th></tr>{rer_rows}</table>
<p class="note">Of {n_rec_unrated} profile records logged UNRATED at run time, {sum(rerated.values())}
({sum(rerated.values())/max(n_rec_unrated,1):.0%}) resolve under current rules — almost all via the
.edu→institutional move. The remaining pile below is measured AFTER this reclaim.</p>

<h2>3 · The unrated pile: {tot} unique records, {len(dom_ct)} domains</h2>
<table><tr><th>head</th><th>share of unrated records</th></tr>{conc}</table>
<p class="note">{once} domains ({once/len(dom_ct):.0%}) appear exactly once. The pile is long-tailed:
no small curated list fixes most of it, but the head is very fixable (next table).</p>

<h2>4 · Categories (labeled head ≥6 records + tail)</h2>
<table><tr><th>category</th><th>records</th><th>share</th><th>became evidence</th><th>posts</th><th>remediation</th></tr>{cat_rows}</table>

<h2>5 · Top 100 unrated domains (ranked by distinct posts)</h2>
<table><tr><th>domain</th><th>records</th><th>posts</th><th>evidence</th><th>snippet-voice</th><th>not-picked</th><th>category</th><th>datasets</th></tr>{dom_rows}</table>

<h2>6 · Behavior: rated vs unrated (profile runs, raw round-1 slots)</h2>
<table><tr><th>fate</th><th>rated</th><th>unrated</th></tr>{fate_rows}</table>
<p class="note">on-claim (of slots the profiler classified): rated {ocr[0]/max(ocr[1],1):.1%}
(n={ocr[1]}), unrated {ocu[0]/max(ocu[1],1):.1%} (n={ocu[1]}). A rated result is ~2.2× as likely
to become evidence; most of the gap is triage never picking unrated results (68% vs 48% not-picked).
Caveat: TRIAGE is instructed to prefer rated sources, so part of this gap is self-inflicted policy,
not source quality.</p>

<h2>7 · What the gap costs: starved claims (newest runs only, current rules)</h2>
<table><tr><th>dataset</th><th>open/unsupported claims</th><th>0 unrated support</th><th>1 voice</th><th>&ge;2 voices (closeable)</th></tr>{starv_rows}</table>
<p class="note">Claims that ended open/unsupported in {", ".join(sorted(STARVED_RUNS))}, by how many
UNRATED full-read supporting voices they had accumulated. The &ge;2 bucket would close under the
existing two-voice rule if those sources were rated ≥60 — the direct upper bound on what rating
coverage buys. The examples below show why a blanket "count unrated" is wrong: credible sources
(rcog.org.uk, statista) sit next to junk (vickihobbs.com) in the same bucket.</p>
<ul>{ex_html}</ul>
"""
        _P(args.out).write_text(doc)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
