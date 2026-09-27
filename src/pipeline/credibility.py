"""Lightweight source-credibility tiers for reranking retrieved evidence.

No LLM, no network — a transparent domain prior used to PREFER primary/institutional
sources over blogs and to DROP social media before summarising. It is a tie-breaker on
top of the provider's own relevance ranking, not an override (a high-tier source can
still be wrong; we only reorder, we don't trust blindly).

Tiers: 2 = trusted (gov/edu/int, fact-checkers, primary orgs, major wires/journals);
       1 = default (everything else); 0 = excluded (social media — dropped).
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from pipeline.config import SCRAPE_BLOCKLIST  # social/login-wall domains -> tier 0

# Professional fact-checking organizations (IFCN-signatory class), extended internationally
# 2026-07-15 (Daniel: "the allowlist should be extended internationally, and we should
# generally trust fact checkers"). A full-read from one of these counts as a STRONG reliable
# voice in the verify loop (NewsGuard's coverage map is US-centric and misses most of them).
TRUSTED_FACTCHECKERS = {
    # US / UK
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "leadstories.com",
    "checkyourfact.com", "factcheckni.org",
    # wire-service fact desks
    "factcheck.afp.com", "apnews.com/ap-fact-check", "reuters.com/fact-check",
    # Africa
    "africacheck.org", "dubawa.org", "pesacheck.org", "ghanafact.com",
    # South Asia
    "boomlive.in", "altnews.in", "factly.in", "newsmeter.in", "newschecker.in",
    "vishvasnews.com", "factcrescendo.com", "rumorscanner.com",
    # SE Asia / Pacific
    "verafiles.org", "tsek.ph", "blackdotresearch.sg", "aap.com.au",
    # Europe
    "correctiv.org", "maldita.es", "newtral.es", "pagellapolitica.it", "facta.news",
    "teyit.org", "faktisk.no", "demagog.org.pl", "demagog.cz", "mimikama.org",
    "euvsdisinfo.eu", "factuel.afp.com",
    # Latin America
    "chequeado.com", "aosfatos.org", "lupa.uol.com.br", "colombiacheck.com",
    "animalpolitico.com",
}


def is_trusted_factchecker(domain: str) -> bool:
    d = (domain or "").lower().removeprefix("www.")
    return any(d == f or d.endswith("." + f) for f in TRUSTED_FACTCHECKERS if "/" not in f)


# Curated trusted set. The gov/int TLD rules below catch most primary sources;
# this set covers fact-checkers, major wires/journals, and primary orgs without those TLDs.
_TRUSTED = TRUSTED_FACTCHECKERS | {
    # major wires / journalism of record
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "npr.org", "wsj.com", "economist.com",
    # primary orgs / data
    "who.int", "opcw.org", "worldbank.org", "data.worldbank.org", "un.org", "imf.org",
    "europa.eu", "oecd.org", "macrotrends.net", "ourworldindata.org", "statista.com",
    # peer-reviewed / scholarly
    "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "doi.org", "nature.com",
    "sciencedirect.com", "springer.com", "frontiersin.org",
}

_TRUSTED_TLDS = (".gov", ".edu", ".int", ".ac.uk")


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().lstrip("www.")


def tier(url: str) -> int:
    """2 = trusted, 1 = default, 0 = excluded (social)."""
    d = _domain(url)
    if not d:
        return 1
    if d in SCRAPE_BLOCKLIST or any(d.endswith("." + b) for b in SCRAPE_BLOCKLIST):
        return 0
    if d.endswith(_TRUSTED_TLDS) or ".gov." in d or ".edu." in d:  # gov/edu, incl. .gov.in/.edu.au
        return 2
    if any(d == t or d.endswith("." + t) for t in _TRUSTED):  # trusted root or its subdomain
        return 2
    return 1


def rerank(results: list[dict], keep: int) -> list[dict]:
    """Drop social (tier 0), then sort by credibility tier (desc), preserving the
    provider's relevance order within a tier. Return the top `keep`."""
    kept = [(tier(r["url"]), i, r) for i, r in enumerate(results)]
    kept = [(t, i, r) for (t, i, r) in kept if t > 0]
    kept.sort(key=lambda x: (-x[0], x[1]))  # tier desc, original position asc
    return [r for _, _, r in kept[:keep]]


# =============================================================================
# Primary-source detection — for the verify_tweet_claims corroboration gate (2026-07-13)
# =============================================================================
# "Primary" = the institution that PRODUCES the record (the statistic, the ruling, the
# vote, the filing), not an outlet reporting on it. ONE qualifying primary full-read can
# close a claim in the gate, so this is deliberately conservative. Detection = government
# TLD suffix patterns + a curated allowlist for what patterns miss (Germany and France
# have no gov TLD). Deliberately NOT primary: PR wires (prnewswire/businesswire — paid,
# unvetted, a documented fake-release vector), state-actor sites beyond what the gov-TLD
# patterns catch, preprint servers, aggregators (statista/macrotrends/ourworldindata),
# publisher umbrellas (wiley/mdpi). Rationale + discuss-list: scratchpad research brief
# 2026-07-13, decisions in docs/verify_loop_versions.md (v4).

# Registry-delegated government/institutional suffixes. Matched as the whole domain or a
# suffix of it. The generic .gov.XX ccTLD shape (gov.uk, gov.au, gov.br, ...) is a regex.
# .edu/.ac.uk removed from the PRIMARY path 2026-07-15: a university-blog essay
# (contendingmodernities.nd.edu) anchored a wrong refutation as "PRIMARY". Academic pages are
# commentary unless they are the record itself; scholarly PUBLISHERS live in PRIMARY_SOURCES.
# (.edu stays in _TRUSTED_TLDS for tier-2 credibility and in is_reference.)
_GOV_SUFFIXES = (
    ".gov", ".mil", ".int",
    ".gouv.fr", ".gc.ca", ".go.jp", ".go.kr", ".go.th", ".go.id", ".go.ke",
    ".gob.es", ".gob.mx", ".gob.ar", ".gub.uy", ".govt.nz", ".gv.at", ".admin.ch",
)
_GOV_CC = re.compile(r"(^|\.)gov\.[a-z]{2}$")

PRIMARY_SOURCES = {
    # intergovernmental organizations (mostly .org — the TLD patterns miss all of these)
    "un.org", "who.int", "imf.org", "worldbank.org", "data.worldbank.org", "oecd.org",
    "europa.eu", "wto.org", "nato.int", "coe.int", "icj-cij.org", "icc-cpi.int",
    "iaea.org", "ilo.org", "fao.org", "wfp.org", "unhcr.org", "unicef.org", "ohchr.org",
    "ipcc.ch", "bis.org", "osce.org", "interpol.int",
    # national statistics agencies without gov TLDs
    "destatis.de", "insee.fr", "istat.it", "ine.es", "ine.pt", "cbs.nl", "ssb.no",
    "scb.se", "dst.dk", "stat.fi",
    # central banks without gov TLDs (ecb.europa.eu is covered by europa.eu)
    "stlouisfed.org", "newyorkfed.org", "bankofengland.co.uk", "banque-france.fr",
    "bundesbank.de", "snb.ch", "riksbank.se", "norges-bank.no", "dnb.nl",
    "bancaditalia.it", "bde.es", "boj.or.jp", "rbi.org.in", "bankofcanada.ca",
    # courts / legal records without gov TLDs
    "courtlistener.com", "supremecourt.uk", "judiciary.uk", "bailii.org",
    # election authorities
    "electoralcommission.org.uk", "elections.ca", "idea.int",
    # legislatures / government portals without gov TLDs
    "parliament.uk", "assemblee-nationale.fr", "senat.fr", "elysee.fr", "gouvernement.fr",
    "bundestag.de", "bundesregierung.de", "bund.de", "canada.ca",
    "rijksoverheid.nl", "overheid.nl",
    # scholarly / journals of record (frontiersin stays trusted-tier only, not primary)
    "doi.org", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "nature.com",
    "sciencedirect.com", "springer.com", "nejm.org", "thelancet.com", "bmj.com",
    "jamanetwork.com", "science.org", "pnas.org", "cell.com", "nber.org",
}


def is_primary_source(domain: str) -> bool:
    """True if `domain` is an official-record producer (gov TLD patterns + allowlist)."""
    d = (domain or "").lower().removeprefix("www.")
    if not d:
        return False
    for s in _GOV_SUFFIXES:
        if d == s[1:] or d.endswith(s):
            return True
    if _GOV_CC.search(d):
        return True
    # .edu = institutional (Daniel 2026-07-21) — EXCEPT user-upload repositories that
    # merely live on an .edu-style domain
    if (d.endswith(".edu") or ".edu." in d) and not d.endswith("academia.edu"):
        return True
    return any(d == p or d.endswith("." + p) for p in PRIMARY_SOURCES)


# Evergreen/reference sources EXEMPT from the eval date-ceiling: their value is historical /
# institutional data, not time-stamped news that could leak a post-claim outcome. NARROWER than
# tier 2 — news wires (reuters/apnews/bbc) and fact-checkers ARE date-filtered, since they can
# report the outcome AFTER the claim (clog 260626).
_REFERENCE = {
    "who.int", "opcw.org", "worldbank.org", "data.worldbank.org", "un.org", "imf.org", "oecd.org",
    "macrotrends.net", "ourworldindata.org", "statista.com", "fred.stlouisfed.org",
    "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "doi.org", "nature.com", "sciencedirect.com",
    "springer.com", "frontiersin.org",
}


def is_reference(url: str) -> bool:
    """Evergreen/institutional/scholarly source — exempt from the date-ceiling (historical data,
    not post-claim news). gov/edu/.int/.ac.uk TLDs + curated data/scholarly orgs."""
    d = _domain(url)
    if d.endswith(_TRUSTED_TLDS) or ".gov." in d or ".edu." in d:
        return True
    return any(d == r or d.endswith("." + r) for r in _REFERENCE)
