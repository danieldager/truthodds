"""Source credibility (trimmed from pipeline/credibility.py): primary-source detection and
the trusted fact-checker roster used by the corroboration bar. The v0.3-era tier/rerank/
is_reference helpers are not part of the v7.5 loop and are not carried."""
from __future__ import annotations

import re

TRUSTED_FACTCHECKERS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "leadstories.com",
    "checkyourfact.com", "factcheckni.org",
    "factcheck.afp.com", "apnews.com/ap-fact-check", "reuters.com/fact-check",
    "africacheck.org", "dubawa.org", "pesacheck.org", "ghanafact.com",
    "boomlive.in", "altnews.in", "factly.in", "newsmeter.in", "newschecker.in",
    "vishvasnews.com", "factcrescendo.com", "rumorscanner.com",
    "verafiles.org", "tsek.ph", "blackdotresearch.sg", "aap.com.au",
    "correctiv.org", "maldita.es", "newtral.es", "pagellapolitica.it", "facta.news",
    "teyit.org", "faktisk.no", "demagog.org.pl", "demagog.cz", "mimikama.org",
    "euvsdisinfo.eu", "factuel.afp.com",
    "chequeado.com", "aosfatos.org", "lupa.uol.com.br", "colombiacheck.com",
    "animalpolitico.com",
}


# Fact-check hosts for the date-leak rule (pipeline/config.py FACT_CHECK_DOMAINS plus the
# domain-only trusted fact-checkers above). An UNDATED page from one of these cannot be
# placed against the claim date and carries a verdict by genre.
FACT_CHECK_DOMAINS = {f for f in TRUSTED_FACTCHECKERS if "/" not in f} | {
    "afp.com", "20minutes.fr", "truthorfiction.com", "healthfeedback.org",
    "sciencefeedback.co", "climatefeedback.org", "logically.ai", "misbar.com",
    "dpa-factchecking.com",
}


def is_factcheck_domain(domain: str) -> bool:
    d = (domain or "").lower().removeprefix("www.")
    return any(d == f or d.endswith("." + f) for f in FACT_CHECK_DOMAINS)


def is_trusted_factchecker(domain: str) -> bool:
    d = (domain or "").lower().removeprefix("www.")
    return any(d == f or d.endswith("." + f) for f in TRUSTED_FACTCHECKERS if "/" not in f)


_GOV_SUFFIXES = (
    ".gov", ".mil", ".int",
    ".gouv.fr", ".gc.ca", ".go.jp", ".go.kr", ".go.th", ".go.id", ".go.ke",
    ".gob.es", ".gob.mx", ".gob.ar", ".gub.uy", ".govt.nz", ".gv.at", ".admin.ch",
)
_GOV_CC = re.compile(r"(^|\.)gov\.[a-z]{2}$")

PRIMARY_SOURCES = {
    "un.org", "who.int", "imf.org", "worldbank.org", "data.worldbank.org", "oecd.org",
    "europa.eu", "wto.org", "nato.int", "coe.int", "icj-cij.org", "icc-cpi.int",
    "iaea.org", "ilo.org", "fao.org", "wfp.org", "unhcr.org", "unicef.org", "ohchr.org",
    "ipcc.ch", "bis.org", "osce.org", "interpol.int",
    "destatis.de", "insee.fr", "istat.it", "ine.es", "ine.pt", "cbs.nl", "ssb.no",
    "scb.se", "dst.dk", "stat.fi",
    "stlouisfed.org", "newyorkfed.org", "bankofengland.co.uk", "banque-france.fr",
    "bundesbank.de", "snb.ch", "riksbank.se", "norges-bank.no", "dnb.nl",
    "bancaditalia.it", "bde.es", "boj.or.jp", "rbi.org.in", "bankofcanada.ca",
    "courtlistener.com", "supremecourt.uk", "judiciary.uk", "bailii.org",
    "electoralcommission.org.uk", "elections.ca", "idea.int",
    "parliament.uk", "assemblee-nationale.fr", "senat.fr", "elysee.fr", "gouvernement.fr",
    "bundestag.de", "bundesregierung.de", "bund.de", "canada.ca",
    "rijksoverheid.nl", "overheid.nl",
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
    if (d.endswith(".edu") or ".edu." in d) and not d.endswith("academia.edu"):
        return True
    return any(d == p or d.endswith("." + p) for p in PRIMARY_SOURCES)
