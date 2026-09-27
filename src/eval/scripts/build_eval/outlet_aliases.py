"""Origin-domain aliases for the outlet TRUE urn.

`evidence_urn_run` excludes a claim's origin publisher so an outlet cannot verify
its own claim. That exclusion is domain-literal, and outlets publish the same copy
under more than one domain, so it leaked. Measured case (2026-08-27): an AP claim
excluded apnews.com and then retrieved **ap.org** reading flag 5, which is the
outlet corroborating itself with an extra step.

WHY THE LIST IS CAPPED. search.serper_payload sends the caller's exclusions as
server-side `-site:` operators only while there are at most `_SERVER_XD_MAX` = 3 of
them, and drops the whole set to client-side past that. The urn runs with
min_results=0, so there is NO page-2 backfill: a client-side drop costs a result
slot outright. Sending a long alias list would therefore cut documents-per-claim
for exactly the outlets that have aliases, which correlates with wire status, which
is the variable the outlet-versus-timeline comparison is testing. So we send at most
`SERVER_SLOTS` aliases, priority-ordered, and keep the full table for `audit()`.

`x.com` is NOT included. x.com and twitter.com are already in SCRAPE_BLOCKLIST,
which is sent server-side inside the social budget AND dropped client-side in
serper_finalize, so listing it again only burns one of the three caller slots.

Two rules, both conservative.

TLD twin. Same second-level label, different TLD, is nearly always the same
organisation (huffpost.com / huffpost.org). Only `.org` is expanded, and only for
labels of four characters or more, so short ambiguous labels like `vox` are left
alone.

ALIASES. A curated table for the 22 outlets actually in the urn, priority-ordered by
how likely the domain is to surface in organic results. Only domains that host the
SAME copy are listed. Sibling brands under a shared parent are NOT listed, because
they publish different articles and blocking them would suppress genuine independent
corroboration, which biases the support rate the other way.

This does NOT solve syndication. Blocking ap.org does nothing about wbur, ktsm, npr
and thehill each republishing the same AP wire item under their own name and each
being counted as an independent RELIABLE voice. That needs cross-domain near-duplicate
detection at the document level and is separate work.
"""
from __future__ import annotations

SERVER_SLOTS = 2      # publisher_site takes the third of search._SERVER_XD_MAX

ALIASES: dict[str, tuple[str, ...]] = {
    # VERIFIED leak. ap.org is AP's own corporate and content site.
    "apnews.com": ("ap.org", "associatedpress.com"),
    # Reuters' own distribution platforms carry the identical wire item.
    "reuters.com": ("reutersagency.com", "thomsonreuters.com", "reutersconnect.com"),
    # One editorial operation, heavy verbatim cross-posting. medscape ranks organically.
    "webmd.com": ("medscape.com", "medicinenet.com", "rxlist.com", "emedicinehealth.com"),
    # DCNF is the Caller's own wire arm and its copy is republished verbatim.
    "dailycaller.com": ("dailycallernewsfoundation.org",),
    # Legacy domains, same articles still resolve there.
    "huffpost.com": ("huffingtonpost.com",),
    "scientificamerican.com": ("sciam.com",),
    # Hearst international edition runs the same features and ranks on fitness queries.
    "menshealth.com": ("menshealth.co.uk",),
    # Parent site hosts some of the same pieces.
    "vox.com": ("voxmedia.com",),
}

# Deliberately NOT aliased, recorded so the decision is visible rather than an omission.
#   thepostmillennial.com / humanevents.com   sibling brands, distinct copy
#   nydailynews.com and other Tribune papers  distinct copy
#   rollingstone.com and other PMC titles     distinct copy
#   theintercept.com, thedailybeast.com, salon.com, thenation.com, thedispatch.com,
#   psychologytoday.com, axios.com, thehill.com, nationalreview.com,
#   washingtonexaminer.com, occupydemocrats.com    no same-copy twin found


def _norm(domain: str) -> str:
    return domain.lower().removeprefix("www.")


def expand(domain: str) -> list[str]:
    """Every domain that may host this outlet's own copy, priority-ordered.

    The origin domain itself is excluded separately by evidence_urn_run, and
    search._drop_domains already handles www and subdomains, so neither is repeated.
    """
    domain = _norm(domain)
    out = list(ALIASES.get(domain, ()))
    label = domain.split(".", 1)[0]
    if len(label) >= 4 and (twin := f"{label}.org") != domain and twin not in out:
        out.append(twin)
    return out


def server_side(domain: str) -> list[str]:
    """The slice that fits the server-side operator budget. Used by the runner."""
    return expand(domain)[:SERVER_SLOTS]


def audit(rows) -> dict[str, int]:
    """Post-run check: did any alias we did NOT send server-side still get through?

    rows is an iterable of dicts with `domain` (the claim's origin outlet) and
    `retrieved` (the list of domains its reads came from). A non-empty result means
    the cap cost us something and the search-side split is worth building.
    """
    hits: dict[str, int] = {}
    for r in rows:
        tail = set(expand(r["domain"])[SERVER_SLOTS:])
        for d in r["retrieved"]:
            d = _norm(d)
            if d in tail or any(d.endswith("." + t) for t in tail):
                hits[f'{r["domain"]} <- {d}'] = hits.get(f'{r["domain"]} <- {d}', 0) + 1
    return dict(sorted(hits.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    import polars as pl
    d = pl.read_parquet("eval/data/tweet_corpus/true_urn_assertions.parquet")
    print(f"{'handle':20s} {'origin':28s} {'n':>4}  server-side | client-only tail")
    for handle, dom, n in d.group_by(["handle", "domain"]).len().sort(
            "len", descending=True).iter_rows():
        srv, tail = server_side(dom), expand(dom)[SERVER_SLOTS:]
        print(f"{handle:20s} {dom:28s} {n:>4}  {', '.join(srv) or '(none)':45s}"
              f" | {', '.join(tail) or '-'}")
