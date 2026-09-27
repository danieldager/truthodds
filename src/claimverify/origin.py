"""Origin-domain helper shared by both comparison legs (claimverify and the urn adapter).

The origin of an AVeriTeC claim is the site of `original_url`. Archive wrappers
(web.archive.org/web/<ts>/<url>) are unwrapped to the archived host — the origin is
the archived site, not the archive. Other archives (archive.ph, perma.cc, archive.vn)
cannot be unwrapped and are returned as themselves. None for a null/empty URL.
"""
from __future__ import annotations

from urllib.parse import urlparse


def origin_domain(url: str | None) -> str | None:
    if not url:
        return None
    u = url
    if urlparse(u).netloc.lower().endswith("web.archive.org"):
        inner = u.split("/web/", 1)[-1]
        inner = inner[inner.find("http"):] if "http" in inner else ""
        u = inner or u
    host = urlparse(u).netloc.lower().removeprefix("www.")
    return host or None
