"""Text normalisation for harvested post/claim text.

Resolved post text (og:description scrapes, syndication payloads, fact-check HTML) arrives with raw HTML
entities (`&#039;` → ', `&quot;` → ", `&amp;` → &) and stray non-breaking spaces. Left un-decoded they
leak into the text we feed the audit, the filter, and the extractor. `clean_text` decodes entities,
canonicalises Unicode (NFC), and flattens non-breaking spaces — applied at parse time and to the
existing harvests.
"""
from __future__ import annotations

import html
import unicodedata


def clean_text(s):
    """Decode HTML entities + NFC-normalise + flatten NBSP. Pass-through for non-strings/None."""
    if not isinstance(s, str):
        return s
    for _ in range(3):                        # &#039; -> ' , &quot; -> " , &amp; -> &
        u = html.unescape(s)                  # some feeds are double-escaped (&amp;#8217;),
        if u == s:                            # so decode to a fixpoint, bounded
            break
        s = u
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\xa0", " ").replace("​", "")  # nbsp -> space, strip zero-width space
    return s
