"""origin_domain: host without www, web.archive.org wrappers unwrapped, socials kept as-is."""
from __future__ import annotations

import pytest

from claimverify.origin import origin_domain


@pytest.mark.parametrize("url, expected", [
    ("https://www.bbc.co.uk/news/world-1", "bbc.co.uk"),
    ("http://example.com/a?b=1", "example.com"),
    ("https://web.archive.org/web/20200101000000/https://www.example.com/story", "example.com"),
    ("https://web.archive.org/web/2020id_/http://example.org/x", "example.org"),
    ("https://archive.ph/abc12", "archive.ph"),               # cannot unwrap: itself
    ("https://twitter.com/user/status/123", "twitter.com"),
    ("https://www.facebook.com/page/posts/1", "facebook.com"),
    ("https://m.facebook.com/page/posts/1", "m.facebook.com"),  # only www is stripped
    (None, None),
    ("", None),
])
def test_origin_domain(url, expected):
    assert origin_domain(url) == expected
