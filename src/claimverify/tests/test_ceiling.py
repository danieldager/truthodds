"""Claim-date ceiling: AVeriTeC D-M-YYYY -> ISO -> Google tbs (unpadded M/D/YYYY, never DD/MM),
and cache-key parity with pipeline/search.py so Serper entries are shared."""
from __future__ import annotations

import pytest

from claimverify import retrieval
from claimverify.config import OrchestrationConfig
from claimverify.pools import SerperPool
from claimverify.run_averitec import _iso
from claimverify.trace import Trace
from pipeline import search as pipeline_search


@pytest.mark.parametrize("iso, tbs", [
    ("2020-10-31", "cdr:1,cd_min:1/1/1900,cd_max:10/31/2020"),
    ("2020-10-01", "cdr:1,cd_min:1/1/1900,cd_max:10/1/2020"),
    ("2020-01-05", "cdr:1,cd_min:1/1/1900,cd_max:1/5/2020"),   # DD/MM would read as 5 Jan
])
def test_tbs_date_ceiling(iso, tbs):
    assert retrieval.tbs_date_ceiling(iso) == tbs


def test_tbs_unparseable_date_is_omitted():
    assert retrieval.tbs_date_ceiling("31-10-2020") == ""
    assert "tbs" not in retrieval.serper_payload("q", 10, "31-10-2020")
    assert "tbs" not in retrieval.serper_payload("q", 10, None)


def test_serper_payload_carries_tbs():
    p = retrieval.serper_payload("bridge reopened", 10, "2020-10-31", ["example.com"])
    assert p["tbs"] == "cdr:1,cd_min:1/1/1900,cd_max:10/31/2020"
    assert p["q"].startswith("bridge reopened -site:") and p["q"].endswith(" -site:example.com")
    assert p["num"] == 10


@pytest.mark.parametrize("averitec, iso", [
    ("31-10-2020", "2020-10-31"),
    ("1-10-2020", "2020-10-01"),
    ("5-1-2020", "2020-01-05"),
])
def test_averitec_date_to_iso(averitec, iso):
    assert _iso(averitec) == iso


@pytest.mark.parametrize("query, k, ceiling, xd, min_results", [
    ("bridge reopened", 10, "2020-10-31", None, 0),
    ("bridge reopened", 10, None, ["example.com"], 0),
    ("bridge reopened", 10, "2020-10-31", ["b.com", "a.com"], 5),
    ("bridge reopened", 5, "2020-10-01", ["a.com", "b.com", "c.com", "d.com"], 0),  # > server max
])
def test_cache_key_matches_pipeline_search(query, k, ceiling, xd, min_results):
    assert retrieval.cache_key("serper", query, k, ceiling, xd, min_results) == \
        pipeline_search.cache_key("serper", query, k, ceiling, xd, min_results)


def test_cache_key_differs_when_the_blocklist_is_off():
    """The no-blocklist arm sends a different query string, so it must never replay a
    blocklisted run's cached Serper call."""
    args = ("serper", "bridge reopened", 10, "2020-10-31", ["example.com"], 0)
    assert retrieval.cache_key(*args) != retrieval.cache_key(*args, blocklist=False)
    assert retrieval.cache_key(*args, blocklist=True) == retrieval.cache_key(*args)
    assert retrieval.serper_payload("bridge reopened", 10, None, blocklist=False)["q"] == \
        "bridge reopened"


class _FakeResponse:
    status_code, headers, text = 200, {}, ""

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _FakeHttp:
    def __init__(self, data):
        self._data = data

    async def post(self, url, json=None, headers=None):
        return _FakeResponse(self._data)


@pytest.mark.asyncio
async def test_search_trace_counts_hits_past_the_ceiling():
    """Measurement only: hits dated after the ceiling are counted, never dropped."""
    organic = [{"link": "https://a.com/1", "snippet": "s", "date": "May 13, 2020"},
               {"link": "https://b.com/2", "snippet": "s", "date": "Jun 2, 2020"},
               {"link": "https://c.com/3", "snippet": "s", "date": "3 days ago"},
               {"link": "https://d.com/4", "snippet": "s", "date": ""}]
    pool = SerperPool(OrchestrationConfig(), http=_FakeHttp({"organic": organic}))
    trace = Trace("c1")
    hits = await pool.search_("bridge reopened", 10, date_ceiling="2020-05-20", trace=trace)
    assert len(hits) == 4                                   # nothing filtered
    assert trace.searches[0]["post_ceiling_hits"] == 1      # Jun 2 > May 20
    assert trace.searches[0]["undated_hits"] == 2           # "3 days ago" and ""
