"""Drive loop.verify_claim end to end with no network: scripted LLM JSON through the REAL
LLMPool (fake streaming client, so CallRecords/cache/usage are the production path), a fake
Serper returning a fixed hit list, the real ScrapePool over a dict of page texts, NewsGuard
scores pinned. The disk cache lives in tmp_path (conftest)."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from claimverify import loop
from claimverify.config import LOOP_VERSION, ClaimVerifyConfig, OrchestrationConfig
from claimverify.pools import LLMPool, Pools, ScrapePool, SerperPool
from claimverify.prompts import (
    QUERY_SYSTEM, READ_SYSTEM, RESOLVE_FINAL_NOTE, RESOLVE_SYSTEM, RESOLVE_SYSTEM_NO_BAR,
)

NG = {"reuters.com": 100.0, "apnews.com": 100.0, **{f"outlet{i}.com": 70.0 for i in range(1, 11)}}
CLAIM = {"claim_id": "c1", "date": "2020-05-20",
         "text": "The Riverside bridge reopened to traffic on 12 May 2020 after repairs.",
         "origin_url": "https://www.example-origin.com/post/1"}
COST_PER_CALL = 0.001


@pytest.fixture(autouse=True)
def _pin_newsguard(monkeypatch):
    monkeypatch.setattr(loop, "newsguard_score_map", lambda: NG)


# --- fakes -------------------------------------------------------------------------------

def chunk(content=None, finish=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)],
        usage=None)


def usage_chunk():
    return SimpleNamespace(choices=[], usage=SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None,
        model_extra={"estimated_cost": COST_PER_CALL}))


class FakeStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def close(self):
        pass


class Script:
    """One scripted answer per step: QUERY by round, READ stance by domain (no entry = no
    evidence), RESOLVE ledger row by round (None = no row; last row repeats)."""

    def __init__(self, queries, reads, resolves):
        self.queries, self.reads, self.resolves = list(queries), dict(reads), list(resolves)
        self.n_query = self.n_resolve = 0
        self.read_urls: list[str] = []

    def __call__(self, messages: list[dict]) -> dict:
        system, user = messages[0]["content"], messages[1]["content"]
        if system == QUERY_SYSTEM:
            q = self.queries[self.n_query]
            self.n_query += 1
            return {"targets": [1], "query": q}
        if system == READ_SYSTEM:
            doc, url = re.search(r"ARTICLE (D\d+) \((\S+)\)", user).groups()
            self.read_urls.append(url)
            stance = self.reads.get(loop._domain_of(url))
            ev = [{"claim_id": 1, "segs": [2, 3], "stance": stance}] if stance else []
            return {"docs": [{"doc": doc, "evidence": ev}]}
        if system in (RESOLVE_SYSTEM, RESOLVE_SYSTEM_NO_BAR):
            row = self.resolves[min(self.n_resolve, len(self.resolves) - 1)]
            self.n_resolve += 1
            return {"ledger": [row] if row else []}
        raise AssertionError(f"unscripted LLM step: {system[:60]!r}")


class FakeClient:
    def __init__(self, script: Script):
        self.script = script
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls += 1
        obj = self.script(kw["messages"])
        return FakeStream([chunk(json.dumps(obj)), chunk(finish="stop"), usage_chunk()])


class FakeSerper:
    def __init__(self, hits):
        self.hits = hits
        self.calls: list[dict] = []

    async def search_(self, query, top_k, *, date_ceiling=None, exclude_domains=None,
                      min_results=0, label="serper", trace=None):
        self.calls.append({"query": query, "top_k": top_k, "date_ceiling": date_ceiling,
                           "exclude_domains": exclude_domains})
        return [dict(h) for h in self.hits]


class FakeExa:
    async def search_(self, *a, **kw):
        raise AssertionError("Exa must not be called")


def hits_for(domains):
    return [{"url": f"https://{d}/story-{i}", "snippet": f"Snippet {i} on the bridge.",
             "date": "2020-05-13", "content": None} for i, d in enumerate(domains, 1)]


def page_text(i: int, dom: str) -> str:
    # every 8-word window carries the article index so no two pages are near-duplicates
    return "\n".join(f"Report {i}-{k} from {dom} says item {i} number {k} confirms bridge {i} "
                     f"reopening {k} in May 2020 with detail {i}." for k in range(1, 9))


def build(script: Script, hits: list[dict], use_cache: bool = False):
    ocfg = OrchestrationConfig()
    pages = {h["url"]: page_text(i, loop._domain_of(h["url"])) for i, h in enumerate(hits, 1)}
    client = FakeClient(script)
    pools = Pools(cfg=ocfg, llm=LLMPool(ocfg, client=client, use_cache=use_cache),
                  serper=FakeSerper(hits), exa=FakeExa(), scrape=ScrapePool(ocfg, scrape_fn=pages.get))
    return pools, client


def resolved(status, gap=None):
    row = {"claim_id": 1, "status": status}
    if gap:
        row["gap"] = gap
    return row


# --- cases -------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supported_two_voices_and_return_shape():
    script = Script(["riverside bridge reopened may 2020"],
                    {"outlet1.com": "supports", "outlet2.com": "supports"},
                    [resolved("supported")])
    pools, client = build(script, hits_for(["outlet1.com", "outlet2.com", "outlet3.com"]))
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig())
    assert r["status"] == "supported"
    assert r["verdict_raw"] == "Supported" and r["verdict_cc"] == "Supported"
    assert "2+ independent reliable voices" in r["resolution"]
    assert r["stopped"] == "resolved" and r["close_round"] == 1 and len(r["rounds"]) == 1
    assert r["guard_events"] == []
    assert [e["domain"] for e in r["evidence"]] == ["outlet1.com", "outlet2.com"]
    assert all("<<" in e["text"] for e in r["evidence"])          # cited sentences marked
    # shape
    assert set(r) >= {"status", "verdict_raw", "verdict_cc", "rounds", "evidence",
                      "guard_events", "serper_calls", "exa_calls", "scrapes", "llm_calls",
                      "llm_cache_hits", "tokens", "cost_usd", "elapsed_s", "loop_version"}
    assert r["loop_version"] == LOOP_VERSION == "v7.6-claim"
    assert r["serper_calls"] == 1 and r["exa_calls"] == 0 and r["scrapes"] == 3
    assert r["llm_calls"] == client.calls == 5                     # query + 3 reads + resolve
    assert r["tokens"] == {"prompt": 50, "cached": 0, "completion": 25}
    assert r["cost_usd"] == pytest.approx(5 * COST_PER_CALL)
    assert r["elapsed_s"] >= 0
    assert set(r["rounds"][0]) >= {"round", "provider", "query", "results", "docs", "dropped",
                                   "ledger", "new_evidence"}


@pytest.mark.asyncio
async def test_supported_single_strong_voice():
    script = Script(["riverside bridge reopened may 2020"], {"reuters.com": "supports"},
                    [resolved("supported")])
    pools, _ = build(script, hits_for(["reuters.com", "outlet1.com", "outlet2.com"]))
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig())
    assert r["status"] == "supported"
    assert "strong secondary (NG>=90)" in r["resolution"]


@pytest.mark.asyncio
async def test_refuted_close():
    script = Script(["riverside bridge reopened may 2020"],
                    {"reuters.com": "refutes", "apnews.com": "partially-refutes"},
                    [resolved("refuted")])
    pools, _ = build(script, hits_for(["reuters.com", "apnews.com", "outlet1.com"]))
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig())
    assert r["status"] == "refuted"
    assert r["verdict_raw"] == "Refuted" and r["verdict_cc"] == "Refuted"
    assert r["resolution"].startswith("refuted — ")


@pytest.mark.asyncio
async def test_budget_exhausted_is_unsupported():
    # round 1: the model tries to close on ONE mid-rated voice -> guard refuses, stays open;
    # round 2: still open -> code marks unsupported after tries_per_claim=2 targeted searches
    script = Script(["riverside bridge reopened may 2020", "mayor statement infrastructure repairs"],
                    {"outlet1.com": "supports"},
                    [resolved("supported"), resolved("open", gap="need the council record")])
    pools, _ = build(script, hits_for(["outlet1.com", "outlet2.com", "outlet3.com"]))
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig(tries_per_claim=2))
    assert r["status"] == "unsupported"
    assert r["verdict_raw"] == "Not Enough Evidence" and r["verdict_cc"] == "Refuted"
    assert r["stopped"] == "budget-exhausted" and r["tries"] == 2 and len(r["rounds"]) == 2
    guards = [g["guard"] for g in r["guard_events"]]
    assert guards == ["close-below-bar", "budget-exhausted"]
    assert r["rounds"][0]["ledger"] == {1: "open"} and r["rounds"][1]["ledger"] == {1: "unsupported"}
    assert r["gaps"] == "need the council record"
    assert r["coerced_open"] is False


@pytest.mark.asyncio
async def test_both_directions_qualify_keeps_the_models_verdict():
    """v7.6: the both-directions-qualify coercion to "conflicting" is gone (fired 16x on the
    AVeriTeC no-bar arm, right once). The RESOLVE verdict stands when it clears the bar."""
    script = Script(["riverside bridge reopened may 2020"],
                    {"reuters.com": "supports", "apnews.com": "refutes"},
                    [resolved("supported")])
    pools, _ = build(script, hits_for(["reuters.com", "apnews.com", "outlet1.com"]))
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig())
    assert r["status"] == "supported"
    assert r["guard_events"] == []


@pytest.mark.asyncio
async def test_no_bar_refuses_a_directional_close_with_no_evidence_and_an_open_gap():
    """Bar off: round 1 finds nothing in the claim's direction and leaves a gap; round 2 the
    model proposes "supported" with still no supporting read -> refused, the claim starves
    to unsupported. A first-round close (no gap recorded yet) is still accepted (fix 3 scope)."""
    script = Script(["riverside bridge reopened may 2020", "mayor statement infrastructure repairs"],
                    {"outlet1.com": "neutral"},
                    [resolved("open", gap="need the council record"), resolved("supported")])
    pools, _ = build(script, hits_for(["outlet1.com"]))
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig(code_bar=False, tries_per_claim=2))
    assert r["status"] == "unsupported"
    guards = [g["guard"] for g in r["guard_events"]]
    assert guards == ["close-no-direction-evidence", "budget-exhausted"]
    assert r["guard_events"][0]["gap"] == "need the council record"


@pytest.mark.parametrize("pages, expect", [(3, 3), (10, 10)])
@pytest.mark.asyncio
async def test_pages_per_round_reads_that_many(pages, expect):
    hits = hits_for([f"outlet{i}.com" for i in range(1, 11)])
    script = Script(["riverside bridge reopened may 2020"],
                    {f"outlet{i}.com": "supports" for i in range(1, 11)}, [resolved("supported")])
    pools, client = build(script, hits)
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig(pages_per_round=pages))
    assert r["status"] == "supported" and len(r["rounds"]) == 1
    assert [d["id"] for d in r["rounds"][0]["docs"]] == [f"D{i}" for i in range(1, expect + 1)]
    assert script.read_urls == [h["url"] for h in hits[:expect]]
    assert r["scrapes"] == expect and len(r["evidence"]) == expect
    assert client.calls == 1 + expect + 1
    assert len(r["rounds"][0]["results"]) == 10                     # the full hit list is kept


@pytest.mark.asyncio
async def test_llm_cache_replays_without_pool_calls():
    script = Script(["riverside bridge reopened may 2020"],
                    {"outlet1.com": "supports", "outlet2.com": "supports"}, [resolved("supported")])
    pools, client = build(script, hits_for(["outlet1.com", "outlet2.com", "outlet3.com"]),
                          use_cache=True)
    first = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig())
    assert first["llm_cache_hits"] == 0 and client.calls == first["llm_calls"] == 5

    from claimverify.trace import Trace
    trace = Trace("c1")
    second = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig(), trace)
    assert client.calls == 5                                        # no new network call
    assert second["llm_calls"] == second["llm_cache_hits"] == 5
    assert all(c["cache_hit"] for c in trace.llm_calls)
    assert all(c["usage"]["cost_source"] == "cache" for c in trace.llm_calls)
    assert second["cost_usd"] == 0.0
    assert second["status"] == first["status"] == "supported"
    assert pools.llm.metrics.cache_hits == 5


@pytest.mark.asyncio
async def test_origin_domain_excluded_from_retrieval_and_reads():
    hits = hits_for(["example-origin.com", "outlet1.com", "outlet2.com"])   # leaked origin hit
    script = Script(["riverside bridge reopened may 2020"],
                    {"example-origin.com": "supports", "outlet1.com": "supports",
                     "outlet2.com": "supports"}, [resolved("supported")])
    pools, _ = build(script, hits)
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig())
    call = pools.serper.calls[0]
    assert call["exclude_domains"] == ["example-origin.com"]
    assert call["date_ceiling"] == "2020-05-20" and call["top_k"] == 10
    assert r["origin_excluded"] == "example-origin.com"
    assert {"url": hits[0]["url"], "reason": "origin"} in r["rounds"][0]["dropped"]
    assert "example-origin.com" not in script.read_urls[0]
    assert {d["domain"] for d in r["rounds"][0]["docs"]} == {"outlet1.com", "outlet2.com"}
    assert {e["domain"] for e in r["evidence"]} == {"outlet1.com", "outlet2.com"}
    assert r["status"] == "supported"


class SegScript(Script):
    """READ that numbers its pointers "S2" instead of 2, plus one entry whose segs are junk."""

    def __call__(self, messages):
        if messages[0]["content"] == READ_SYSTEM:
            doc, url = re.search(r"ARTICLE (D\d+) \((\S+)\)", messages[1]["content"]).groups()
            self.read_urls.append(url)
            return {"docs": [{"doc": doc, "evidence": [
                {"claim_id": 1, "segs": ["S2", "3"],
                 "stance": self.reads[loop._domain_of(url)]},
                {"claim_id": 1, "segs": ["intro"], "stance": "supports"}]}]}
        return super().__call__(messages)


@pytest.mark.asyncio
async def test_string_segs_are_coerced_and_seg_less_entries_are_flagged():
    script = SegScript(["riverside bridge reopened may 2020"],
                       {"outlet1.com": "supports", "outlet2.com": "supports"},
                       [resolved("supported")])
    pools, _ = build(script, hits_for(["outlet1.com", "outlet2.com"]))
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig())
    assert r["status"] == "supported"                       # the "S2"/"3" pointers survived
    assert [e["segs"] for e in r["evidence"]] == [[2, 3], [2, 3]]
    assert r["guard_events"] == [
        {"round": 1, "guard": "seg-coerced", "doc": "D1", "raw": ["S2", "3"]},
        {"round": 1, "guard": "evidence-dropped-no-segs", "doc": "D1"},
        {"round": 1, "guard": "seg-coerced", "doc": "D2", "raw": ["S2", "3"]},
        {"round": 1, "guard": "evidence-dropped-no-segs", "doc": "D2"}]


@pytest.mark.asyncio
async def test_last_serper_round_is_final_when_exa_is_off():
    from claimverify.trace import Trace
    script = Script(["riverside bridge reopened may 2020", "council record bridge repairs"],
                    {}, [resolved("open", gap="no coverage found"), resolved("unsupported")])
    pools, _ = build(script, hits_for(["outlet1.com", "outlet2.com", "outlet3.com"]))
    trace = Trace("c1")
    r = await loop.verify_claim(CLAIM, pools,
                                ClaimVerifyConfig(exa_enabled=False, tries_per_claim=2), trace)
    assert len(r["rounds"]) == 2 and r["status"] == "unsupported"
    notes = [RESOLVE_FINAL_NOTE in c["messages"][1]["content"]
             for c in trace.llm_calls if c["label"].startswith("resolve-")]
    assert notes == [False, True]
    assert [t["final"] for t in trace.resolves] == [False, True]


class CappedReadClient(FakeClient):
    """Streams a truncated body then finish_reason="length" for READ on one doc — the
    max_tokens cap the live top3_noceil run hit on a 28-page court PDF."""

    def __init__(self, script: Script, cap_domain: str):
        super().__init__(script)
        self.cap_domain = cap_domain

    async def _create(self, **kw):
        msgs = kw["messages"]
        if msgs[0]["content"].startswith(READ_SYSTEM) and self.cap_domain in msgs[1]["content"]:
            self.calls += 1
            return FakeStream([chunk('{"docs": [{"doc": "D1", "evidence": [{"claim_id": 1,'),
                               chunk(finish="length"), usage_chunk()])
        return await super()._create(**kw)


@pytest.mark.asyncio
async def test_read_length_cap_drops_the_doc_and_the_claim_still_closes():
    hits = hits_for(["outlet1.com", "outlet2.com", "outlet3.com"])
    script = Script(["riverside bridge reopened may 2020"],
                    {"outlet1.com": "supports", "outlet2.com": "supports",
                     "outlet3.com": "supports"},
                    [resolved("supported")])
    ocfg = OrchestrationConfig()
    pages = {h["url"]: page_text(i, loop._domain_of(h["url"])) for i, h in enumerate(hits, 1)}
    client = CappedReadClient(script, "outlet1.com")
    pools = Pools(cfg=ocfg, llm=LLMPool(ocfg, client=client, use_cache=False),
                  serper=FakeSerper(hits), exa=FakeExa(),
                  scrape=ScrapePool(ocfg, scrape_fn=pages.get))
    from claimverify.trace import Trace
    trace = Trace("c1")
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig(), trace)
    # the capped doc contributes nothing; the other two close the claim
    assert r["status"] == "supported"
    assert [e["domain"] for e in r["evidence"]] == ["outlet2.com", "outlet3.com"]
    assert {"round": 1, "guard": "length-cap", "label": "read-D1", "doc": "D1",
            "action": "doc-dropped"} in r["guard_events"]
    # the doc is still recorded, flagged as a failed read, and READ is never re-asked
    d1 = next(d for d in r["rounds"][0]["docs"] if d["id"] == "D1")
    assert d1["read"]["read_status"] == "length_cap"
    assert not any(c["label"].endswith("-brief") for c in trace.llm_calls)


@pytest.mark.asyncio
async def test_origin_exclusion_off_passes_no_exclusions():
    script = Script(["riverside bridge reopened may 2020"], {}, [None])
    pools, _ = build(script, hits_for(["outlet1.com"]))
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig(origin_exclusion=False,
                                                                date_ceiling=False, tries_per_claim=1))
    call = pools.serper.calls[0]
    assert call["exclude_domains"] is None and call["date_ceiling"] is None
    assert r["origin_excluded"] is None and r["status"] == "unsupported"


class _FakeHttp:
    """Minimal stand-in for the pooled httpx client: one canned Serper JSON body."""

    def __init__(self, data):
        self._data = data

    async def post(self, url, json=None, headers=None):
        return SimpleNamespace(status_code=200, headers={}, text="", json=lambda: self._data)


@pytest.mark.parametrize("code_bar, status, guards", [
    (True, "unsupported", ["close-below-bar", "budget-exhausted"]),
    (False, "supported", ["bar-off-accepted"]),
])
@pytest.mark.asyncio
async def test_code_bar_switch(code_bar, status, guards):
    """One UNRATED voice, RESOLVE proposes "supported": the code bar refuses the close and
    the claim starves to unsupported; with the bar off the proposed close stands, traceably."""
    script = Script(["riverside bridge reopened may 2020"], {"blogsite.com": "supports"},
                    [resolved("supported")])
    pools, _ = build(script, hits_for(["blogsite.com"]))
    r = await loop.verify_claim(CLAIM, pools,
                                ClaimVerifyConfig(code_bar=code_bar, tries_per_claim=1))
    assert r["status"] == status
    assert [g["guard"] for g in r["guard_events"]] == guards
    if not code_bar:
        assert r["guard_events"][0] == {"round": 1, "guard": "bar-off-accepted",
                                        "claim_id": 1, "proposed": "supported"}


@pytest.mark.parametrize("blocklist, urls", [
    (True, ["https://outlet1.com/1"]),
    (False, ["https://facebook.com/page/1", "https://outlet1.com/1"]),
])
@pytest.mark.asyncio
async def test_ugc_blocklist_switch(blocklist, urls):
    """Off: no -site: suffix in the query Serper receives, and a blocklisted domain
    survives the client-side finalize drop."""
    from claimverify.trace import Trace
    organic = [{"link": "https://facebook.com/page/1", "snippet": "s", "date": "May 13, 2020"},
               {"link": "https://outlet1.com/1", "snippet": "s", "date": "May 13, 2020"}]
    pool = SerperPool(OrchestrationConfig(), blocklist=blocklist,
                      http=_FakeHttp({"organic": organic}))
    trace = Trace("c1")
    hits = await pool.search_("bridge reopened", 10, date_ceiling="2020-05-20", trace=trace)
    assert [h["url"] for h in hits] == urls
    q = trace.searches[0]["payload"]["q"]
    assert ("-site:" in q) is blocklist
    assert trace.searches[0]["blocklist"] is blocklist


@pytest.mark.parametrize("drop, read_domains", [
    (True, {"outlet1.com", "outlet2.com"}),
    (False, {"snopes.com", "outlet1.com", "outlet2.com"}),
])
@pytest.mark.asyncio
async def test_fc_undated_drop_switch(drop, read_domains):
    """An UNDATED hit from a fact-check host is dropped before snippets and reads when the
    switch is on (the urn run's fc-undated rule); a dated one is never touched."""
    hits = hits_for(["snopes.com", "outlet1.com", "outlet2.com"])
    hits[0]["date"] = None
    script = Script(["riverside bridge reopened may 2020"],
                    {"snopes.com": "refutes", "outlet1.com": "supports", "outlet2.com": "supports"},
                    [resolved("supported")])
    pools, _ = build(script, hits)
    r = await loop.verify_claim(CLAIM, pools, ClaimVerifyConfig(fc_undated_drop=drop))
    assert {d["domain"] for d in r["rounds"][0]["docs"]} == read_domains
    guards = [g["guard"] for g in r["guard_events"]]
    assert ("fc-undated-dropped" in guards) is drop
    if drop:
        assert not any("snopes.com" in (e.get("domain") or "") for e in r["evidence"])
