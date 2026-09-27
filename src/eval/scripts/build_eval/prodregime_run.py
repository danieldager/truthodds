"""Production-regime re-evaluation of fc_gold recall, two retrieval arms (Daniel 2026-09-15).

WHY. Three audits on 2026-09-15 (miss_decomp, fp_decomp, kimi_disagree) show the recall gap
on fc_gold is mostly RETRIEVAL-bound: ~45% of false claims have no refuting document and 160
true claims sit at the all-irrelevant score. Production runs with NO date ceiling. This runner
re-measures under the production regime (ceiling OFF, leak control by echo pass instead of the
date rule) on the fitting population fc_gold_rep1500 (750T/750F), with two retrieval arms, so it
becomes the new baseline for every later lever.

ARMS (pre-registered).
  Arm S (production baseline): the production Serper query VERBATIM (reused from the pinned
    e1_ctx run, i.e. query-v3), date ceiling OFF, 10 results, production scrape+prep, read-v6.1
    with the production claim block (no bridge). Origin/reviewing publisher still excluded
    server-side (that is production leak control, not the date ceiling).
  Arm E (two-hop): a NEW query-writer (one Flash call/claim) emitting JSON {target, bearing,
    exa_query}. Exa neural search, 10 results, no date bound. Same scrape+prep. Read TWICE:
    E-bridge = read-v6.1 with the bridge appended to the claim block; E-plain = read-v6.1 with
    the production claim block. Query prompt is abstract-principles only (no dataset examples).
  Union U (scoring-time only): S docs + E docs deduped by URL, <=20.

LEAK CONTROL (all arms). Drop fc_domain documents; then an ECHO PASS (one cheap Flash call/doc:
"does this document report, cite or summarise a fact-check verdict on this claim?"). Echo-yes
docs are excluded from scoring but kept in the record. The old date-leak flag is still computed
and recorded (_is_leak vs the claim's saved ceiling) but NOT applied.

Everything reuses the production chain: evidence_urn_run.select_regions / read_doc claim-block /
_is_leak / fc_domain tagging; the Exa fetch pattern from exa_probe; reader_lab's RateGate for the
reads. Nothing under urn_runs/ is modified; output -> eval/data/urn_runs/e1_prodregime/.

Reads/echo/query all go through reader_lab.gated_post under a RateGate (target 160/ceiling 190).
Serper and Exa requests are counted (cache-miss = paid) and hard-capped.

  # SMOKE (20 claims, 10T/10F seed 20260915), $1 cap, both arms, both read conditions, echo:
  uv run python -m eval.scripts.build_eval.prodregime_run --smoke
  # FULL LEG (only after Daniel approves):
  uv run python -m eval.scripts.build_eval.prodregime_run --full --preflight
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from pipeline import disk_cache  # noqa: E402
from pipeline.config import FACT_CHECK_DOMAINS  # noqa: E402
from pipeline.credibility import TRUSTED_FACTCHECKERS  # noqa: E402
from pipeline.search import SearchError, cache_key, scrape, search  # noqa: E402
from pipeline.verify_tweet_claims import _domain_of  # noqa: E402
from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval import reader_lab  # noqa: E402
from eval.scripts.build_eval.read_v5_prompts import READ_SYS_V6_1 as READ_SYS  # noqa: E402
from eval.scripts.build_eval.reader_lab_prompts import PROMPTS as _READ_PROMPTS  # noqa: E402

POP = SRC / "eval/data/populations/fc_gold_rep1500.parquet"
E1_DOCS = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"   # claim metadata + production query
OUT_DIR = SRC / "eval/data/urn_runs/e1_prodregime"
FC_DOMS = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}
TOP_K = 10
SNIPPET_MIN_TEXT = eur.SNIPPET_MIN_TEXT
SMOKE_SEED = 20260915

# -------------------------------------------------------------------- Exa / Serper budgets
EXA_PACE = 0.125   # Exa documents 10 QPS on /search; 8/s leaves margin (was 0.75, a frugality guard inherited from exa_probe)


class ExaBudgetExhausted(Exception):
    """Raised when the Exa soft budget (5% margin under the cap) is reached: arm E skips this
    claim rather than crashing the run or exceeding the cap."""


class ApiCounter:
    """Persisted paid-request counter with a hard cap; take() BEFORE the request leaves so a
    crash over-counts. Cache hits never call it. Mirrors exa_probe.ExaBudget."""

    def __init__(self, path: Path, cap: int, pace: float = 0.0):
        self.path, self.cap, self.pace = path, cap, pace
        self.lock = threading.Lock()
        self.last = 0.0
        self.n = json.loads(path.read_text())["n"] if path.exists() else 0

    def take(self, soft: bool = False) -> int | None:
        with self.lock:
            if self.n >= self.cap:
                if soft:
                    return None      # graceful: caller skips this request instead of crashing
                raise SystemExit(f"API CAP {self.cap} REACHED ({self.n} spent) on {self.path.name}")
            self.n += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"n": self.n, "cap": self.cap}))
            if self.pace:
                wait = self.pace - (time.time() - self.last)
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
            return self.n


# -------------------------------------------------------------------- prompts
# Arm-E query writer. Abstract principles only; NO dataset-derived or structure-matching
# examples (standing rule). Two-hop: name the proposition that settles the claim (target), the
# logical bearing, and a neural-search description of the document that would contain the target.
E_QUERY_SYS = (
    "You are preparing to fact-check ONE claim by finding independent evidence about the world, "
    "not commentary on the claim. First work out the single underlying fact that, once settled, "
    "decides whether the claim is true or false. Then describe the document that would report "
    "that fact.\n\n"
    "STRICT GROUNDING RULE. Use ONLY the specific people, organisations, places, numbers and dates "
    "that appear in the claim text you are given. Do NOT introduce any name, number, count, date or "
    "place that is not present in the claim. If the claim states no date, do not supply one; if it "
    "gives no count, do not invent one. When you are unsure of a specific, leave it out rather than "
    "guess. Never add your own knowledge of when or where something happened.\n\n"
    "Respond JSON only, three fields:\n"
    '{"target": "...", "bearing": "...", "exa_query": "..."}\n\n'
    "target — the proposition whose truth settles the claim, stated in the plain words a document "
    "reporting on it would use, using only specifics drawn from the claim. If the claim is already "
    "directly checkable against ordinary reporting, the target may restate the claim itself. Do not "
    "hedge or generalise, and do not add any specific the claim does not contain.\n"
    "bearing — one or two sentences making the link explicit for THIS claim: if the target holds "
    "the claim is true or false, and if the target fails the claim is the opposite, because ... "
    "State the direction concretely, not in the abstract.\n"
    "exa_query — a natural-language description of the document that would contain the target, "
    "written the way a person describes an article they are looking for: what it reports and about "
    "whom. It is for a neural search engine that matches on meaning, so write one descriptive "
    "sentence, no keyword lists, no boolean operators, no quotation marks, no site filters. Do NOT "
    "specify when the document was published and do NOT restrict it to the time of the claim: the "
    "evidence that settles a claim — especially a refutation — is usually reported LATER, so allow "
    "reporting from any time. Mention a date only if the claim itself is about a specific dated "
    "event, and even then describe the event, not the document's publication date. Do not ask "
    "whether the claim is true and do not describe a fact-check of it; describe the underlying "
    "reporting or record.")
E_QUERY_V = "prodregime-eq-v2"   # v2: Fix 2 (strict grounding, no invented specifics) +
                                 # Fix 3 (no date anchoring, allow later-published evidence)

# Echo pass. One cheap call per document: is this document itself a fact-check verdict on this
# exact claim (a leak of the answer key)? Abstract, no examples.
ECHO_SYS = (
    "You are screening a web document for answer-key leakage before it is used as evidence. You "
    "get one claim and a document. Answer ONE question: does this document report, cite, quote or "
    "summarise a FACT-CHECK VERDICT on THIS SPECIFIC claim (for example a rating such as true, "
    "false, misleading, or a fact-checker's conclusion about this exact claim)? Reporting the "
    "underlying facts, events or figures is NOT a fact-check verdict; only an explicit assessment "
    'of the claim\'s truth counts.\n\n'
    'Respond JSON only: {"echo": "yes" or "no", "reason": "<one short line>"}')
ECHO_V = "prodregime-echo-v1"

# read-v6.1 claim blocks. PLAIN = the production block (byte-identical to evidence_urn_run's).
# BRIDGE = plain + the two-hop bridge lines. The system prompt (READ_SYS) is unchanged in both.


def claim_block_plain(claim: str, claim_date: str | None) -> str:
    return (f"CLAIM: {claim}\n"
            f"(claimed on {claim_date or 'unknown date'}; judge the "
            f"document's bearing on this exact proposition)")


def claim_block_bridge(claim: str, claim_date: str | None, target: str, bearing: str) -> str:
    return (f"CLAIM: {claim}\n"
            f"Verification target: {target}\n"
            f"Bearing: {bearing}\n"
            f"(claimed on {claim_date or 'unknown date'}; judge the "
            f"document's bearing on this exact proposition)")


# Hash the read instrument for the record: the system prompt + the claim-block template shape.
READ_HASH = prompt_hash(READ_SYS)
READ_LABEL = "read-v6.1"   # reader prompt id; overridden by --read-prompt in main()
BRIDGE_TEMPLATE_HASH = prompt_hash(
    READ_SYS + "\n<<BRIDGE>>CLAIM/Verification target/Bearing/(claimed on ...)")
PLAIN_TEMPLATE_HASH = prompt_hash(READ_SYS + "\n<<PLAIN>>CLAIM/(claimed on ...)")
QW_HASH = prompt_hash(E_QUERY_SYS)
ECHO_HASH = prompt_hash(ECHO_SYS)
HASHES = {"read_sys": READ_HASH, "read_plain_block": PLAIN_TEMPLATE_HASH,
          "read_bridge_block": BRIDGE_TEMPLATE_HASH,
          "e_query": QW_HASH, "echo": ECHO_HASH,
          "serper_query_prompt": prompt_hash(eur.QUERY_SYS)}

# -------------------------------------------------------------------- per-claim cache (Fix 1)
# Query-writer output AND retrieval results are cached by (claim id, prompt hash, arm) — NOT by the
# generated (non-deterministic) query text — so a resume/rerun over the same population spends ZERO
# Exa requests and ZERO Serper credits for claims already done. --refresh forces re-querying.
CACHE = OUT_DIR / "cache"


def _ck(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]


def cache_get(kind: str, key: str, refresh: bool):
    if refresh:
        return None
    p = CACHE / f"{kind}_{key}.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:  # noqa: BLE001
            return None
    return None


def cache_put(kind: str, key: str, val) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    # unique temp name per writer: two threads may compute the SAME key concurrently, and a
    # shared .tmp made one rename clobber the other (FileNotFoundError). Atomic rename onto the
    # final path is still safe (last writer wins; the value is identical).
    tmp = CACHE / f".{kind}_{key}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(val, f)
    try:
        tmp.rename(CACHE / f"{kind}_{key}.json")
    except FileNotFoundError:
        pass


# -------------------------------------------------------------------- gated Flash JSON call
def flash_json(sys_prompt: str, user: str, cache_key_str: str | None = None,
               max_tokens: int = 300) -> tuple[dict, float]:
    """A json-mode Flash call routed through reader_lab.gated_post (same RateGate as the reads)."""
    body = {"model": eur.VERIFICATION_MODEL, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": user}]}
    if cache_key_str:
        body["prompt_cache_key"] = cache_key_str
    base, key = reader_lab._endpoint()
    r = reader_lab.gated_post(f"{base}/chat/completions",
                              {"Authorization": f"Bearer {key}"}, body, HTTP_TIMEOUT, attempts=3)
    j = r.json()
    obj = reader_lab._parse_json(j["choices"][0]["message"].get("content"))
    usage = j.get("usage") or {}
    return obj, (usage.get("estimated_cost") or 0.0)


# Read timeout. reader_lab.read_one hardcodes a 300s timeout x 6 retries, so ONE hung DeepInfra
# connection (no response, common under pool contention) stalls a whole batch for up to ~30 min
# (the 2026-09-15 hang: 32 idle ESTABLISHED sockets to DeepInfra, 0% CPU). prod_read mirrors
# read_one but with a SHORT timeout and few attempts, so a stuck read fails fast, is marked
# read-failed, and is recovered by the end-of-batch sweep or the next resumed launch.
HTTP_TIMEOUT = 45
READ_ATTEMPTS = 3


def prod_read(sys_prompt, claim_block, ids, sents):
    """read_one's parse/validation with a bounded timeout (reuses reader_lab.gated_post: RateGate
    + per-request retry)."""
    doc = "\n".join(f"[{i}] {s}" for i, s in zip(ids, sents))
    body = {"model": eur.VERIFICATION_MODEL, "temperature": 0, "max_tokens": 400,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": f"{claim_block}\n\nDOCUMENT:\n{doc}"}]}
    base, key = reader_lab._endpoint()
    r = reader_lab.gated_post(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"},
                              body, HTTP_TIMEOUT, attempts=READ_ATTEMPTS)
    j = r.json()
    usage = j.get("usage") or {}
    cost = usage.get("estimated_cost") or 0.0
    obj = reader_lab._parse_json(j["choices"][0]["message"].get("content"))
    d = obj.get("direction")
    if isinstance(d, int):
        d = str(d)
    ev = obj.get("evidence") or []
    idset = set(ids)
    if d not in eur.DIRECTIONS or not all(isinstance(i, int) and i in idset for i in ev):
        return {"direction": "I", "evidence": [], "reason": "", "qc_flag": "read-failed"}, cost
    ev = sorted(set(ev))
    flag = ""
    if d != "I" and not ev:
        d, flag = "X", "empty-directional"
    return {"direction": d, "evidence": ev, "reason": (obj.get("reason") or "")[:120],
            "qc_flag": obj.get("qc_flag") or flag}, cost


# -------------------------------------------------------------------- retrieval
def serper_fetch(cid: str, query: str, xd: list[str], serper: ApiCounter,
                 refresh: bool) -> tuple[list[dict], bool]:
    """Production Serper query, ceiling OFF. Cached PER CLAIM (Fix 1): a rerun over a done claim
    takes no paid credit. The prod query is deterministic, but the per-claim cache guarantees zero
    Serper on resume regardless of the shared disk cache's state."""
    xds = sorted(d for d in xd if d)
    ck = _ck("serper", cid, prompt_hash(query))
    hit = cache_get("serper", ck, refresh)
    if hit is not None:
        return hit, True
    # secondary: the shared query-text disk cache (free) before spending a credit
    dk = cache_key("serper", query, TOP_K, None, xds, 0)
    disk = disk_cache.get("serper", dk)
    if disk is not None and not refresh:
        cache_put("serper", ck, disk)
        return disk, True
    serper.take()
    res = search(query, TOP_K, date_ceiling=None, exclude_domains=xds,
                 min_results=0, provider="serper")
    cache_put("serper", ck, res)
    return res, False


def exa_fetch(cid: str, query: str, xd: list[str], exa: ApiCounter,
              refresh: bool) -> tuple[list[dict], bool]:
    """Exa neural search, no date bound. Cached PER CLAIM by (claim id, query-writer hash, arm) —
    NOT by the LLM-generated query text (Fix 1), so a rerun spends zero Exa. NEVER live-test; cap
    is hard."""
    xds = sorted(d for d in xd if d)
    ck = _ck("exa", cid, QW_HASH, "E")
    hit = cache_get("exa", ck, refresh)
    if hit is not None:
        for r in hit:
            r.setdefault("provider", "exa")
        return hit, True
    exa.take()      # HARD cap: SystemExit if the Exa cap is genuinely reached (no soft skip, no
    #                 skipping arm E when a cap is near) -- Daniel 2026-09-15: caps ARE the run size.
    res = search(query, TOP_K, date_ceiling=None, exclude_domains=xds,
                 min_results=0, provider="exa")
    cache_put("exa", ck, res)
    return res, False


def _post_ceiling(raw_date, ceil) -> bool:
    """True if the document is published after the claim's ceiling. Handles BOTH Serper's human
    format (eur._is_leak) AND Exa's ISO timestamp ('2021-09-17T00:00:00.000Z'), which _is_leak
    cannot parse (it silently returned False for every Exa doc). Recorded, not applied."""
    if not (raw_date and ceil):
        return False
    t = str(raw_date).strip()
    if len(t) >= 10 and t[4:5] == "-" and t[7:8] == "-":   # ISO date or timestamp
        return t[:10] > ceil[:10]
    return eur._is_leak(raw_date, ceil)


def prep_doc(h: dict, claim: str, query: str, ceiling: str | None) -> dict:
    """Fetch text (provider content or scrape, snippet fallback), select_regions, tag fc/leak.
    Returns a doc entry WITHOUT a read (reads are a later gated phase)."""
    url = h.get("url") or ""
    dom = _domain_of(url)
    fc = any(dom == f or dom.endswith("." + f) for f in FC_DOMS)
    entry = {"url": url, "domain": dom, "fc_domain": fc, "date": h.get("date"),
             "leak_flag": _post_ceiling(h.get("date"), ceiling),  # computed, recorded, NOT applied
             "provider": h.get("provider")}
    if fc:
        entry["read_status"] = "fc-dropped"          # leak control: fc_domain docs dropped
        return entry
    text = h.get("content") or scrape(url)
    prov = "content" if h.get("content") else "scrape"
    if not text or len(text) < SNIPPET_MIN_TEXT:
        text, prov = h.get("snippet") or "", "snippet"
    entry["provenance"] = prov
    regions, meta = eur.select_regions(text, claim, query)
    ids = [i for reg in regions for i in reg[0]]
    smap = {}
    for rids, sel in regions:
        for i, sn in zip(rids, sel):
            smap[i] = sn
    if not smap:
        entry["read_status"] = "empty-doc"
        entry["sent_ids"], entry["sents"] = [], []
        return entry
    entry["read_status"] = "prepped"
    entry["sent_ids"] = sorted(smap)
    entry["sents"] = [smap[i] for i in sorted(smap)]
    return entry


def prep_docs_parallel(hits, claim: str, query: str, ceiling: str | None) -> list[dict]:
    """Prep a claim's hits CONCURRENTLY (Fix 3). prep_doc -> scrape() is a 10s-timeout network call
    per Serper hit, so serial prep made a 10-hit claim take up to ~100s and capped throughput.
    Order is preserved by rank via ex.map. Exa hits carry their own content and skip scraping."""
    hits = list(hits or [])[:TOP_K]
    if not hits:
        return []
    with ThreadPoolExecutor(max_workers=min(TOP_K, len(hits)),
                            thread_name_prefix="prep") as ex:
        return list(ex.map(lambda h: prep_doc(h, claim, query, ceiling), hits))


# -------------------------------------------------------------------- resume / dedupe helpers
def record_complete(rec: dict) -> bool:
    """A results record counts as DONE (skip on resume) only if it has NO hard error and NO
    retrieval failure to retry (Fix 2). A record with a top-level "error" (unexpected crash), an
    arm_S serper SearchError, or an arm_E Exa SearchError is INCOMPLETE and must be rerun. A
    legitimate empty arm E -- the query writer produced no exa_query, or Exa returned 0 hits --
    is complete and is NOT rerun."""
    if rec.get("error"):
        return False
    if rec.get("arm_S", {}).get("error"):
        return False
    ee = rec.get("arm_E", {}).get("error", "")
    if ee and ee.startswith("exa:"):      # real Exa retrieval failure; "empty exa_query" is fine
        return False
    return True


def load_last_wins(path: Path) -> dict[str, dict]:
    """Read a results file keeping the LAST record per review_url (a rerun appends a fresh record;
    last-wins is the authoritative one)."""
    seen: dict[str, dict] = {}
    if path.exists():
        for l in path.open():
            try:
                r = json.loads(l)
                seen[r["review_url"]] = r
            except Exception:  # noqa: BLE001
                pass
    return seen


def dedupe_results(path: Path) -> int:
    """Rewrite the results file last-wins (one record per review_url) via tmp + atomic rename, so
    scoring sees exactly one record per claim. Returns the unique-claim count."""
    seen = load_last_wins(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in seen.values():
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)
    return len(seen)


# -------------------------------------------------------------------- claim loading
def load_claims(ids: list[str]) -> dict[str, dict]:
    idset = set(ids)
    out = {}
    for l in E1_DOCS.open():
        r = json.loads(l)
        if r["review_url"] in idset:
            out[r["review_url"]] = {
                "review_url": r["review_url"],
                "claim": r.get("claim_resolved") or r.get("claim_text"),
                "claim_text": r.get("claim_text"),
                "claim_date": r.get("claim_date_shown") or "",
                "publisher_site": r.get("publisher_site"),
                "veracity": r.get("veracity"),
                "ceiling": r.get("ceiling"),
                "prod_query": r.get("query"),
                "topic": r.get("topic"), "claim_type": r.get("claim_type")}
    return out


# -------------------------------------------------------------------- run (per-claim pipeline)
def run(ids: list[str], out_tag: str, budget_cap: float, serper_cap: int, exa_cap: int,
        workers: int, retr_workers: int, preflight: bool, refresh: bool = False,
        batch: int | None = None) -> None:
    """Per-claim pipeline (Daniel 2026-09-15: "just do a fresh run ... keep it simple").

    ONE pool of `retr_workers` claim tasks. Each task, for its single claim, does query-writing,
    Serper fetch, Exa fetch, scrape + prep, ALL reads (S-plain, E-plain, E-bridge) and the echo
    pass, then appends the finished record to results_<tag>.jsonl under a write lock. Retrieval of
    later claims therefore overlaps the reads of earlier claims (the batched runner serialised
    retrieval and reads, which capped throughput at ~12 claims/min). Reads and echo are submitted
    to a shared read pool (`workers` threads) and rate-limited by reader_lab's RateGate (target 160,
    ceiling 190); Exa is paced to 8/s globally, Serper is unpaced. No lock is ever held across a
    network call. Resume is per claim: any claim already in the results file is skipped, and its
    Serper/Exa/qw/reads/echo are cached, so a resume re-spends nothing. Caps (Serper/Exa/budget)
    are hard: if one is genuinely reached the run stops (SystemExit) and reports; nothing is
    silently skipped. `batch` is accepted for CLI compatibility and ignored."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    serper = ApiCounter(OUT_DIR / f"serper_credits_{out_tag}.json", serper_cap)
    exa = ApiCounter(OUT_DIR / f"exa_requests_{out_tag}.json", exa_cap, pace=EXA_PACE)
    reader_lab.GATE = reader_lab.RateGate(target=160, ceiling=190, floor=32)

    claims = load_claims(ids)
    order = [c for c in ids if c in claims]
    outp = OUT_DIR / f"results_{out_tag}.jsonl"
    done_ids = set()
    if outp.exists() and not refresh:
        # resume: a claim counts as done only if its record is complete (Fix 2). Errored records
        # (crash / serper or exa SearchError) stay in `todo` so they are rerun.
        for cid, r in load_last_wins(outp).items():
            if record_complete(r):
                done_ids.add(cid)
    todo = [c for c in order if c not in done_ids]
    print(f"loaded {len(order)}/{len(ids)} claims | {len(done_ids)} already done | "
          f"{len(todo)} to run | per-claim pipeline: retr_workers={retr_workers} "
          f"read_workers={workers} | gate target 160/ceiling 190 | exa pace {EXA_PACE}s", flush=True)

    spent = [0.0]
    charge_lock = threading.Lock()      # guards the running $ total; held only for the addition
    write_lock = threading.Lock()       # guards the append to the results file; held only for write
    prog_lock = threading.Lock()        # guards the progress counters
    prog = {"claims": 0, "reads": 0, "echo": 0, "errors": 0}
    t0 = time.time()
    abort = threading.Event()
    abort_reason = [None]

    def charge(c):
        with charge_lock:               # no network inside this lock
            spent[0] += c
            if spent[0] > budget_cap:
                raise SystemExit(f"BUDGET CAP ${budget_cap} breached at ${spent[0]:.3f}")

    def _read_key(blk, sids, sents):
        doc = "\n".join(f"[{i}] {s}" for i, s in zip(sids, sents))
        return _ck("read", READ_HASH, blk, doc)

    def do_read(job, force=False):
        """Runs in the shared read pool. Network call (prod_read) happens OUTSIDE every lock;
        charge()/prog only take their locks after the call returns."""
        cid, arm, rank, cond, blk, sids, sents = job
        rk = _read_key(blk, sids, sents)
        res = None if force else cache_get("read", rk, refresh)
        if res is not None and res.get("qc_flag") == "read-failed" and res.get("err"):
            res = None
        cost = 0.0
        if res is None:
            try:
                res, cost = prod_read(READ_SYS, blk, sids, sents)   # gated_post: 45s x3 attempts
            except Exception as e:  # noqa: BLE001
                res = {"direction": "I", "evidence": [], "reason": "",
                       "qc_flag": "read-failed", "err": repr(e)}
            cache_put("read", rk, res)
        charge(cost or 0.0)
        with prog_lock:
            prog["reads"] += 1
        return (cid, arm, rank, cond), res

    def do_echo(job):
        """Runs in the shared read pool. flash_json (gated_post 45s x3) is outside every lock."""
        cid, url, doc_txt, claim = job
        ek = _ck("echo", ECHO_HASH, claim, doc_txt)
        ce = cache_get("echo", ek, refresh)
        if ce is not None:
            obj, cost = ce, 0.0
        else:
            obj, cost = flash_json(ECHO_SYS, f"CLAIM: {claim}\n\nDOCUMENT:\n{doc_txt}", max_tokens=120)
            cache_put("echo", ek, obj)
        charge(cost or 0.0)
        with prog_lock:
            prog["echo"] += 1
        return (cid, url), {"echo": (obj.get("echo") or "").strip().lower(),
                            "reason": (obj.get("reason") or "")[:160]}

    read_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="read")

    def _write_rec(rec, fh):
        with write_lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    def process_claim(cid, fh):
        if abort.is_set():          # a cap/budget SystemExit fired: queued tasks drain instantly
            return
        c = claims[cid]
        rec = {"review_url": cid, "claim": c["claim"], "claim_text": c["claim_text"],
               "claim_date": c["claim_date"], "publisher_site": c["publisher_site"],
               "veracity": c["veracity"], "ceiling": c["ceiling"], "topic": c["topic"],
               "prod_query": c["prod_query"], "cost": 0.0,
               "arm_S": {"docs": []}, "arm_E": {"docs": []}}
        tc = time.time()
        try:
            xd = [c["publisher_site"]]
            # ---- arm S: production Serper query, ceiling OFF ----
            try:
                hits, cached = serper_fetch(cid, c["prod_query"], xd, serper, refresh)
                rec["arm_S"]["query"] = c["prod_query"]
                rec["arm_S"]["cached"] = cached
                rec["arm_S"]["docs"] = prep_docs_parallel(hits, c["claim"], c["prod_query"],
                                                          c["ceiling"])
            except SearchError as e:
                rec["arm_S"]["error"] = f"serper: {e}"
            # ---- arm E: two-hop query writer -> Exa neural ----
            try:
                qw = cache_get("qw", _ck("qw", cid, QW_HASH), refresh)
                if qw is None:
                    obj, cost = flash_json(E_QUERY_SYS, f"CLAIM: {c['claim']}",
                                           cache_key_str=None, max_tokens=400)
                    rec["cost"] += cost
                    charge(cost)
                    qw = {"target": (obj.get("target") or "").strip(),
                          "bearing": (obj.get("bearing") or "").strip(),
                          "exa_query": (obj.get("exa_query") or "").strip()[:400]}
                    cache_put("qw", _ck("qw", cid, QW_HASH), qw)
                rec["arm_E"].update(target=qw["target"], bearing=qw["bearing"],
                                    exa_query=qw["exa_query"], query_v=E_QUERY_V)
                exa_q = qw["exa_query"]
                if exa_q:
                    hits, cached = exa_fetch(cid, exa_q, xd, exa, refresh)
                    rec["arm_E"]["cached"] = cached
                    rec["arm_E"]["docs"] = prep_docs_parallel(hits, c["claim"], exa_q, c["ceiling"])
                else:
                    rec["arm_E"]["error"] = "empty exa_query"
            except SearchError as e:
                rec["arm_E"]["error"] = f"exa: {e}"
            # ---- build read jobs for this claim ----
            blk_plain = claim_block_plain(c["claim"], c["claim_date"])
            blk_bridge = None
            e = rec["arm_E"]
            if e.get("target") or e.get("bearing"):
                blk_bridge = claim_block_bridge(c["claim"], c["claim_date"],
                                                e.get("target", ""), e.get("bearing", ""))
            jobs = []
            for rank, d in enumerate(rec["arm_S"]["docs"]):
                if d.get("read_status") == "prepped":
                    jobs.append((cid, "S", rank, "plain", blk_plain, d["sent_ids"], d["sents"]))
            for rank, d in enumerate(rec["arm_E"]["docs"]):
                if d.get("read_status") == "prepped":
                    jobs.append((cid, "E", rank, "plain", blk_plain, d["sent_ids"], d["sents"]))
                    if blk_bridge:
                        jobs.append((cid, "E", rank, "bridge", blk_bridge, d["sent_ids"], d["sents"]))
            # ---- reads: submit to the shared gated pool, wait for this claim's reads ----
            read_out = {}
            for f in [read_pool.submit(do_read, j) for j in jobs]:
                key, res = f.result()
                read_out[key] = res
            # ---- sweep read-failed (up to 2 passes) ----
            job_by_key = {(j[0], j[1], j[2], j[3]): j for j in jobs}
            for _ in range(2):
                failed = [job_by_key[k] for k, r in read_out.items()
                          if r.get("qc_flag") == "read-failed" and k in job_by_key]
                if not failed:
                    break
                for f in [read_pool.submit(do_read, j, True) for j in failed]:
                    key, res = f.result()
                    read_out[key] = res
            for arm, key_r in (("S", "arm_S"), ("E", "arm_E")):
                for rank, d in enumerate(rec[key_r]["docs"]):
                    for cond in ("plain", "bridge"):
                        r = read_out.get((cid, arm, rank, cond))
                        if r is not None:
                            d[f"read_{cond}"] = {"direction": r.get("direction"),
                                                 "evidence": r.get("evidence"),
                                                 "reason": r.get("reason"),
                                                 "qc_flag": r.get("qc_flag")}
            # ---- echo pass (one per unique prepped doc URL) ----
            echo_jobs = []
            seen_echo = set()
            for key_r in ("arm_S", "arm_E"):
                for d in rec[key_r]["docs"]:
                    if d.get("read_status") == "prepped" and d["url"] not in seen_echo:
                        seen_echo.add(d["url"])
                        echo_jobs.append((cid, d["url"],
                                          "\n".join(f"- {s}" for s in d["sents"])[:6000], c["claim"]))
            echo_out = {}
            for f in [read_pool.submit(do_echo, j) for j in echo_jobs]:
                key, res = f.result()
                echo_out[key] = res
            for key_r in ("arm_S", "arm_E"):
                for d in rec[key_r]["docs"]:
                    eo = echo_out.get((cid, d["url"]))
                    if eo is not None:
                        d["echo"] = eo["echo"]
                        d["echo_reason"] = eo["reason"]
            # ---- append the finished record (write lock; no network inside) ----
            _write_rec(rec, fh)
            with prog_lock:
                prog["claims"] += 1
            if os.environ.get("PRODREGIME_TIME"):
                print(f"  [claim {cid[-40:]}] wall {time.time()-tc:.1f}s | "
                      f"S docs {len(rec['arm_S']['docs'])} E docs {len(rec['arm_E']['docs'])} | "
                      f"reads {len(jobs)} echo {len(echo_jobs)}", flush=True)
        except SystemExit as se:
            # a hard cap or the budget was reached (ApiCounter.take / charge). Stop the run: set
            # abort so every queued claim drains instantly; do NOT write a partial record.
            if not abort.is_set():
                abort.set()
                abort_reason[0] = str(se)
                print(f"  [ABORT] {se}", flush=True)
        except Exception as ex:  # noqa: BLE001
            # unexpected failure on this claim: record it with an "error" field (so resume reruns
            # it), log the traceback, and keep the run going.
            print(f"  [claim-error] {cid}: {ex!r}", flush=True)
            traceback.print_exc()
            rec["error"] = repr(ex)
            _write_rec(rec, fh)
            with prog_lock:
                prog["claims"] += 1
                prog["errors"] += 1

    # ---- optional preflight: warm up at 96 in-flight, check the 429 rate at 60s ----
    if preflight and todo:
        reader_lab.GATE.target = 96.0
        print("  [preflight] warming up at up to 96 in-flight; will check 429 rate at 60s", flush=True)

        def _pf_monitor():
            time.sleep(60)
            n, _n429, rate = reader_lab.GATE.window_stats()
            print(f"  [preflight] 429 rate {rate:.1%} over {n} reqs", flush=True)
            if rate > 0.20:
                print("  [preflight] ABORT: 429 rate over 20%", flush=True)
                abort.set()
            else:
                with reader_lab.GATE.cv:
                    reader_lab.GATE.target = 160.0
                    reader_lab.GATE.events.clear()
                    reader_lab.GATE.cv.notify_all()
                print("  [preflight] OK -> target raised to 160", flush=True)
        threading.Thread(target=_pf_monitor, daemon=True).start()

    # ---- per-minute progress reporter (claims/min + ETA) ----
    stop_prog = threading.Event()

    def _reporter():
        while not stop_prog.wait(60):
            with prog_lock:
                cd, rd, ed = prog["claims"], prog["reads"], prog["echo"]
            el = (time.time() - t0) / 60
            cpm = cd / el if el > 0 else 0.0
            remaining = len(todo) - cd
            eta = (remaining / cpm) if cpm > 0 else float("inf")
            print(f"  [progress] {cd}/{len(todo)} claims | {cpm:.1f} claims/min | "
                  f"reads {rd} echo {ed} | serper {serper.n} exa {exa.n} | ${spent[0]:.3f} | "
                  f"{reader_lab.GATE.summary()} | {el:.1f}m elapsed | "
                  f"ETA {eta:.1f}m", flush=True)
    threading.Thread(target=_reporter, daemon=True).start()

    # ---- drive: one pool of per-claim tasks ----
    with outp.open("a") as fh:
        with ThreadPoolExecutor(max_workers=retr_workers, thread_name_prefix="claim") as cex:
            list(cex.map(lambda cid: process_claim(cid, fh), todo))
        # ---- one automatic rerun pass over claims that errored THIS session (Fix 2) ----
        if not abort.is_set():
            todo_set = set(todo)
            retry = sorted(cid for cid, r in load_last_wins(outp).items()
                           if cid in todo_set and not record_complete(r))
            if retry:
                print(f"  [rerun] {len(retry)} errored claims, one pass", flush=True)
                with ThreadPoolExecutor(max_workers=retr_workers, thread_name_prefix="rerun") as cex:
                    list(cex.map(lambda cid: process_claim(cid, fh), retry))
    stop_prog.set()
    read_pool.shutdown(wait=True)

    # ---- dedupe last-wins so scoring sees one record per claim (Fix 2) ----
    unique = dedupe_results(outp)
    still_bad = [cid for cid, r in load_last_wins(outp).items() if not record_complete(r)]

    if abort.is_set():
        r = abort_reason[0] or "aborted"
        print(f"\nABORTED {out_tag}: {r} | {prog['claims']} claims written this process | "
              f"serper {serper.n} | exa {exa.n} | ${spent[0]:.3f} | {unique} unique records", flush=True)
        raise SystemExit(f"ABORTED: {r}")

    total = unique
    meta = {"tag": out_tag, "n_claims": total, "errored_records": len(still_bad),
            "serper_credits": serper.n, "exa_requests": exa.n,
            "api_cost_usd_this_process": round(spent[0], 4), "hashes": HASHES,
            "prompts": {"e_query": E_QUERY_V, "echo": ECHO_V, "read": READ_LABEL,
                        "serper_query": "query-v3 (reused from e1_ctx)"},
            "wall_min_this_process": round((time.time() - t0) / 60, 2)}
    (OUT_DIR / f"meta_{out_tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDONE {out_tag}: {total} claims total | {len(still_bad)} still errored | "
          f"this process ${spent[0]:.3f} | serper {serper.n} | exa {exa.n} | "
          f"{(time.time()-t0)/60:.1f}m -> {outp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--ids", type=Path, help="json list of review_urls to run")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--budget", type=float, default=1.0, help="hard USD cap on API cost")
    ap.add_argument("--serper-cap", type=int, default=40)
    ap.add_argument("--exa-cap", type=int, default=40)
    ap.add_argument("--workers", type=int, default=190)
    ap.add_argument("--retr-workers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--refresh", action="store_true",
                    help="force re-querying/re-reading; bypass the per-claim caches (Fix 1)")
    ap.add_argument("--read-prompt", default="v6.1", choices=sorted(_READ_PROMPTS),
                    help="reader system prompt from reader_lab_prompts.PROMPTS (default v6.1). The "
                         "read cache key includes its hash, so switching prompts reads fresh while "
                         "the serper/echo caches (keyed independently of the read prompt) are reused.")
    a = ap.parse_args()

    # Select the reader system prompt (Arm S read stage). Only READ_SYS/READ_HASH are load-bearing
    # for the read cache key; the recorded HASHES/label are updated to match.
    global READ_SYS, READ_HASH, READ_LABEL
    READ_SYS = _READ_PROMPTS[a.read_prompt]
    READ_HASH = prompt_hash(READ_SYS)
    READ_LABEL = f"read-{a.read_prompt}"
    HASHES["read_sys"] = READ_HASH
    HASHES["read_plain_block"] = prompt_hash(READ_SYS + "\n<<PLAIN>>CLAIM/(claimed on ...)")
    HASHES["read_bridge_block"] = prompt_hash(
        READ_SYS + "\n<<BRIDGE>>CLAIM/Verification target/Bearing/(claimed on ...)")

    if a.smoke:
        # smoke ids drawn deterministically from rep1500 + e1 veracities (10T/10F, seed)
        pop = list(pl.read_parquet(POP)["claim_id"])
        ver = {}
        idset = set(pop)
        for l in E1_DOCS.open():
            r = json.loads(l)
            if r["review_url"] in idset and r.get("query"):
                ver[r["review_url"]] = r["veracity"]
        T = sorted(u for u, v in ver.items() if v >= 4)
        F = sorted(u for u, v in ver.items() if v <= 2)
        rng = random.Random(SMOKE_SEED)
        ids = sorted(rng.sample(T, 10)) + sorted(rng.sample(F, 10))
        run(ids, a.tag or "smoke20", a.budget, a.serper_cap, a.exa_cap,
            a.workers, a.retr_workers, a.preflight, a.refresh, a.batch)
    elif a.full:
        ids = list(pl.read_parquet(POP)["claim_id"])
        run(ids, a.tag or "full", a.budget if a.budget != 1.0 else 20.0,
            a.serper_cap if a.serper_cap != 40 else 15000,
            a.exa_cap if a.exa_cap != 40 else 1500,
            a.workers, a.retr_workers, a.preflight, a.refresh, a.batch)
    elif a.ids:
        ids = json.loads(a.ids.read_text())
        run(ids, a.tag or "custom", a.budget, a.serper_cap, a.exa_cap,
            a.workers, a.retr_workers, a.preflight, a.refresh, a.batch)
    else:
        ap.error("one of --smoke / --full / --ids required")


if __name__ == "__main__":
    main()
