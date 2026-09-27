"""Evidence-profile run: ONE loop round per post, full visibility, three resolution arms.

Daniel's 2026-07-21 design: over dev-1000 (A+B, v4.7 chain, ALL checkable claims — topic
exclusions deliberately off), run CONTEXT + round 1 only (QUERY -> Serper 10 -> TRIAGE ->
gauntlet -> READ -> RESOLVE) and stop. Production prompts run UNTOUCHED; two extra
visibility-only classifier calls (source kind per result, evidence kind per cited entry)
and two extra resolution arms are computed on the side:

  arm A = the stock RESOLVE that ran in the round (sentence-text dossiers)
  arm B = pure code: the existing corroboration bar (_qualifying/_meets_bar) applied as
          the resolver — supported/refuted/conflicting/open from votes alone
  (a votes-only LLM arm was measured LOOSER than both on the 25-post smoke and dropped)

Labeled-claims mode (--claims-input): a parquet of gold-labeled claims (claim_text,
gold_label, claim_date, speaker, original_claim_url) is wrapped into pseudo-posts
(AVeriTeC-runner convention) for the same round-1 profile; gold travels in the record.

Open claims stay open (this is a snapshot, not a verdict run); budget coercions are
restored to open in the arms. Self-sourced closes carry into every arm.

  cd src && uv run python eval/scripts/claim_sourcing/evidence_profile_run.py \
      --inputs eval/data/survey_claims/dev500a_verify_input_v47.parquet \
               eval/data/survey_claims/dev500b_verify_input_v47.parquet \
      -o eval/data/survey_claims/evidence_profile_v1 [--sample N] [--seed 7] [--name smoke]
"""
import argparse, asyncio, random, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

from build_verify_input import load_verify_posts
from pipeline.harness import run_posts
from pipeline.pools import OrchestrationConfig, make_pools
import pipeline.verify_tweet_claims as v

PROFILE_CFG = v.PostVerifyConfig(max_rounds=1, exa_enabled=False, date_ceiling=False)  # date_ceiling set per-run in main()

# --- visibility-only classifiers (profiling artifacts, NEVER fed back into the loop) ----

SOURCE_CLASS_SYSTEM = """You classify web search results for a media-research profile. You receive a social-media POST by a news outlet, the search QUERY a fact-checking system issued for its claims, and up to 10 results (id, domain, date, snippet).
For each result, judge from the domain and snippet:
- "kind": one of "news-report" (a news organization's reporting), "primary-official" (government, court, agency, company/institution's own record or statement), "fact-check" (a dedicated fact-checking page), "aggregator" (portal/wire-reprint/roundup), "opinion" (op-ed, editorial, blog commentary), "press-release", "video" (video page or channel), "social" (social-media page), "reference" (encyclopedia, database, archive), "other".
- "on_claim": true if the snippet indicates the page addresses the events or statements the post's claims are about; false if it is merely topically adjacent or unrelated.
Output strictly valid JSON, nothing else:
{"results": [{"i": <result id>, "kind": "<...>", "on_claim": <bool>}]}
Cover every result id given."""

EVIDENCE_CLASS_SYSTEM = """You classify evidence excerpts gathered for fact-checking a social-media post's claims. You receive the CLAIMS (numbered) and evidence ENTRIES (id, source domain, the claim each entry was matched to, and the excerpt with the reader's cited sentences marked like <<this>>).
For each entry, judge:
- "kind": one of "official-record" (law, ruling, filing, agency data quoted or described), "actor-statement" (a statement by a person or organization the claim is about), "journalist-reporting" (the article's own reported facts), "expert-quote" (analyst/academic/professional commentary), "statistic" (figures or data points), "opinion" (the writer's judgment or advocacy), "context" (background that neither affirms nor contradicts).
- "specificity": "exact" (speaks to the claim's exact proposition), "partial" (part of the proposition or an adjacent fact), "tangential".
Output strictly valid JSON, nothing else:
{"entries": [{"id": "<entry id>", "kind": "<...>", "specificity": "<...>"}]}
Cover every entry id given."""

def _restore_open(ledger: dict, coerced: list) -> dict:
    out = {str(k): s for k, s in ledger.items()}
    for cid in coerced:
        out[str(cid)] = "open"
    return out


def _arm_b2(claims: list[dict], evidence: list[dict], self_closed: dict) -> dict:
    """The corroboration bar AS the resolver — pure code, no LLM."""
    out = {}
    for i in range(1, len(claims) + 1):
        if str(i) in self_closed:
            out[str(i)] = self_closed[str(i)]
            continue
        okS, _, _ = v._meets_bar(v._qualifying(evidence, i, "supported", claims), "supports")
        okR, _, _ = v._meets_bar(v._qualifying(evidence, i, "refuted", claims), "refutes")
        out[str(i)] = "conflicting" if (okS and okR) else \
                      "supported" if okS else "refuted" if okR else "open"
    return out


async def profile_post(post: dict, pools) -> dict:
    rec = await v.verify_post(post, pools, PROFILE_CFG)
    claims = rec["claims"]
    evidence = rec["evidence"]
    n = len(claims)
    # self-sourced closes carry into every arm (they bypass evidence resolution)
    self_closed = {str(i): "supported" for i in range(1, n + 1)
                   if "self-sourced" in (rec["resolutions"].get(str(i)) or "")}
    arms = {"A": _restore_open(rec["ledger"], rec.get("coerced_open") or []),
            "B": _arm_b2(claims, evidence, self_closed)}
    # visibility-only classifiers
    src_classes, ev_classes = [], []
    rounds = rec.get("rounds") or []
    if rounds and rounds[0].get("results"):
        res = rounds[0]["results"]
        res_lines = "\n".join(f"[{r['i']}] {r['domain']} | {r.get('date') or 'date unknown'} | "
                              f"{(r.get('snippet') or '(no snippet)')[:240]}" for r in res)
        obj = await pools.llm.chat_json(
            [{"role": "system", "content": SOURCE_CLASS_SYSTEM},
             {"role": "user", "content": f"POST:\n{rec['text']}\n\nQUERY: {rounds[0]['query']}"
                                         f"\n\nRESULTS:\n{res_lines}\n\nClassify every result."}],
            max_tokens=700, label="source-class")
        got = {r.get("i"): r for r in (obj.get("results") or []) if isinstance(r, dict)}
        src_classes = [{"i": r["i"], "kind": (got.get(r["i"]) or {}).get("kind"),
                        "on_claim": (got.get(r["i"]) or {}).get("on_claim")} for r in res]
    full_reads = [e for e in evidence if not e.get("snippet_only")]
    if full_reads:
        claim_lines = "\n".join(f"{i}. {c['c']}" for i, c in enumerate(claims, 1))
        ent_lines = "\n\n".join(
            f"[{e['src']}:{e['claim_id']}] {e['domain']} | for claim {e['claim_id']}:\n"
            f"{(e.get('text') or '')[:900]}" for e in full_reads)
        obj = await pools.llm.chat_json(
            [{"role": "system", "content": EVIDENCE_CLASS_SYSTEM},
             {"role": "user", "content": f"CLAIMS:\n{claim_lines}\n\nENTRIES:\n{ent_lines}"
                                         f"\n\nClassify every entry."}],
            max_tokens=900, label="evidence-class")
        got = {str(r.get("id")): r for r in (obj.get("entries") or []) if isinstance(r, dict)}
        ev_classes = [{"id": f"{e['src']}:{e['claim_id']}",
                       "kind": (got.get(f"{e['src']}:{e['claim_id']}") or {}).get("kind"),
                       "specificity": (got.get(f"{e['src']}:{e['claim_id']}") or {}).get("specificity")}
                      for e in full_reads]
    rec["profile"] = {"arms": arms, "source_classes": src_classes,
                      "evidence_classes": ev_classes,
                      "self_closed": sorted(self_closed)}
    if post.get("gold"):
        rec["gold"] = post["gold"]
    if post.get("dataset"):
        rec["dataset"] = post["dataset"]
    return rec


def load_labeled_claims(path: str) -> list[dict]:
    """Gold-labeled claims parquet -> pseudo-posts (AVeriTeC-runner convention). One claim
    per post; the claim text IS the post; origin = the claim's appearance domain."""
    from urllib.parse import urlparse
    import pandas as pd
    posts = []
    for r in pd.read_parquet(path).itertuples():
        url = r.original_claim_url if isinstance(r.original_claim_url, str) else ""
        dom = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
        date = str(getattr(r, "claim_date", "") or "")[:10] or None
        spk = getattr(r, "speaker", None)
        posts.append({"id": str(r.claim_id), "post_id": str(r.claim_id), "url": url or None,
                      "handle": (spk if isinstance(spk, str) and spk else "claimant")[:40],
                      "domain": dom, "date": date, "text": r.claim_text,
                      "gold": r.gold_label,
                      "claims": [{"c": r.claim_text, "t": "assertion", "cw": True,
                                  "claim_id": str(r.claim_id)}]})
    return posts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=[], help="verify-input parquets (A and B)")
    ap.add_argument("--claims-input", nargs="*", default=[],
                    help="gold-labeled claims parquets (claim_id, claim_text, gold_label, "
                         "claim_date, speaker, original_claim_url)")
    ap.add_argument("--ceiling", action="store_true",
                    help="date ceiling ON (labeled-claims calibration convention)")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--sample", type=int, default=0, help="random post sample across inputs")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--posts", default="", help="comma-separated post_id filter")
    ap.add_argument("--k", type=int, default=0, help="posts in flight (OrchestrationConfig k_posts)")
    ap.add_argument("--name", default="profile")
    args = ap.parse_args()

    PROFILE_CFG.date_ceiling = bool(args.ceiling)

    posts = []
    for path in args.inputs:
        batch = load_verify_posts(path)
        for p in batch:
            p["dataset"] = Path(path).name
            # profile scope (Daniel 2026-07-21): ALL checkable claims — topic exclusions OFF
            for c in p["claims"]:
                if c.get("cat") is not None:
                    c["cw"] = c["cat"] == "checkable"
        posts.extend(batch)
    for path in args.claims_input:
        batch = load_labeled_claims(path)
        for p in batch:
            p["dataset"] = Path(path).name
        posts.extend(batch)
    posts = [p for p in posts if any(c.get("cw") for c in p["claims"])]
    if args.posts:
        keep = {s.strip() for s in args.posts.split(",")}
        posts = [p for p in posts if str(p["post_id"]) in keep]
    if args.sample:
        # deterministic NESTED sampling (2026-07-21): ordering by hash(seed, post_id) means
        # a larger --sample is a superset of a smaller one, so scaling a profile up reuses
        # every record already on disk instead of drawing a fresh set
        import hashlib
        posts.sort(key=lambda p: hashlib.sha1(f"{args.seed}:{p['post_id']}".encode()).hexdigest())
        posts = posts[:args.sample]
    n_claims = sum(sum(c.get("cw", False) for c in p["claims"]) for p in posts)
    print(f"profiling {len(posts)} posts / {n_claims} checkable claims "
          f"(1 Serper query per post, round 1 only)", flush=True)

    cfg = OrchestrationConfig(**({"k_posts": args.k} if args.k else {}))
    async def run():
        pools = make_pools(cfg)
        try:
            return await run_posts(posts, profile_post, pools, args.out_dir, name=args.name)
        finally:
            await pools.close()
    print(asyncio.run(run()))


if __name__ == "__main__":
    main()
