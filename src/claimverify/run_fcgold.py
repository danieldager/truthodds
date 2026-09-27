"""Run the claim-level loop on the pinned fc-gold population (n=3,274), the urn's fit corpus.

  cd src
  uv run python -m claimverify.run_fcgold --dry-run
  uv run python -m claimverify.run_fcgold --smoke 50 --k 50 --exa
  uv run python -m claimverify.run_fcgold --exa --budget 12

Population = eval/data/urn_runs/e1_ctx/model_ladder/oof_scores.parquet (written by
claimverify.fcgold_urn_oof), so the loop runs on exactly the claims the urn is scored on.
Inputs mirror the urn run (evidence_urn_run.py): text = claim_resolved, ceiling = the urn's
ceiling (claim date, clamped below the fact-check date), origin = publisher_site (the
fact-checker's own site), context = x_context only where the urn used it. The urn's own
query is carried in extras and never shown to the model.
Arm: top3 (pages_per_round 3). Same switches as run_averitec.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from claimverify.config import SRC, ClaimVerifyConfig, OrchestrationConfig
from claimverify.harness import run_claims, write_manifest
from claimverify.origin import origin_domain
from claimverify.pools import make_pools
from claimverify.retrieval import tbs_date_ceiling

RESULTS = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
OOF = SRC / "eval/data/urn_runs/e1_ctx/model_ladder/oof_scores.parquet"
CONTEXT = SRC / "eval/data/claim_context_cal.parquet"
OUT_ROOT = SRC / "eval/data/claimverify_runs/fc_gold"
SEED = 707
PAGES_PER_ROUND = 3


def claim_id_of(review_url: str) -> str:
    return hashlib.blake2b(review_url.encode(), digest_size=8).hexdigest()


def build_claims() -> list[dict]:
    oof = pd.read_parquet(OOF)
    keep = set(oof["review_url"])
    ctx = pd.read_parquet(CONTEXT).set_index("review_url")["x_context"].to_dict()
    claims = []
    for line in RESULTS.open():
        r = json.loads(line)
        u = r["review_url"]
        if u not in keep:
            continue
        v = int(r["veracity"])
        context = (ctx.get(u) or "").strip() if r.get("context_used") else ""
        claims.append({
            "claim_id": claim_id_of(u), "text": r["claim_resolved"] or r["claim_text"],
            "date": r["ceiling"], "origin_url": f"https://{r['publisher_site']}",
            "context": context or None,
            "gold_label": "TRUE" if v >= 4 else "FALSE", "type": "assertion",
            "extras": {"review_url": u, "veracity": v, "rating_subtype": r.get("rating_subtype"),
                       "claim_type": r.get("claim_type"), "ceiling_src": r.get("ceiling_src"),
                       "claim_date_shown": r.get("claim_date_shown"),
                       "resolution_status": r.get("resolution_status"),
                       "urn_query": r.get("query")}})
    assert len(claims) == len(keep), (len(claims), len(keep))
    return claims


def dry_run(claims: list[dict], cfg: ClaimVerifyConfig) -> None:
    print(f"{len(claims)} claims | pages_per_round={cfg.pages_per_round} exa={cfg.exa_enabled} "
          f"ceiling={cfg.date_ceiling} origin_exclusion={cfg.origin_exclusion} "
          f"code_bar={cfg.code_bar} ugc_blocklist={cfg.ugc_blocklist} "
          f"fc_undated_drop={cfg.fc_undated_drop} model={cfg.model}")
    for c in claims[:5]:
        o = origin_domain(c["origin_url"]) if cfg.origin_exclusion else None
        print(f"  {c['claim_id']} | {c['text'][:80]!r}\n"
              f"      date {c['date']} | tbs {tbs_date_ceiling(c['date']) if cfg.date_ceiling else '(off)'}"
              f" | exclude {[o] if o else []} | gold {c['gold_label']} | ctx {bool(c['context'])}")
    print("labels: " + json.dumps(Counter(c["gold_label"] for c in claims)))
    print("context present: " + str(sum(bool(c["context"]) for c in claims)))
    print("ceiling_src: " + json.dumps(Counter(c["extras"]["ceiling_src"] for c in claims)))
    print("exclusions: " + json.dumps(Counter(origin_domain(c["origin_url"]) for c in claims).most_common(8)))
    print(f"dates {min(c['date'] for c in claims)} .. {max(c['date'] for c in claims)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", type=int, default=0, help="first N after a seeded shuffle")
    ap.add_argument("--k", type=int, default=0, help="claims in flight (0 = config default)")
    ap.add_argument("--exa", action="store_true", help="enable the Exa escalation round")
    ap.add_argument("--no-ceiling", action="store_true")
    ap.add_argument("--no-origin", action="store_true")
    ap.add_argument("--no-bar", action="store_true")
    ap.add_argument("--no-blocklist", action="store_true")
    ap.add_argument("--read-undated-fc", action="store_true",
                    help="read undated fact-check pages (default drops them, as the urn run did)")
    ap.add_argument("--budget", type=float, default=0.0, help="USD abort cap (0 = none)")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = ClaimVerifyConfig(pages_per_round=PAGES_PER_ROUND, triage_enabled=False,
                            exa_enabled=a.exa, date_ceiling=not a.no_ceiling,
                            origin_exclusion=not a.no_origin, code_bar=not a.no_bar,
                            ugc_blocklist=not a.no_blocklist,
                            fc_undated_drop=not a.read_undated_fc)
    ocfg = OrchestrationConfig(**({"k_claims": a.k} if a.k else {}))
    claims = build_claims()
    if a.dry_run:
        dry_run(claims, cfg)
        return
    out = Path(a.out) if a.out else OUT_ROOT / ("top3_smoke" if a.smoke else "top3")
    claims = sorted(claims, key=lambda c: c["claim_id"])
    random.Random(SEED).shuffle(claims)
    if a.smoke:
        claims = claims[:a.smoke]
        out.mkdir(parents=True, exist_ok=True)
        (out / "sample.json").write_text(json.dumps(
            {"seed": SEED, "n": len(claims), "claim_ids": [c["claim_id"] for c in claims]}, indent=1))
    write_manifest(out, cfg=cfg, ocfg=ocfg, dataset_path=OOF, n=len(claims), arm="top3",
                   seed=SEED, smoke=a.smoke or None,
                   extra={"results_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
                          "budget_usd": a.budget})

    async def _run():
        pools = make_pools(ocfg, llm_cache=cfg.llm_cache, blocklist=cfg.ugc_blocklist)
        try:
            return await run_claims(claims, pools, cfg, out, name="fcgold-top3", budget_usd=a.budget)
        finally:
            await pools.close()

    print(json.dumps(asyncio.run(_run())))


if __name__ == "__main__":
    main()
