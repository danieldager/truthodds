"""Run the claim-level loop on AVeriTeC dev (the 491 unique claims of the 500-row split
ClaimCheck reports 76.4% on), one arm at a time.

  cd src
  uv run python -m claimverify.run_averitec --arm top3 --dry-run
  uv run python -m claimverify.run_averitec --arm top3 --smoke 10 --k 10
  uv run python -m claimverify.run_averitec --arm top3 --budget 5
  uv run python -m claimverify.run_averitec --arm all10 --budget 8

Arms: top3 = pages_per_round 3, all10 = pages_per_round 10 (triage off in both; the
Serper call, scrapes and identical READ calls are shared through the disk caches).
Protocol: claim-date ceiling ON (tbs cd_max, M/D/YYYY, inclusive), origin site excluded
(web.archive.org unwrapped), no speaker in the prompt, Exa OFF unless --exa.
--no-ceiling / --no-origin / --no-bar / --no-blocklist turn those four off; all four off
is the ClaimCheck-protocol arm. Every switch is recorded in the run's manifest.
Input: claims_dev_500_gold.parquet (ids) deduped to 491, joined with averitec_full dev
for claim_types / reporting_source / fc_url (carried, never shown to the model).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
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

GOLD = SRC / "eval/scripts/verification_grading/data/claims_dev_500_gold.parquet"
FULL = SRC / "eval/data/averitec/averitec_full.parquet"
OVERRIDES = SRC / "eval/data/urn_runs/averitec_dev/id_overrides.json"
OUT_ROOT = SRC / "eval/data/claimverify_runs/averitec_dev"
SEED = 707
ARMS = {"top3": 3, "all10": 10}


def _iso(d: str) -> str:
    """AVeriTeC dates are D-M-YYYY -> ISO YYYY-MM-DD."""
    day, month, year = str(d).split("-")
    return f"{year}-{int(month):02d}-{int(day):02d}"


def build_claims() -> list[dict]:
    gold = pd.read_parquet(GOLD).drop_duplicates("claim_id")
    full = pd.read_parquet(FULL)
    full = full[full["split"] == "dev"].drop_duplicates("claim")
    norm = {" ".join(c.split()): r for c, r in zip(full["claim"], full.to_dict("records"))}
    claims = []
    for r in gold.itertuples():
        extra = norm.get(" ".join(r.claim_text.split()), {})
        url = r.original_claim_url if isinstance(r.original_claim_url, str) else None
        claims.append({
            "claim_id": r.claim_id, "text": r.claim_text, "date": _iso(r.claim_date),
            "origin_url": url, "gold_label": r.gold_label, "type": "assertion",
            "extras": {"speaker": r.speaker if isinstance(r.speaker, str) else None,
                       "claim_types": extra.get("claim_types"),
                       "reporting_source": extra.get("reporting_source"),
                       "fc_url": extra.get("fc_url")}})
    return claims


def dry_run(claims: list[dict], cfg: ClaimVerifyConfig) -> None:
    print(f"{len(claims)} unique claims | arm pages_per_round={cfg.pages_per_round} "
          f"triage={cfg.triage_enabled} exa={cfg.exa_enabled} ceiling={cfg.date_ceiling} "
          f"origin_exclusion={cfg.origin_exclusion} code_bar={cfg.code_bar} "
          f"ugc_blocklist={cfg.ugc_blocklist} model={cfg.model}")
    for c in claims[:5]:
        o = origin_domain(c["origin_url"]) if cfg.origin_exclusion else None
        print(f"  {c['claim_id']} | {c['text'][:80]!r}\n"
              f"      date {c['date']} | tbs {tbs_date_ceiling(c['date']) if cfg.date_ceiling else '(off)'}"
              f" | exclude {[o] if o else []} | gold {c['gold_label']}")
    print("labels: " + json.dumps(Counter(c["gold_label"] for c in claims)))
    print("exclusions: " + json.dumps(Counter(origin_domain(c["origin_url"]) for c in claims).most_common(6)))
    print(f"dates {min(c['date'] for c in claims)} .. {max(c['date'] for c in claims)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--smoke", type=int, default=0, help="first N after a seeded shuffle")
    ap.add_argument("--k", type=int, default=0, help="claims in flight (0 = config default)")
    ap.add_argument("--exa", action="store_true", help="enable the Exa escalation round")
    ap.add_argument("--no-ceiling", action="store_true")
    ap.add_argument("--ceiling", default=None, help="fixed ISO ceiling for every claim (implies the ceiling on)")
    ap.add_argument("--no-origin", action="store_true", help="do not exclude the origin site")
    ap.add_argument("--no-bar", action="store_true", help="accept the RESOLVE close as final")
    ap.add_argument("--no-blocklist", action="store_true", help="drop the UGC blocklist")
    ap.add_argument("--snippets", choices=["fallback", "all"], default="fallback")
    ap.add_argument("--budget", type=float, default=0.0, help="USD abort cap (0 = none)")
    ap.add_argument("--no-llm-cache", action="store_true", help="repeat run: every LLM call live")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = ClaimVerifyConfig(pages_per_round=ARMS[a.arm], triage_enabled=False,
                            exa_enabled=a.exa, date_ceiling=bool(a.ceiling) or not a.no_ceiling,
                            ceiling_date=a.ceiling,
                            origin_exclusion=not a.no_origin, code_bar=not a.no_bar,
                            ugc_blocklist=not a.no_blocklist,
                            snippet_evidence=a.snippets, llm_cache=not a.no_llm_cache)
    ocfg = OrchestrationConfig(**({"k_claims": a.k} if a.k else {}))
    claims = build_claims()
    if a.dry_run:
        dry_run(claims, cfg)
        return
    out = Path(a.out) if a.out else OUT_ROOT / (a.arm + ("_smoke" if a.smoke else ""))
    claims = sorted(claims, key=lambda c: c["claim_id"])
    random.Random(SEED).shuffle(claims)
    if a.smoke:
        claims = claims[:a.smoke]
        out.mkdir(parents=True, exist_ok=True)
        (out / "sample.json").write_text(json.dumps(
            {"seed": SEED, "n": len(claims), "claim_ids": [c["claim_id"] for c in claims]}, indent=1))
    write_manifest(out, cfg=cfg, ocfg=ocfg, dataset_path=GOLD, n=len(claims), arm=a.arm,
                   seed=SEED, smoke=a.smoke or None,
                   extra={"full_parquet_sha256": __import__("hashlib").sha256(FULL.read_bytes()).hexdigest(),
                          "budget_usd": a.budget})

    async def _run():
        pools = make_pools(ocfg, llm_cache=cfg.llm_cache, blocklist=cfg.ugc_blocklist)
        try:
            return await run_claims(claims, pools, cfg, out, name=f"averitec-{a.arm}",
                                    budget_usd=a.budget)
        finally:
            await pools.close()

    print(json.dumps(asyncio.run(_run())))


if __name__ == "__main__":
    main()
