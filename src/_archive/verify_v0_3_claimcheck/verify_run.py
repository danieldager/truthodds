"""Run Tier 3 verification on the sampled AVeriTeC claims, save full diagnostics.

For each row in claims_n{N}.parquet, calls pipeline.verify() and writes the
ClaimVerdict + diagnostics to verdicts_n{N}.parquet. Resumable via claim_id
dedup. All errors are captured per-row; the run does not abort on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl

from eval.scripts._pool import pooled_checkpointed
from pipeline.embedding import embed
from pipeline.models import AtomicClaim
from pipeline.verify import get_cache_stats, reset_cache_stats, verify


def to_date_ceiling(date_str: str | None, fallback: str) -> str:
    """AVeriTeC dates are mostly MM/DD/YYYY, occasionally YYYY-MM-DD. Normalise to ISO YYYY-MM-DD.

    ISO is used because verify() now post-filters docs by string-comparing the
    summariser-extracted publication_date (also ISO) against this ceiling.
    """
    if not date_str:
        return fallback
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return fallback


def _error_row(claim_id: str, claim_text: str, t0: float, exc: Exception) -> dict:
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "provider": "", "date_ceiling": None,
        "resolving_provider": "", "providers_used": [],
        "veracity": 0, "evidence_sufficiency": 0,
        "evidence_agreement": 0, "source_reliability": 0,
        "justification": "", "analysis": "",
        "evidence_urls": [], "past_queries": [],
        "rounds_used": 0, "n_urls_seen": 0, "n_scraped": 0,
        "n_blocked_or_failed": 0, "n_snippet_used": 0, "n_irrelevant": 0,
        "n_search_errors": 0, "n_resampled": 0,
        "elapsed_seconds": round(time.time() - t0, 3),
        "llm_calls": 0,
        "cap_hit": False, "redundant_exit": False, "stopped_reason": "",
        "tier_resolved": 0,
        "verdict_4class": "", "verdict_4class_justification": "",
        "nudge_flag": None, "nudge_reason": "", "nudge_post_used": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_one(row: dict, fallback_date: str, trace_dir: Path, provider: str | None,
            cascade: list[str] | None, no_date_ceiling: bool, exclude_domains: list[str] | None,
            rounds_per_provider: list[int] | None = None, futility_stale: int = 0) -> dict:
    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")  # absolute clock → time-correlate to incidents
    try:
        claim = AtomicClaim(
            text=row["claim_text"],
            embedding=embed(row["claim_text"]).tolist(),
        )
        date_ceiling = None if no_date_ceiling else to_date_ceiling(row.get("claim_date"), fallback_date)
        fact_check_url = (row.get("fact_checking_article") or "").strip()
        exclude_urls = [fact_check_url] if fact_check_url else []
        trace: dict = {}
        rpp_kw = {"rounds_per_provider": rounds_per_provider} if rounds_per_provider else {}
        v = verify(
            claim,
            date_ceiling=date_ceiling,
            exclude_urls=exclude_urls,
            exclude_domains=exclude_domains,
            futility_stale=futility_stale,
            **rpp_kw,
            image_context=(row.get("image_serialization") or None),  # WS1 post-image serialization (verdict eval)
            post_context=(row.get("raw_context") or None),  # raw post → separate nudge call (not the verdict)
            verbose=False,
            provider=provider if not cascade else None,
            providers=cascade,
            trace=trace,
        )
        # Full-visibility trace: one JSON per claim (every query/source/summary/decision/timing).
        trace["claim_id"] = row["claim_id"]
        trace["gold_label"] = row.get("gold_label")
        trace["fact_checking_article"] = fact_check_url
        trace["started_at"] = started
        trace["finished_at"] = datetime.now().isoformat(timespec="seconds")
        (trace_dir / f"{row['claim_id']}.json").write_text(json.dumps(trace, indent=1, default=str))
        nudge = trace.get("nudge") or {}
        return {
            "claim_id": row["claim_id"],
            "claim_text": row["claim_text"],
            "provider": (cascade or [provider or ""])[0],
            "date_ceiling": date_ceiling,
            "resolving_provider": v.resolving_provider,
            "providers_used": list(v.providers_used),
            "veracity": v.scores.veracity,
            "evidence_sufficiency": v.scores.evidence_sufficiency,
            "evidence_agreement": v.scores.evidence_agreement,
            "source_reliability": v.scores.source_reliability,
            "justification": v.justification,
            "analysis": v.analysis,
            "evidence_urls": list(v.evidence_urls),
            "past_queries": list(v.past_queries),
            "rounds_used": v.rounds_used,
            "n_urls_seen": v.n_urls_seen,
            "n_scraped": v.n_scraped,
            "n_blocked_or_failed": v.n_blocked_or_failed,
            "n_snippet_used": v.n_snippet_used,
            "n_irrelevant": v.n_irrelevant,
            "n_search_errors": v.n_search_errors,
            "n_resampled": (trace.get("funnel") or {}).get("n_resampled", 0),  # FC-block starvation count
            "elapsed_seconds": float(v.elapsed_seconds),
            "llm_calls": v.llm_calls,
            "cap_hit": v.cap_hit,
            "redundant_exit": v.redundant_exit,
            "stopped_reason": v.stopped_reason,
            "tier_resolved": v.tier_resolved,
            "verdict_4class": v.verdict_4class,
            "verdict_4class_justification": v.verdict_4class_justification,
            "nudge_flag": nudge.get("flag"),
            "nudge_reason": nudge.get("reason", ""),
            "nudge_post_used": nudge.get("post_used", False),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        r = _error_row(row["claim_id"], row.get("claim_text", ""), t0, exc)
        r["provider"] = (cascade or [provider or ""])[0]
        return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="claims parquet from load_claims.py")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("-w", "--workers", type=int, default=2)
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="cap rows for smoke testing")
    ap.add_argument("--fallback-date",
                    default=datetime.today().strftime("%Y-%m-%d"))
    ap.add_argument("--provider", default=None,
                    help="single search provider (serper|tavily|searxng|exa); else config default")
    ap.add_argument("--cascade", default=None,
                    help="comma-separated provider cascade (e.g. searxng,tavily,exa); "
                         "overrides --provider. Use 'default' for config.SEARCH_CASCADE")
    ap.add_argument("--rounds-per-provider", default=None,
                    help="per-provider search-round budget, comma-separated aligned to the cascade "
                         "(e.g. '3,2' = serper 3 rounds then exa max 2); else config uniform default")
    from pipeline.config import FUTILITY_STALE_ROUNDS
    ap.add_argument("--futility-stale", type=int, default=FUTILITY_STALE_ROUNDS,
                    help="conclude early after this many consecutive rounds add NO new relevant "
                         "evidence (Exa still gets one shot first); 0 = disabled. Default from config.")
    ap.add_argument("--no-date-ceiling", action="store_true",
                    help="disable the eval date ceiling (match ClaimCheck's loose conditions)")
    ap.add_argument("--block-fc-domains", action="store_true",
                    help="exclude all config.FACT_CHECK_DOMAINS from retrieval (independent-verification "
                         "eval); Serper backfills a 2nd page if the block starves a query below SEARCH_KEEP_K")
    args = ap.parse_args()

    cascade = None
    if args.cascade:
        from pipeline.config import SEARCH_CASCADE
        cascade = SEARCH_CASCADE if args.cascade == "default" else [p.strip() for p in args.cascade.split(",")]

    rounds_per_provider = ([int(x) for x in args.rounds_per_provider.split(",")]
                           if args.rounds_per_provider else None)

    exclude_domains = None
    if args.block_fc_domains:
        from pipeline.config import FACT_CHECK_DOMAINS
        exclude_domains = sorted(FACT_CHECK_DOMAINS)

    out = args.output or args.input.parent / f"verdicts_{args.input.stem.replace('claims_', '')}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = out.parent / f"{out.stem}_trace"  # one JSON per claim — full visibility
    trace_dir.mkdir(parents=True, exist_ok=True)

    claims_df = pl.read_parquet(args.input)
    if args.limit:
        claims_df = claims_df.head(args.limit)

    existing_ids: set[str] = set()
    existing_rows: list[dict] = []
    if out.exists():
        existing = pl.read_parquet(out)
        existing_ids = set(existing["claim_id"].to_list())
        existing_rows = existing.to_dicts()
        print(f"existing: {len(existing_ids)} done")

    to_run = [r for r in claims_df.to_dicts() if r["claim_id"] not in existing_ids]
    print(f"output: {out}")
    print(f"trace:  {trace_dir}/<claim_id>.json")
    print(f"{'cascade: ' + '→'.join(cascade) if cascade else 'provider: ' + (args.provider or 'config default')}"
          f"  rounds/provider: {rounds_per_provider or 'config default'}"
          f"  date_ceiling: {'OFF' if args.no_date_ceiling else 'on'}"
          f"  fc-block: {'ON (%d domains)' % len(exclude_domains) if exclude_domains else 'off'}")
    print(f"workers: {args.workers}, to do: {len(to_run)}")
    if not to_run:
        print("nothing to do.")
        return

    embed("warmup")  # pre-load the embedding singleton before the worker pool (avoid load race)

    reset_cache_stats()
    t0 = time.time()
    new_rows: list[dict] = []

    def _work(r):
        return r["claim_id"], run_one(r, args.fallback_date, trace_dir, args.provider,
                                      cascade, args.no_date_ceiling, exclude_domains, rounds_per_provider,
                                      args.futility_stale)

    def _flush():
        pl.DataFrame(existing_rows + new_rows).write_parquet(out)

    # pooled_checkpointed adds the stall detection verify_run lacked: a hung claim makes it announce
    # "STALLED" and abandon after idle_timeout instead of going dark (clog 250626). Claims run ~60-90s,
    # so 180s with no completion = genuinely stuck. Also gives checkpointing + flushed throughput/ETA.
    stalled = pooled_checkpointed(to_run, _work, lambda _cid, rowdict: new_rows.append(rowdict),
                                  _flush, args.workers, "verify", checkpoint_every=10, idle_timeout=180)
    df = pl.DataFrame(existing_rows + new_rows)

    errs = sum(1 for r in new_rows if r.get("error"))
    print(f"\ndone in {time.time() - t0:.1f}s. wrote {df.height} total rows -> {out}")
    print(f"new rows: {len(new_rows)}, errors: {errs}")
    if new_rows:
        cap_hit = sum(1 for r in new_rows if r["cap_hit"])
        avg_rounds = sum(r["rounds_used"] for r in new_rows) / len(new_rows)
        avg_llm = sum(r["llm_calls"] for r in new_rows) / len(new_rows)
        avg_elapsed = sum(r["elapsed_seconds"] for r in new_rows) / len(new_rows)
        print(f"diagnostics: cap_hit={cap_hit}/{len(new_rows)}, "
              f"avg rounds={avg_rounds:.2f}, avg llm_calls={avg_llm:.1f}, "
              f"avg elapsed_s={avg_elapsed:.1f}")
        cs = get_cache_stats()
        print(f"prompt cache: {cs['cached_tokens']}/{cs['prompt_tokens']} prompt tokens cached "
              f"({cs['cached_pct']}%) over {cs['llm_calls']} LLM calls")

    if stalled:
        # a hung claim left a non-daemon worker thread that would keep the process alive; all rows
        # are flushed, so force-exit. Re-run to finish the abandoned claims.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
