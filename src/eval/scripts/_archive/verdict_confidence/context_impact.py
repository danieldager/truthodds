"""Does feeding the RECONSTRUCTED contextualized claim to the existing verify() fix the
soft_flag misses? Re-verify a few cases on (a) the bare claim_text vs (b) the contextualized
claim, compare veracity → binary vs gold. Zero pipeline change — only the input claim differs.

Run (from src/): uv run python -m eval.scripts.verdict_confidence.context_impact
"""
from __future__ import annotations

import polars as pl

from eval.scripts.verdict_confidence.reconstruct_context import reconstruct
from pipeline.config import SEARCH_CASCADE
from pipeline.embedding import embed
from pipeline.models import AtomicClaim
from pipeline.verify import verify

SPLITS = "eval/scripts/verdict_confidence/data/eval_v1_splits.parquet"
IDS = ["ev1229", "ev1630", "ev1689", "ev2215"]  # 2 recency wins + 2 hard (causation, comparison)


def _verify(text: str, exclude: str) -> int:
    claim = AtomicClaim(text=text, embedding=embed(text).tolist())
    v = verify(claim, date_ceiling=None, exclude_urls=[exclude] if exclude else [],
               providers=list(SEARCH_CASCADE))
    return v.scores.veracity


def main() -> None:
    df = pl.read_parquet(SPLITS)
    rows = {r["claim_id"]: r for r in df.filter(pl.col("claim_id").is_in(IDS)).iter_rows(named=True)}
    embed("warmup")
    print(f"{'id':7} {'gold':5} {'v_bare':>6} {'v_ctx':>6} {'bare':>5} {'ctx':>5}  contextualized claim")
    for cid in IDS:
        r = rows[cid]
        ctx = reconstruct(r["claim_text"], r["claimant"], r["claim_date"], r["review_title"] or "")
        excl = r["review_url"] or ""
        v_bare = _verify(r["claim_text"], excl)
        v_ctx = _verify(ctx["contextualized_claim"], excl)
        pred_bare = "pass" if v_bare >= 4 else "flag"
        pred_ctx = "pass" if v_ctx >= 4 else "flag"
        mark = lambda p: "✓" if p == r["binary_label"] else "✗"
        print(f"{cid:7} {r['binary_label']:5} {v_bare:>6} {v_ctx:>6} {pred_bare:>4}{mark(pred_bare)} "
              f"{pred_ctx:>4}{mark(pred_ctx)}  {ctx['contextualized_claim'][:60]}")


if __name__ == "__main__":
    main()
