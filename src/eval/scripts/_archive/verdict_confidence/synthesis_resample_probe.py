"""Control: is the verdict's variance in the SCORING step or the SYNTHESIS step?

We proved resampling the Likert off a fixed analysis is frozen — but that's the last step, a
transcription of a verdict the synthesis already wrote. We never resampled the SYNTHESIS. And the
multi-model run conflates 'different models' with 'different synthesis samples'. This isolates it:
one model (gpt-oss), one FIXED evidence pool, two resampling conditions —

  A. resample LIKERT only off the fixed analysis (K×, temp 1.0)        -> expect frozen (known)
  B. resample SYNTHESIS off the fixed pool (K×, temp 1.0), score each  -> the untested control

If B spreads where A doesn't, single-model self-consistency is ALIVE at the synthesis level (no
panel needed). Compared against the 4-model panel spread on the same claims.

Run (from src/): uv run python -m eval.scripts.verdict_confidence.synthesis_resample_probe
"""
from __future__ import annotations

import numpy as np
import polars as pl

from config import VERIFICATION_MODEL
from pipeline import verify_prompts as vp
from pipeline.config import SEARCH_CASCADE
from pipeline.embedding import embed
from pipeline.models import AtomicClaim, EvidenceDoc
from pipeline.verify import (_SYNTHESISE_MAX_TOKENS, _chat, _evaluate_likert,
                             _format_evidence, verify)

GPTOSS = VERIFICATION_MODEL
K = 8
# claims spanning panel spread: ev238/ev123 had ZERO panel spread; ev1229/ev293/ev700 had HIGH.
IDS = ["ev238", "ev123", "ev1229", "ev293", "ev700"]


def synth_once(claim_text: str, pool: list[EvidenceDoc], past_q: list[str], temp: float) -> str:
    raw = _chat(vp.build_synthesise_messages(claim_text, _format_evidence(pool), past_q),
                _SYNTHESISE_MAX_TOKENS, GPTOSS, temperature=temp)
    return vp.parse_synthesise(raw)["analysis"]


def main() -> None:
    mod = (pl.read_parquet("eval/scripts/verdict_confidence/data/eval_v1_modality.parquet")
           .with_row_index("ridx").with_columns(claim_id=pl.format("ev{}", "ridx")))
    agree = pl.read_parquet("eval/scripts/verdict_confidence/data/agreement.parquet")
    panel = {r["claim_id"]: r["veracity_by_model"] for r in agree.iter_rows(named=True)}
    rows = {r["claim_id"]: r for r in mod.filter(pl.col("claim_id").is_in(IDS)).iter_rows(named=True)}

    embed("warmup")
    print(f"model={GPTOSS}  K={K}  (A=resample Likert | B=resample synthesis, both temp 1.0)\n")
    for cid in IDS:
        row = rows[cid]
        claim = AtomicClaim(text=row["claim_text"], embedding=embed(row["claim_text"]).tolist())
        exclude = [row["review_url"]] if row["review_url"] else []
        tr: dict = {}
        v = verify(claim, date_ceiling=None, exclude_urls=exclude,
                   providers=list(SEARCH_CASCADE), trace=tr, model=GPTOSS)
        pool = [EvidenceDoc(**d) for rnd in tr.get("rounds", []) for d in rnd["docs"]]
        past_q = list(v.past_queries)

        # A: resample the Likert off the FIXED analysis
        ver_A = [_evaluate_likert(claim.text, v.analysis, GPTOSS, temperature=1.0)["veracity"]
                 for _ in range(K)]
        # B: resample the SYNTHESIS off the FIXED pool, score each fresh analysis
        ver_B = [_evaluate_likert(claim.text, synth_once(claim.text, pool, past_q, 1.0),
                                  GPTOSS, temperature=0.1)["veracity"] for _ in range(K)]

        print(f"{cid}  gold={row['binary_label']:4}  [{row['original_rating'][:18]:18}] "
              f"{row['claim_text'][:46]}")
        print(f"   panel(4 models): {panel.get(cid, '?')}  std={np.std(panel.get(cid, [0])):.2f}")
        print(f"   A likert-resample: {ver_A}  std={np.std(ver_A):.2f}")
        print(f"   B synth-resample:  {ver_B}  std={np.std(ver_B):.2f}\n")


if __name__ == "__main__":
    main()
