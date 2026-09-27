"""Smoke test — K-sample self-consistency resampling of the verdict+Likert step.

For a tiny N of eval_v1 claims: run verify() ONCE to get the fixed synthesis `analysis`,
then call the Likert evaluator K times at temperature>0 on that SAME analysis (no
re-retrieval). Confirms, before any full 2370×K run:
  (a) the resampling path works end-to-end on the renamed veracity/3-dim fields;
  (b) temp>0 actually yields DIVERSE samples — the whole confidence layer depends on this
      (at the _chat default temp ~0.1 the samples may be identical → zero signal);
  (c) per-claim cost/latency, to project the full run.

Paid Groq calls — keep N tiny (default 2). Per claim: one full verify() (cascade =
serper→exa) + K Likert calls; plus ONE global token-usage probe for the $ projection.

Run (from src/):
  uv run python -m eval.scripts.verdict_confidence.resample_smoke -n 2 -k 10 --temp 1.0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from config import VERIFICATION_MODEL
from eval.scripts.verdict_confidence.modality import tag_modality
from pipeline import verify_prompts as vp
from pipeline.config import SEARCH_CASCADE
from pipeline.embedding import embed
from pipeline.models import AtomicClaim
from pipeline.verify import _EVALUATE_MAX_TOKENS, _client, _evaluate_likert, verify

DIMS = ("veracity", "evidence_sufficiency", "evidence_agreement", "source_reliability")
OUT = Path("eval/scripts/verdict_confidence/data/smoke_resample.json")


def _resample_likert(claim_text: str, analysis: str, model: str, k: int, temp: float) -> tuple[list[dict], float]:
    """K Likert samples off the fixed analysis at temperature `temp`. Returns (samples, total_s)."""
    samples: list[dict] = []
    t0 = time.time()
    for _ in range(k):
        samples.append(_evaluate_likert(claim_text, analysis, model, temperature=temp))
    return samples, round(time.time() - t0, 2)


def _diversity(samples: list[dict]) -> dict:
    """Per-claim spread of the K samples + the binary derivation (pass iff veracity >= 4)."""
    arr = {d: np.array([s[d] for s in samples], dtype=float) for d in DIMS}
    vectors = {tuple(s[d] for d in DIMS) for s in samples}
    justifs = {s["justification"] for s in samples}
    passes = arr["veracity"] >= 4
    pass_frac = float(passes.mean())
    return {
        "veracity_unique": sorted({int(v) for v in arr["veracity"]}),
        "std": {d: round(float(arr[d].std()), 3) for d in DIMS},
        "mean": {d: round(float(arr[d].mean()), 2) for d in DIMS},
        "n_distinct_vectors": len(vectors),
        "n_distinct_justifications": len(justifs),
        "binary_pass_fraction": round(pass_frac, 3),
        "majority_verdict": "pass" if pass_frac >= 0.5 else "flag",
        "verdict_agreement": round(max(pass_frac, 1 - pass_frac), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--limit", type=int, default=2, help="# claims (keep tiny — paid)")
    ap.add_argument("-k", "--samples", type=int, default=10, help="K resamples per claim")
    ap.add_argument("--temp", type=float, default=1.0, help="resampling temperature (>0)")
    ap.add_argument("--offset", type=int, default=0, help="skip the first OFFSET text claims")
    args = ap.parse_args()
    model = VERIFICATION_MODEL

    src = Path("eval/scripts/verdict_confidence/data/eval_v1_modality.parquet")
    df = pl.read_parquet(src) if src.exists() else tag_modality(pl.read_parquet("eval/data/eval_v1.parquet"))
    claims = df.filter(pl.col("modality") == "text").slice(args.offset, args.limit).to_dicts()

    print(f"model={model}  cascade={'→'.join(SEARCH_CASCADE)}  K={args.samples}  temp={args.temp}")
    print(f"claims: {len(claims)} (text-modality, offset {args.offset})\n")
    embed("warmup")

    results, usage_probe = [], None
    for i, row in enumerate(claims):
        claim_text = row["claim_text"]
        print(f"[{i + 1}/{len(claims)}] {claim_text[:90]}")
        claim = AtomicClaim(text=claim_text, embedding=embed(claim_text).tolist())
        exclude = [row["review_url"]] if row.get("review_url") else []

        tv = time.time()
        v = verify(claim, date_ceiling=None, exclude_urls=exclude, providers=list(SEARCH_CASCADE))
        verify_s = round(time.time() - tv, 1)
        print(f"    verify: {verify_s}s, {v.llm_calls} llm calls, {v.rounds_used} rounds, "
              f"stop={v.stopped_reason}, {len(v.evidence_urls)} evidence urls")

        # One global token-usage probe (first claim only) for the $ projection.
        if usage_probe is None:
            p = _client.chat.completions.create(
                model=model, messages=vp.build_likert_messages(claim_text, v.analysis),
                response_format={"type": "json_object"}, temperature=args.temp,
                max_tokens=_EVALUATE_MAX_TOKENS)
            usage_probe = {"prompt_tokens": p.usage.prompt_tokens,
                           "completion_tokens": p.usage.completion_tokens}

        samples, likert_s = _resample_likert(claim_text, v.analysis, model, args.samples, args.temp)
        div = _diversity(samples)
        print(f"    {args.samples} Likert resamples: {likert_s}s "
              f"({likert_s / args.samples:.1f}s/call)")
        print(f"    veracity samples={[s['veracity'] for s in samples]}  std={div['std']['veracity']}  "
              f"pass_frac={div['binary_pass_fraction']}  distinct_vectors={div['n_distinct_vectors']}/{args.samples}  "
              f"distinct_justif={div['n_distinct_justifications']}/{args.samples}")
        print(f"    gold binary_label={row.get('binary_label')}\n")

        results.append({
            "claim_id": i + args.offset,
            "claim_text": claim_text,
            "gold_binary_label": row.get("binary_label"),
            "modality": row["modality"],
            "verify": {
                "elapsed_s": verify_s, "llm_calls": v.llm_calls, "rounds_used": v.rounds_used,
                "stopped_reason": v.stopped_reason, "providers_used": list(v.providers_used),
                "n_evidence_urls": len(v.evidence_urls), "analysis": v.analysis,
            },
            "production_likert_temp0p1": {d: v.scores.model_dump()[d] for d in DIMS},
            "samples": samples,
            "diversity": div,
            "likert_total_s": likert_s,
        })

    # --- Cost projection for the full study ---
    n_full = 2370
    avg_likert_s = np.mean([r["likert_total_s"] / args.samples for r in results])
    avg_verify_s = np.mean([r["verify"]["elapsed_s"] for r in results])
    print("=" * 70)
    print("COST PROJECTION (full study, N=2370, one run)")
    if usage_probe:
        pt, ct = usage_probe["prompt_tokens"], usage_probe["completion_tokens"]
        print(f"  Likert tokens/call: prompt~{pt}, completion~{ct} (incl. reasoning)")
        print(f"  full Likert calls = 2370×{args.samples} = {n_full * args.samples:,}  "
              f"→ ~{n_full * args.samples * (pt + ct) / 1e6:.1f}M tokens")
    print(f"  avg Likert latency: {avg_likert_s:.1f}s/call  →  resampling wall-clock "
          f"~{n_full * args.samples * avg_likert_s / 3600:.1f}h serial "
          f"(÷ workers in the real harness)")
    print(f"  avg verify latency: {avg_verify_s:.0f}s/claim  →  verify pass ~"
          f"{n_full * avg_verify_s / 3600:.1f}h serial")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"config": vars(args), "usage_probe": usage_probe,
                               "results": results}, indent=1, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
