"""
Head-to-head verification model comparison on AVeriTeC dev split.

Retrieval runs once per claim and is shared across all configs.
Only the verification call varies — so differences in output are purely model differences.

Edit CONFIGS below to add/remove models. Set disable_thinking=True for reasoning
models (DeepSeek-R1, QwQ) — Groq disables their chain-of-thought via extra_body.

Usage:
    uv run python3 compare.py               # 10 claims, seed 42
    uv run python3 compare.py --n 5         # 5 claims
    uv run python3 compare.py --seed 1      # different sample
    uv run python3 compare.py --verbose     # show Likert scores + reasoning
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI

from config import (
    EXTRACTION_MODEL,
    VERIFICATION_BASE_URL,
    VERIFICATION_API_KEY,
    VERDICT_OPTIONS,
)
from pipeline.claim_extraction import extract_claims
from pipeline.models import AtomicClaim, ClaimEvidence, LikertScores, VerdictResult
from pipeline.retrieval import retrieve_evidence
from pipeline.verification import (
    _format_evidence,
    _format_questions,
    _parse_likert,
    _validate_verdict,
    _SYSTEM as VERIFY_SYSTEM,
)

# ── Model configs ─────────────────────────────────────────────────────────────
# Add or remove entries here. disable_thinking passes Groq's thinking-disable
# parameter for reasoning models (DeepSeek-R1, QwQ). Has no effect on standard
# instruction-following models.

@dataclass
class ModelConfig:
    name: str
    model: str


CONFIGS: List[ModelConfig] = [
    ModelConfig(name="llama-3.3-70b",    model="llama-3.3-70b-versatile"),
    ModelConfig(name="llama-4-scout-17b", model="meta-llama/llama-4-scout-17b-16e-instruct"),
]

# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "Supported": "Supported",
    "Refuted": "Refuted",
    "Not Enough Evidence": "Not Enough Evidence",
    "Conflicting Evidence": "Conflicting Evidence",
    "Cherrypicking": "Conflicting Evidence",
}

W = 72
_client = OpenAI(base_url=VERIFICATION_BASE_URL, api_key=VERIFICATION_API_KEY)


def _verify_with(ce: ClaimEvidence, cfg: ModelConfig) -> VerdictResult:
    claim = ce.claim.text
    if not ce.evidence:
        nei = LikertScores(Supported=1, Refuted=1, NotEnoughEvidence=5, ConflictingEvidence=1)
        return VerdictResult(claim=claim, reasoning="No evidence retrieved.",
                             verdict="Not Enough Evidence", likert=nei,
                             justification="No evidence.", key_sources=[])

    user_content = (
        f"Claim:\n{claim}\n\n"
        f"Verifying questions:\n{_format_questions(ce.claim.questions)}\n\n"
        f"Evidence:\n{_format_evidence(ce.evidence)}"
    )

    response = _client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(raw)
    likert = _parse_likert(data.get("likert", {}))
    reasoning = data.get("reasoning", "")
    if isinstance(reasoning, list):
        reasoning = " ".join(str(x) for x in reasoning)
    return VerdictResult(
        claim=claim,
        reasoning=reasoning,
        verdict=_validate_verdict(data.get("verdict", "Not Enough Evidence")),
        likert=likert,
        justification=data.get("justification", ""),
        key_sources=data.get("key_sources", []),
    )


def _confidence(vr: VerdictResult) -> float:
    return {
        "Supported": vr.likert.Supported,
        "Refuted": vr.likert.Refuted,
        "Not Enough Evidence": vr.likert.NotEnoughEvidence,
        "Conflicting Evidence": vr.likert.ConflictingEvidence,
    }.get(vr.verdict, 1) / 5.0


def _likert_str(vr: VerdictResult) -> str:
    l = vr.likert
    return (f"S={l.Supported} R={l.Refuted} "
            f"N={l.NotEnoughEvidence} C={l.ConflictingEvidence}")


def _parse_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").strftime("%m/%d/%Y")
    except ValueError:
        return None


def main(n: int, seed: int, verbose: bool) -> None:
    ds = load_dataset("pminervini/averitec", split="dev")
    indices = random.Random(seed).sample(range(len(ds)), n)
    samples = [ds[i] for i in indices]

    # per-config tallies
    tallies = {cfg.name: {"correct": 0, "total": 0} for cfg in CONFIGS}

    for i, sample in enumerate(samples, 1):
        post = sample["claim"]
        gold = LABEL_MAP.get(sample["label"], sample["label"])
        date_cutoff = _parse_date(sample.get("claim_date"))

        print(f"\n{'═'*W}")
        print(f"  CLAIM {i}/{n}  |  Gold: {gold}")
        print(f"  {post[:W-2]}")
        print(f"{'═'*W}")

        # ── Extraction (shared) ───────────────────────────────────────────────
        extraction = extract_claims(post)
        if not extraction.is_checkable:
            print("  ✗ Not checkable — skipping")
            for cfg in CONFIGS:
                tallies[cfg.name]["total"] += 1
            continue

        print(f"  Extraction [{EXTRACTION_MODEL}]: "
              f"{len(extraction.claims)} claim(s)")

        # ── Retrieval (shared across all configs) ─────────────────────────────
        print(f"  Retrieval ...")
        claim_evidences: List[ClaimEvidence] = []
        for claim in extraction.claims:
            ce = retrieve_evidence(claim, date_cutoff=date_cutoff, verbose=False)
            claim_evidences.append(ce)
            print(f"    [{claim.text[:55]}] → {len(ce.evidence)} items")

        # ── Verification (once per config) ────────────────────────────────────
        print()
        name_w = max(len(cfg.name) for cfg in CONFIGS) + 2
        hdr = f"  {'MODEL':<{name_w}}  {'VERDICT':<24}  CONF  LIKERT (S/R/N/C)  MATCH"
        print(hdr)
        print(f"  {'─'*(W-2)}")

        for cfg in CONFIGS:
            results = [_verify_with(ce, cfg) for ce in claim_evidences]

            # simple aggregation: worst-case / first-refuted wins
            verdict = results[0].verdict
            conf = _confidence(results[0])
            for r in results[1:]:
                if r.verdict == "Refuted":
                    verdict, conf = r.verdict, _confidence(r)
                    break
                if r.verdict == "Conflicting Evidence" and verdict not in ("Refuted",):
                    verdict, conf = r.verdict, _confidence(r)

            match = verdict == gold
            tallies[cfg.name]["correct"] += match
            tallies[cfg.name]["total"] += 1

            tag = "✓" if match else "✗"
            lk = _likert_str(results[0]) if len(results) == 1 else "multi"
            print(f"  {cfg.name:<{name_w}}  {verdict:<24}  {conf:.2f}  {lk:<16}  {tag}")

            if verbose:
                for j, r in enumerate(results):
                    label = chr(64 + j + 1)
                    print(f"      [{label}] {r.justification}")
                    print(f"           {_likert_str(r)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  SUMMARY  (n={n}, seed={seed})")
    print(f"{'═'*W}")
    name_w = max(len(cfg.name) for cfg in CONFIGS) + 2
    print(f"  {'MODEL':<{name_w}}  CORRECT   ACCURACY")
    print(f"  {'─'*(W-2)}")
    for cfg in CONFIGS:
        t = tallies[cfg.name]
        acc = 100 * t["correct"] / t["total"] if t["total"] else 0
        print(f"  {cfg.name:<{name_w}}  {t['correct']}/{t['total']:<6}    {acc:.0f}%")
    print(f"{'═'*W}")

    # save results
    out_path = f"docs/compare_{n}_{seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "n": n, "seed": seed,
            "configs": [cfg.name for cfg in CONFIGS],
            "tallies": tallies,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.n, args.seed, args.verbose)
