"""
Dashboard-style pipeline evaluation against AVeriTeC dev split.

Usage:
    uv run python3 evaluate.py                          # 5 claims, seed 1, FIRE
    uv run python3 evaluate.py --n 50 --seed 42         # 50 claims
    uv run python3 evaluate.py --approach sc --n 200    # self-consistency, 200 claims
    uv run python3 evaluate.py --approach baseline      # single-pass, no FIRE
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from config import EXTRACTION_MODEL, MAX_RETRIEVAL_ROUNDS, VERIFICATION_MODEL
from pipeline.aggregation import aggregate_verdicts
from pipeline.claim_extraction import extract_claims
from pipeline.iterative import iterative_verify
from pipeline.retrieval import retrieve_evidence
from pipeline.verification import verify_claim, verify_claim_sc

APPROACHES = ("baseline", "fire", "sc")

SC_N = 3  # number of samples for self-consistency


def _ckpt_path(n: int, seed: int, approach: str, verifier: str = "") -> str:
    suffix = f"_{verifier.replace('/', '-')}" if verifier else ""
    return f"docs/eval_ckpt_{n}_{seed}_{approach}{suffix}.json"


def _load_ckpt(n: int, seed: int, approach: str, verifier: str = "") -> tuple[list, int]:
    path = _ckpt_path(n, seed, approach, verifier)
    if os.path.exists(path):
        saved = json.load(open(path))
        return saved["results"], saved["next"]
    return [], 0


def _save_ckpt(n: int, seed: int, approach: str, results: list, next_i: int, verifier: str = "") -> None:
    json.dump({"results": results, "next": next_i}, open(_ckpt_path(n, seed, approach, verifier), "w"))


def _append_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


LABEL_MAP = {
    "Supported": "Supported",
    "Refuted": "Refuted",
    "Not Enough Evidence": "Not Enough Evidence",
    "Conflicting Evidence": "Conflicting Evidence",
    "Cherrypicking": "Conflicting Evidence",
}

W = 70  # display width


def bar(char="─", width=W):
    return char * width


def parse_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.strftime("%m/%d/%Y")
    except ValueError:
        return None


def main(n: int, seed: int, approach: str, verifier: str = "") -> None:
    if verifier:
        import pipeline.verification as _vmod
        import pipeline.iterative as _imod
        _vmod.VERIFICATION_MODEL = verifier
        _imod.VERIFICATION_MODEL = verifier

    effective_model = verifier or VERIFICATION_MODEL

    results, start = _load_ckpt(n, seed, approach, verifier)
    if start > 0:
        print(f"\n  Resuming from claim {start + 1}/{n}  ({start} already done)\n")

    ds = load_dataset("pminervini/averitec", split="dev")
    indices = random.Random(seed).sample(range(len(ds)), n)
    samples = [ds[i] for i in indices]

    for i, sample in enumerate(samples, 1):
        if i <= start:
            continue
        post = sample["claim"]
        gold = LABEL_MAP.get(sample["label"], sample["label"])
        date_cutoff = parse_date(sample.get("claim_date"))
        source = sample.get("reporting_source", "unknown")

        print(f"\n{'═'*W}")
        print(f"  CLAIM {i}/{n}")
        print(f"{'═'*W}")
        print(f"  Source : {source}   Date: {sample.get('claim_date', 'N/A')}")
        print(f"  Gold   : {gold}")
        print(f"  Post   : {post}")

        # ── STEP 1: Claim extraction ──────────────────────────────────────────
        print(f"\n{bar()}")
        print(f"  STEP 1 — CLAIM EXTRACTION  [{EXTRACTION_MODEL}]")
        print(bar())

        extraction = extract_claims(post)

        if not extraction.is_checkable:
            print("  ✗ Not checkable — pipeline exits early")
            results.append({"gold": gold, "post_verdict": "Not Checkable", "post_confidence": 0.0,
                            "sub_verdicts": [], "match": False, "source": source})
            _save_ckpt(n, seed, approach, results, i, verifier)
            continue

        print(f"  Checkable: Yes   |   {len(extraction.claims)} atomic claim(s)\n")
        for ci, claim in enumerate(extraction.claims, 1):
            label = chr(64 + ci)
            print(f"  [{label}] {claim.text}")
            for qi, q in enumerate(claim.queries, 1):
                print(f"       Q{qi}: {q}")

        # ── STEP 2: Retrieval ─────────────────────────────────────────────────
        print(f"\n{bar()}")
        print(f"  STEP 2 — RETRIEVAL  [Serper → Trafilatura → CRED-1]")
        print(bar())

        claim_evidences = []
        for ci, claim in enumerate(extraction.claims, 1):
            label = chr(64 + ci)
            print(f"\n  [{label}] {claim.text[:65]}")
            ce = retrieve_evidence(claim, date_cutoff=date_cutoff, verbose=True)
            print(f"      → {len(ce.evidence)} evidence items  |  {ce.blocked_count} blocked by domain filter")
            claim_evidences.append((label, claim, ce))

        # ── STEP 3: Verification ──────────────────────────────────────────────
        step3_label = {
            "baseline": f"single-pass",
            "fire":     f"FIRE ≤{MAX_RETRIEVAL_ROUNDS} rounds",
            "sc":       f"self-consistency n={SC_N}",
        }[approach]
        print(f"\n{bar()}")
        print(f"  STEP 3 — VERIFICATION ({step3_label})  [{effective_model}]")
        print(bar())

        verdicts = []
        for label, claim, ce in claim_evidences:
            if approach == "fire":
                vr = iterative_verify(claim, ce, date_cutoff=date_cutoff, verbose=True)
            elif approach == "sc":
                vr = verify_claim_sc(ce, n=SC_N)
            else:
                vr = verify_claim(ce)
            verdicts.append(vr)

            match_gold = "✓" if vr.verdict == gold else "✗"
            conf_bar = "█" * int(vr.confidence * 10) + "░" * (10 - int(vr.confidence * 10))

            print(f"\n  [{label}] {claim.text[:65]}")
            print(f"       Verdict    : {vr.verdict}  {match_gold} (gold={gold})")
            print(f"       Confidence : {conf_bar}  {vr.confidence:.2f}")
            print(f"       Justific.  : {vr.justification}")
            print(f"       Reasoning  :")
            words = vr.reasoning.split()
            line, lines = [], []
            for w in words:
                if sum(len(x)+1 for x in line) + len(w) > W - 18:
                    lines.append(" ".join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines.append(" ".join(line))
            for ln in lines:
                print(f"                  {ln}")
            if vr.key_sources:
                print(f"       Sources    :")
                for s in vr.key_sources:
                    print(f"         • {s}")

        # ── POST-LEVEL AGGREGATION ────────────────────────────────────────────
        post_verdict, post_confidence = aggregate_verdicts(verdicts)
        match = post_verdict == gold

        print(f"\n{bar('─')}")
        conf_bar = "█" * int(post_confidence * 10) + "░" * (10 - int(post_confidence * 10))
        match_tag = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"  POST VERDICT  {post_verdict}  {conf_bar}  {post_confidence:.2f}  |  Gold={gold}  |  {match_tag}")
        if len(verdicts) > 1:
            print(f"  Sub-claims: {[v.verdict for v in verdicts]}")
        print(bar('─'))

        results.append({
            "gold": gold, "post_verdict": post_verdict, "post_confidence": post_confidence,
            "sub_verdicts": [v.verdict for v in verdicts], "match": match, "source": source,
        })
        _save_ckpt(n, seed, approach, results, i, verifier)

    # cleanup checkpoint on successful completion
    ckpt = _ckpt_path(n, seed, approach, verifier)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    correct = sum(r["match"] for r in results)
    print(f"\n{'═'*W}")
    print(f"  SUMMARY   {correct}/{n} correct  ({100*correct/n:.0f}%)")
    print(f"{'═'*W}")
    for i, r in enumerate(results, 1):
        tag = "✓" if r["match"] else "✗"
        sub = f"  sub={r['sub_verdicts']}" if len(r["sub_verdicts"]) > 1 else ""
        print(f"  {tag}  [{i}]  gold={r['gold']:<30}  post={r['post_verdict']}{sub}")

    # ── CSV LOGGING ───────────────────────────────────────────────────────────
    model_tag = effective_model.replace("/", "-")
    run_id = f"{approach}_{n}_{seed}_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    by_label: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_label[r["gold"]]["total"] += 1
        if r["match"]:
            by_label[r["gold"]]["correct"] += 1

    def acc(label):
        t = by_label[label]["total"]
        return round(by_label[label]["correct"] / t, 4) if t else None

    # ECE: weighted gap between post_confidence and accuracy across 10 bins
    # AUROC: probability that a correct prediction has higher confidence than an incorrect one
    from docs.plot_results import auroc as _auroc, ece as _ece
    confs = [r.get("post_confidence", 0.0) for r in results]
    corrs = [r["match"] for r in results]
    ece_score = round(_ece(confs, corrs, n_bins=10), 4) if results else None
    auroc_score = round(_auroc(confs, corrs), 4) if results else None

    run_row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "approach": approach,
        "sc_n": SC_N if approach == "sc" else 1,
        "n_claims": len(results),
        "seed": seed,
        "accuracy": round(correct / len(results), 4) if results else None,
        "ece_score": ece_score,
        "auroc_score": auroc_score,
        "n_refuted": by_label["Refuted"]["total"],
        "n_supported": by_label["Supported"]["total"],
        "n_nei": by_label["Not Enough Evidence"]["total"],
        "n_conflicting": by_label["Conflicting Evidence"]["total"],
        "acc_refuted": acc("Refuted"),
        "acc_supported": acc("Supported"),
        "acc_nei": acc("Not Enough Evidence"),
        "acc_conflicting": acc("Conflicting Evidence"),
        "extraction_model": EXTRACTION_MODEL,
        "verification_model": effective_model,
    }
    _append_csv(Path("docs/eval_results.csv"), run_row)

    for idx, r in enumerate(results, 1):
        claim_row = {
            "run_id": run_id,
            "claim_idx": idx,
            "source": r.get("source", ""),
            "gold": r["gold"],
            "post_verdict": r["post_verdict"],
            "correct": r["match"],
            "post_confidence": r.get("post_confidence", ""),
        }
        _append_csv(Path("docs/eval_claims.csv"), claim_row)

    print(f"\n  Results logged → docs/eval_results.csv  (run_id={run_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--approach", choices=APPROACHES, default="fire",
                        help="baseline=single verify_claim, fire=iterative FIRE, sc=self-consistency")
    parser.add_argument("--verifier", type=str, default="",
                        help="Override VERIFICATION_MODEL (e.g. qwen/qwen3-32b, openai/gpt-oss-120b)")
    args = parser.parse_args()
    main(args.n, args.seed, args.approach, args.verifier)
