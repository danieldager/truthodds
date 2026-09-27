"""Score predictions with METEOR + BERTScore.

METEOR is the CheckThat! official metric. BERTScore-F1 (roberta-large) captures
semantic similarity that METEOR's lexical-overlap bias misses.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.translate.meteor_score import single_meteor_score

for pkg in ("wordnet", "punkt_tab", "omw-1.4"):
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.tokenize import word_tokenize  # noqa: E402


def _tokenize(s: str) -> list[str]:
    return word_tokenize(s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--show-examples", type=int, default=0,
                    help="print N qualitative examples per system")
    ap.add_argument("--no-bertscore", action="store_true",
                    help="skip BERTScore (faster; METEOR only)")
    args = ap.parse_args()

    gold_path = args.predictions.with_suffix(".gold.jsonl")
    gold_by_id = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            gold_by_id[g["id"]] = g

    by_system: dict[str, list[dict]] = defaultdict(list)
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_system[r["system"]].append(r)

    # METEOR (per-example)
    meteor_by_system: dict[str, list[float]] = {}
    errs_by_system: dict[str, int] = {}
    for system, rows in by_system.items():
        scores = []
        errs = 0
        for r in rows:
            if r["error"]:
                errs += 1
                continue
            gold = gold_by_id[r["id"]]["gold"]
            scores.append(single_meteor_score(_tokenize(gold), _tokenize(r["pred"])))
        meteor_by_system[system] = scores
        errs_by_system[system] = errs

    # BERTScore (batched across all systems for speed)
    bert_by_system: dict[str, list[float]] = {}
    if not args.no_bertscore:
        # transformers>=5 removed build_inputs_with_special_tokens from fast tokenizers,
        # but bert-score 0.3.13 still calls it. Restore for Roberta-family tokenizers.
        from transformers import PreTrainedTokenizerBase  # noqa: PLC0415
        if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
            def _build(self, token_ids_0, token_ids_1=None):
                bos = [self.bos_token_id] if self.bos_token_id is not None else [self.cls_token_id]
                eos = [self.eos_token_id] if self.eos_token_id is not None else [self.sep_token_id]
                if token_ids_1 is None:
                    return bos + list(token_ids_0) + eos
                return bos + list(token_ids_0) + eos + eos + list(token_ids_1) + eos
            PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build  # type: ignore[attr-defined]
        from bert_score import score as bert_score  # noqa: PLC0415

        cands: list[str] = []
        refs: list[str] = []
        index: list[tuple[str, int]] = []  # (system, row_idx within system's non-error rows)
        for system, rows in by_system.items():
            for r in rows:
                if r["error"]:
                    continue
                cands.append(r["pred"] or "")
                refs.append(gold_by_id[r["id"]]["gold"])
                index.append((system, len(bert_by_system.get(system, []))))
                bert_by_system.setdefault(system, []).append(None)  # placeholder
        import torch  # noqa: PLC0415
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Computing BERTScore on {len(cands)} (pred, gold) pairs (device={device})...")
        _, _, F1 = bert_score(cands, refs, lang="en", model_type="roberta-large",
                              rescale_with_baseline=True, verbose=True, device=device, batch_size=32)
        f1_list = F1.tolist()
        # Refill
        bert_by_system = defaultdict(list)
        for (system, _), f in zip(index, f1_list):
            bert_by_system[system].append(f)

    print(f"\nDataset: {args.predictions.name}  ({len(gold_by_id)} posts)\n")
    if args.no_bertscore:
        print(f"{'System':<14} {'METEOR':>8} {'N':>5} {'errs':>5} {'avg_lat_s':>10}")
        print("-" * 50)
    else:
        print(f"{'System':<14} {'METEOR':>8} {'BERTScore':>10} {'N':>5} {'errs':>5} {'avg_lat_s':>10}")
        print("-" * 62)

    summary = []
    for system in sorted(by_system.keys()):
        rows = by_system[system]
        m_scores = meteor_by_system[system]
        m_avg = sum(m_scores) / len(m_scores) if m_scores else 0.0
        avg_lat = sum(r["latency_s"] for r in rows) / len(rows)
        if args.no_bertscore:
            print(f"{system:<14} {m_avg:>8.4f} {len(m_scores):>5} {errs_by_system[system]:>5} {avg_lat:>10.2f}")
        else:
            b_scores = bert_by_system[system]
            b_avg = sum(b_scores) / len(b_scores) if b_scores else 0.0
            print(f"{system:<14} {m_avg:>8.4f} {b_avg:>10.4f} {len(m_scores):>5} {errs_by_system[system]:>5} {avg_lat:>10.2f}")
        summary.append((system, m_avg, m_scores, rows))

    if args.show_examples:
        print("\n" + "=" * 70)
        print(f"QUALITATIVE EXAMPLES (first {args.show_examples} per system)")
        print("=" * 70)
        for system, _, _, rows in summary:
            print(f"\n--- {system} ---")
            for r in rows[: args.show_examples]:
                if r["error"]:
                    continue
                g = gold_by_id[r["id"]]
                post_preview = g["post"][:120].replace("\n", " ")
                print(f"\n  POST: {post_preview}{'...' if len(g['post']) > 120 else ''}")
                print(f"  GOLD: {g['gold']}")
                print(f"  PRED: {r['pred']}")


if __name__ == "__main__":
    main()
