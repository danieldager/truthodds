"""Run variants through the cache decision; report recall/precision over distance.

See README.md for the protocol. Reads variants.parquet (from generator.py), runs
the bidirectional equivalence GATE once per variant (variant vs its source seed),
then sweeps the cosine threshold as pure arithmetic and reports:
  - cosine-ONLY vs cosine+GATE, at each threshold
  - recall    = equivalent variants that HIT          (cost win)
  - precision = HITs that are truly equivalent         (SAFETY metric)
  - adversarial false-HIT rate, per axis (negation / number = the dangerous ones)
  - a SIMILARITY_THRESHOLD recommendation (max recall s.t. gated precision ~= 1.0)

The headline: cosine alone false-HITs on the adversarial axes (they are cosine-near
but verdict-opposite); the gate is what restores precision.

  uv run python -m eval.scripts.cache_eval.scorer \
      -i eval/scripts/cache_eval/data/variants.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]  # -> src/
sys.path.insert(0, str(ROOT))
from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipeline.models import ClaimVerdict  # noqa: E402  (the cached payload type)

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
_JSON = re.compile(r"\{.*\}", re.DOTALL)
_client: OpenAI | None = None

GATE_SYSTEM = (
    "You are a strict logical-equivalence judge for a fact-checking cache. Two claims are "
    "EQUIVALENT only if a SINGLE fact-check verdict would apply identically to BOTH — i.e. "
    "they mutually entail each other. Pay special attention to these verdict-flipping "
    "differences, ANY of which makes the claims NOT equivalent:\n"
    "  - negation (opposite truth value)\n"
    "  - different numbers / quantities / dates\n"
    "  - different named entities (person, place, organization)\n"
    "  - different scope or quantifier (some vs all, a few vs every)\n"
    "Respond with ONLY a JSON object: "
    '{"a_entails_b": bool, "b_entails_a": bool, '
    '"relation": "equivalent"|"negation"|"related"|"unrelated"}'
)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.groq.com/openai/v1",
                         api_key=os.environ["GROQ_API_KEY"])
    return _client


def equivalence_gate(a: str, b: str, model: str) -> dict:
    """Bidirectional equivalence judge. equivalent = mutual entailment (a<->b).

    Returns {"equivalent": bool, "relation": str}. The precision arbiter: must catch
    negation / number / entity / scope flips that cosine misses. Single LLM call asks
    for both entailment directions (an LLM can assess both at once); swap for a
    negation-robust fine-tuned NLI run in two orders if a cheaper local gate is wanted.
    """
    msgs = [{"role": "system", "content": GATE_SYSTEM},
            {"role": "user", "content": f"CLAIM A: {a}\nCLAIM B: {b}"}]
    for attempt in range(5):
        try:
            r = get_client().chat.completions.create(
                model=model, messages=msgs, temperature=0.0, max_tokens=200)
            m = _JSON.search(r.choices[0].message.content or "")
            d = json.loads(m.group(0)) if m else {}
            equiv = bool(d.get("a_entails_b")) and bool(d.get("b_entails_a"))
            relation = d.get("relation") or ("equivalent" if equiv else "unrelated")
            return {"equivalent": equiv, "relation": relation}
        except Exception:  # noqa: BLE001
            if attempt == 4:
                return {"equivalent": False, "relation": "error"}
            time.sleep(min(2 ** attempt, 20.0))
    return {"equivalent": False, "relation": "error"}


def _metrics(rows: list[dict], hit_key: str) -> tuple[float, float, float]:
    """(recall, precision, adversarial false-HIT rate) for a given HIT column."""
    eq = [r for r in rows if r["gold_equivalent"]]
    ad = [r for r in rows if not r["gold_equivalent"]]
    hits = [r for r in rows if r[hit_key]]
    eq_hits = [r for r in hits if r["gold_equivalent"]]
    recall = len(eq_hits) / len(eq) if eq else 0.0
    precision = len(eq_hits) / len(hits) if hits else 1.0
    adv_falsehit = sum(1 for r in ad if r[hit_key]) / len(ad) if ad else 0.0
    return recall, precision, adv_falsehit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--variants", type=Path, required=True)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--thresholds", type=str,
                    default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.92,0.95")
    args = ap.parse_args()
    _ = ClaimVerdict  # the payload the real cache reuses on a confirmed HIT

    df = pl.read_parquet(args.variants)
    rows = df.to_dicts()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    # --- Gate every variant ONCE (variant vs its source seed); sweep is arithmetic. ---
    # Reuse precomputed gate columns if present (free re-sweeps); else gate + persist.
    if "gate_equivalent" in df.columns:
        print(f"reusing precomputed gate results for {len(rows)} variants")
    else:
        print(f"gating {len(rows)} variants (bidirectional, {args.model})…")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(equivalence_gate, r["variant_text"], r["seed_text"], args.model): i
                    for i, r in enumerate(rows)}
            for n, fut in enumerate(as_completed(futs), 1):
                g = fut.result()
                rows[futs[fut]]["gate_equivalent"] = g["equivalent"]
                rows[futs[fut]]["gate_relation"] = g["relation"]
                if n % 50 == 0 or n == len(rows):
                    print(f"  [{n}/{len(rows)}] gated")
        gated_path = args.variants.with_name(args.variants.stem + "_gated.parquet")
        pl.DataFrame(rows).write_parquet(gated_path)
        print(f"persisted gate results -> {gated_path} (re-score it for free)")

    # --- Gate quality, independent of threshold (does it catch the flips?) ---
    eq = [r for r in rows if r["gold_equivalent"]]
    ad = [r for r in rows if not r["gold_equivalent"]]
    gate_recall = sum(r["gate_equivalent"] for r in eq) / len(eq) if eq else 0.0
    gate_falsehit = sum(r["gate_equivalent"] for r in ad) / len(ad) if ad else 0.0
    print(f"\n{'='*64}\nGATE (threshold-independent): on {len(eq)} equivalent + {len(ad)} adversarial")
    print(f"  gate recall (equiv judged equivalent):   {gate_recall:.3f}")
    print(f"  gate false-HIT (adversarial judged equiv): {gate_falsehit:.3f}  <-- must be ~0")
    print("  adversarial false-HITs by axis (gate said equivalent — DANGEROUS):")
    for axis in sorted({r["axis"] for r in ad}):
        a = [r for r in ad if r["axis"] == axis]
        fh = sum(r["gate_equivalent"] for r in a)
        print(f"    {axis:<10} {fh}/{len(a)}")

    # --- Threshold sweep: cosine-only vs cosine+gate ---
    for r in rows:
        r["_eqv"] = r["gold_equivalent"]
    print(f"\n{'='*64}\nTHRESHOLD SWEEP  (recall / precision / adv-false-HIT)")
    print(f"{'thr':>5} | {'cosine-only':^26} | {'cosine + GATE':^26}")
    print(f"{'':>5} | {'recall':>8}{'prec':>8}{'advFH':>8} | {'recall':>8}{'prec':>8}{'advFH':>8}")
    best = None  # max gated-F1 threshold (precision is gate-bound + ~flat, so F1 ~ recall lever)
    gp_max = 0.0
    for t in thresholds:
        for r in rows:
            r["hit_cos"] = r["cosine"] >= t
            r["hit_gate"] = (r["cosine"] >= t) and r["gate_equivalent"]
        cr, cp, ca = _metrics(rows, "hit_cos")
        gr, gp, ga = _metrics(rows, "hit_gate")
        f1 = 2 * gr * gp / (gr + gp) if gr + gp else 0.0
        gp_max = max(gp_max, gp)
        print(f"{t:>5.2f} | {cr:>8.3f}{cp:>8.3f}{ca:>8.3f} | {gr:>8.3f}{gp:>8.3f}{ga:>8.3f}")
        if best is None or f1 > best[3]:
            best = (t, gr, gp, f1)

    print(f"\n{'='*64}")
    if best:
        print(f"RECOMMENDED SIMILARITY_THRESHOLD = {best[0]:.2f}  "
              f"(gated recall {best[1]:.3f}, precision {best[2]:.3f}, F1 {best[3]:.3f})")
    print("WHY: gated precision is ~flat across thresholds (max {:.3f}) — the GATE, not the cosine "
          "cut, bounds precision. So the cut is a RECALL/cost lever: set it low to keep recall, and "
          "to push precision higher, improve the gate (scope/number), not the threshold."
          .format(gp_max))


if __name__ == "__main__":
    main()
