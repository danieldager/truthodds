"""Phase-4 sampler: draw the locked survey subsets from the 50-source harvest.

Per source (design: docs/survey_5k_roadmap.md):
  1. EXCLUDE: identified foreign-language posts (EN-only round) and quote-tweets (quoted
     content isn't in our text field, so the post is not self-contained), plus
     promo/housekeeping posts (regex). Short and image-only posts are KEPT (Daniel,
     2026-07-10): production sees them, and gracefully emitting no checkworthy claims on
     them is measured behavior — so X's non-linguistic lang codes (zxx no-text, qme
     media-link, qst short-text, qam/qht mentions/hashtags-only) pass; 'und' does not
     (can't confirm the post is EN).
  2. DEDUPE same-story clusters: content-word Jaccard >= --dupe-threshold within a
     source marks two posts as the same story; keep the earliest post per cluster.
     (Catches the dominant case — outlets re-posting the same headline/article all
     day. Paraphrased same-story pairs are partially caught; the audit measures residue.)
  3. TIME-STRATIFIED DRAW: sort survivors by created_at, systematic sample (every
     k-th, seeded random start) -> even coverage of the source's whole harvest window.
     Eval (100/source) is drawn first; dev (10/source) from the remainder the same way.
     Disjoint by construction. Sources with fewer usable posts fill eval first (flagged).

Writes survey_eval_5k.parquet + survey_dev_500.parquet (roster band/lean/register merged
in) and a provenance JSON (seed, thresholds, source-parquet hash, per-source funnel).

  cd src && uv run python eval/scripts/claim_sourcing/sample_survey_subsets.py \
      [--seed 20260710] [--dupe-threshold 0.6] [--eval-n 100] [--dev-n 10]
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "survey_claims"
TWEETS = DATA / "outlet_tweets.parquet"
ROSTER = DATA / "source_roster.csv"

SAMPLER_VERSION = "1.1"   # 1.1: keep short + image-only posts (production-graceful eval)
_KEEP_LANGS = frozenset({"en", "zxx", "qme", "qst", "qam", "qht"})
_URL_RE = re.compile(r"https?://\S+")
_PROMO_RE = re.compile(
    r"\b(?:subscribe|sign up|signup|newsletter|sweepstakes|giveaway|promo code|discount|"
    r"use code|% off|shop now|on sale|our merch|store)\b", re.I)
_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP = frozenset("a an and are as at be but by for from has he in is it its of on or "
                  "s that the to was were will with this these those i you we they".split())


def content_words(text: str) -> frozenset:
    t = _URL_RE.sub(" ", text.lower())
    return frozenset(w for w in _WORD_RE.findall(t) if w not in _STOP and len(w) > 2)


def dedupe_same_story(g: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    """Greedy same-story clustering by content-word Jaccard; keeps the EARLIEST post
    per cluster. O(n^2) per source (~560 posts) is fine."""
    g = g.sort_values("created_at").reset_index(drop=True)
    words = [content_words(t) for t in g.text]
    keep, kept_words = [], []
    for i in range(len(g)):
        wi = words[i]
        dup = False
        if wi:
            for wk in kept_words:
                if not wk:
                    continue
                inter = len(wi & wk)
                if inter and inter / len(wi | wk) >= threshold:
                    dup = True
                    break
        if not dup:
            keep.append(i)
            kept_words.append(wi)
    return g.iloc[keep], len(g) - len(keep)


def systematic_draw(g: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Every-k-th draw over the time-sorted frame with a seeded random start —
    even coverage of the source's whole time window."""
    g = g.sort_values("created_at").reset_index(drop=True)
    if len(g) <= n:
        return g
    step = len(g) / n
    start = rng.random() * step
    idx = sorted({min(len(g) - 1, int(start + k * step)) for k in range(n)})
    # set() can collapse adjacent indices on rounding; top up from unused positions
    pool = [i for i in range(len(g)) if i not in set(idx)]
    while len(idx) < n and pool:
        idx.append(pool.pop(0))
    return g.iloc[sorted(idx)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--dupe-threshold", type=float, default=0.6)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--dev-n", type=int, default=10)
    args = ap.parse_args()
    import random
    rng = random.Random(args.seed)

    df = pd.read_parquet(TWEETS)
    src_hash = hashlib.sha256(TWEETS.read_bytes()).hexdigest()[:16]
    roster = pd.read_csv(ROSTER)
    meta_cols = roster[["domain", "ng_score", "lean", "register", "followers"]]

    funnel, eval_parts, dev_parts = [], [], []
    for domain, g in df.groupby("domain", sort=True):
        n0 = len(g)
        g = g[g.lang.isin(_KEEP_LANGS)
              & ~g.is_quote.astype(bool) & ~g.text.str.contains(_PROMO_RE)]
        n_excl = n0 - len(g)
        g, n_dupes = dedupe_same_story(g, args.dupe_threshold)
        ev = systematic_draw(g, args.eval_n, rng)
        rest = g[~g.post_id.isin(set(ev.post_id))]
        dv = systematic_draw(rest, args.dev_n, rng)
        funnel.append({"domain": domain, "harvested": n0, "excluded": n_excl,
                       "same_story_dupes": n_dupes, "usable": len(g),
                       "eval": len(ev), "dev": len(dv),
                       "short": bool(len(ev) < args.eval_n or len(dv) < args.dev_n)})
        eval_parts.append(ev)
        dev_parts.append(dv)

    ev = pd.concat(eval_parts).merge(meta_cols, on="domain", how="left")
    dv = pd.concat(dev_parts).merge(meta_cols, on="domain", how="left")
    assert not set(ev.post_id) & set(dv.post_id), "dev/eval overlap!"
    assert ev.post_id.is_unique and dv.post_id.is_unique

    ev.to_parquet(DATA / "survey_eval_5k.parquet", index=False)
    dv.to_parquet(DATA / "survey_dev_500.parquet", index=False)
    prov = {"sampler_version": SAMPLER_VERSION, "seed": args.seed,
            "dupe_threshold": args.dupe_threshold,
            "eval_n": args.eval_n, "dev_n": args.dev_n,
            "source_parquet_sha256_16": src_hash,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "eval_rows": len(ev), "dev_rows": len(dv), "funnel": funnel}
    (DATA / "survey_subsets_provenance.json").write_text(json.dumps(prov, indent=2))

    f = pd.DataFrame(funnel)
    print(f.to_string(index=False))
    print(f"\neval: {len(ev)} rows -> survey_eval_5k.parquet")
    print(f"dev:  {len(dv)} rows -> survey_dev_500.parquet")
    print(f"provenance -> survey_subsets_provenance.json (source hash {src_hash})")
    short = f[f["short"]]
    if len(short):
        print(f"\n⚠️  SHORT sources (couldn't fill quota): {list(short.domain)}")


if __name__ == "__main__":
    main()
