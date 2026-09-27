"""Independent-judge cross-check of the image-dependence gate on HARD cases.

  uv run python -m eval.scripts.image_gate_crosscheck --judge google/gemma-3-27b-it --n 100

The gate (image_audit.py) uses Qwen3-VL — the SAME family we'll evaluate for extraction, so its
labels could be self-flattering. This re-judges a HARD/ambiguous subset with an INDEPENDENT VLM
family (default Gemma-3-27B; Llama-4-Maverick also works) using the IDENTICAL prompt, then measures
agreement on the binary image-dependence call (image|both = needs the picture vs text|neither) +
Cohen's kappa, and prints the disagreements to eyeball. De-biases the gate (clog 260626).

"Hard" = where the gate is most likely wrong/ambiguous: claim_location both/neither, Qwen-vs-
raw_related disagreements (text-only gate said one thing, the VLM another), and borderline roles
(decorative / adds_detail / authenticity_subject). A few clear cases are mixed in as controls.
"""
from __future__ import annotations

import argparse
import glob
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from eval.scripts.image_audit import OUT as AUDIT
from eval.scripts.image_audit import _gate


def _dep(loc: str | None) -> bool:
    """Binary image-dependence: does the claim need the picture?"""
    return loc in ("image", "both")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="google/gemma-3-27b-it")
    ap.add_argument("--n", type=int, default=100, help="hard cases to cross-judge")
    ap.add_argument("--controls", type=int, default=20, help="clear cases mixed in as controls")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="eval/data/image_gate_crosscheck.parquet")
    args = ap.parse_args()

    a = pl.read_parquet(AUDIT)
    frames = []
    for f in glob.glob("eval/data/*_harvest.parquet"):
        df = pl.read_parquet(f)
        if "has_image" not in df.columns:
            continue
        frames.append(df.filter(pl.col("has_image") == True).select(  # noqa: E712
            "review_url", "claim_text", "raw_context", "image_paths", "image_est_tokens"))
    m = a.join(pl.concat(frames, how="diagonal_relaxed"), on="review_url", how="inner")

    hard = m.filter(
        pl.col("claim_location").is_in(["both", "neither"])
        | ((pl.col("raw_related") == True) & (pl.col("claim_location") == "image"))   # noqa: E712
        | ((pl.col("raw_related") == False) & (pl.col("claim_location") == "text"))   # noqa: E712
        | pl.col("image_role").is_in(["decorative", "adds_detail", "authenticity_subject"])
    ).head(args.n)
    ctrl = m.filter((pl.col("image_role") == "carries_claim")
                    & (pl.col("claim_location") == "image")).head(args.controls)
    sample = pl.concat([hard, ctrl], how="diagonal_relaxed").unique("review_url", keep="first").to_dicts()
    print(f"cross-judging {len(sample)} cases ({hard.height} hard + {ctrl.height} control) "
          f"with {args.judge}", flush=True)

    def work(r):
        try:
            v = _gate(r, model=args.judge)
            jl = v.get("claim_location")
        except Exception as e:
            jl = None
            print("  judge err:", str(e)[:80])
        return {"review_url": r["review_url"], "qwen_loc": r["claim_location"], "judge_loc": jl,
                "role": r["image_role"], "raw_related": r["raw_related"]}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(work, sample))

    paired = [(x["qwen_loc"], x["judge_loc"]) for x in res if x["qwen_loc"] and x["judge_loc"]]
    n = len(paired)
    agree = sum(1 for q, j in paired if _dep(q) == _dep(j))
    pq = sum(_dep(q) for q, _ in paired) / n
    pj = sum(_dep(j) for _, j in paired) / n
    pe = pq * pj + (1 - pq) * (1 - pj)
    po = agree / n
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"\nbinary image-dependence agreement: {agree}/{n} = {po*100:.0f}%  |  Cohen's kappa = {kappa:.2f}")

    dis = [x for x in res if x["qwen_loc"] and x["judge_loc"] and _dep(x["qwen_loc"]) != _dep(x["judge_loc"])]
    print(f"disagreements (eyeball these): {len(dis)}")
    jname = args.judge.split("/")[-1]
    for x in dis[:20]:
        print(f"  Qwen={x['qwen_loc']:<7} {jname}={x['judge_loc']:<7} role={x['role']:<18} "
              f"rr={x['raw_related']}  {x['review_url'][:55]}")

    pl.DataFrame(res).write_parquet(args.out)
    print(f"\nwrote {args.out} ({len(res)} rows)")


if __name__ == "__main__":
    main()
