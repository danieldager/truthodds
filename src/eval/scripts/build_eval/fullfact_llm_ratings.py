"""Rate the unmapped Full Fact fact-checks on the 1-5 veracity scale (fc-gold v2).

Full Fact writes free-text verdict sentences; the regex harmoniser leaves 2,296
post-2020 rows with `veracity` null. Two independent DeepInfra judges read
(claim, verdict) with `harmonize.VERACITY_LLM_SYSTEM` (+ a `not_a_verdict`
subtype); rows where they disagree on side, or where either judge says true /
failed to parse, go to a Sonnet adjudication pass run OUTSIDE this script (the
orchestrator drives Claude subagents over the emitted batch files).

  uv run python -m eval.scripts.build_eval.fullfact_llm_ratings --judge
  uv run python -m eval.scripts.build_eval.fullfact_llm_ratings --emit-sonnet
  # ... Sonnet writes sonnet_batches/batch_NN_sonnet.json ...
  uv run python -m eval.scripts.build_eval.fullfact_llm_ratings --merge

Output: eval/data/fullfact_llm/{judged.parquet, sonnet_batches/, ratings.parquet}
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.harmonize import (  # noqa: E402
    DEEPINFRA_API_KEY,
    DEEPINFRA_BASE_URL,
    VERACITY_LLM_SYSTEM,
)
from eval.prompt_hash import prompt_hash  # noqa: E402

SRC = Path("eval/data/fc_gold_v2.parquet")
OUTDIR = Path("eval/data/fullfact_llm")
JUDGED = OUTDIR / "judged.parquet"
BATCHDIR = OUTDIR / "sonnet_batches"
RATINGS = OUTDIR / "ratings.parquet"

PROMPT_HASH = prompt_hash(VERACITY_LLM_SYSTEM)

# gpt-oss reasons before answering; a small cap gets eaten by hidden reasoning and
# returns empty content, hence 2000 tokens + low effort.
JUDGES = {
    "gptoss": {"model": "openai/gpt-oss-120b", "max_tokens": 2000,
               "extra": {"reasoning_effort": "low"}},
    "qwen": {"model": "Qwen/Qwen3-235B-A22B-Instruct-2507", "max_tokens": 300, "extra": {}},
}

VALID_SUBTYPES = {"clear_true", "mostly_true", "mixed", "unprovable", "mostly_false",
                  "clear_false", "altered_media", "satire", "not_a_verdict"}


def side(v) -> str | None:
    if v is None:
        return None
    return "T" if v >= 4 else ("F" if v <= 2 else "M")


def judge(cfg: dict, claim: str, rating: str, timeout: int = 90) -> tuple:
    """(veracity|None, subtype, in_tok, out_tok). Unknown subtype -> parse_error/None."""
    body = {"model": cfg["model"], "temperature": 0, "max_tokens": cfg["max_tokens"],
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": VERACITY_LLM_SYSTEM},
                         {"role": "user",
                          "content": f"CLAIM:\n{claim}\n\nFact-checker VERDICT:\n{rating}"}],
            **cfg["extra"]}
    for attempt in range(3):
        try:
            r = requests.post(f"{DEEPINFRA_BASE_URL}/chat/completions",
                              headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
                              json=body, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            usage = d.get("usage") or {}
            in_tok = usage.get("prompt_tokens") or 0
            out_tok = usage.get("completion_tokens") or 0
            obj = json.loads(d["choices"][0]["message"]["content"] or "{}")
            sub = str(obj.get("subtype", "")).strip().lower()
            if sub not in VALID_SUBTYPES:
                return (None, "parse_error", in_tok, out_tok)
            if sub == "not_a_verdict":
                return (None, sub, in_tok, out_tok)
            try:
                ver = int(obj["veracity"])
            except (KeyError, TypeError, ValueError):
                return (None, "parse_error", in_tok, out_tok)
            return (max(1, min(5, ver)), sub, in_tok, out_tok)
        except Exception:
            if attempt == 2:
                return (None, "parse_error", 0, 0)
            time.sleep(2 * (attempt + 1))


def load_rows() -> pl.DataFrame:
    return (pl.read_parquet(SRC)
            .filter((pl.col("publisher_site") == "fullfact.org")
                    & pl.col("veracity").is_null()
                    & (pl.col("review_date") >= "2020"))
            .select("review_url", "claim_text",
                    pl.col("original_rating").alias("verdict"),
                    "review_date", "claim_date"))


def run_judge(workers: int, limit: int) -> None:
    df = load_rows()
    done = set()
    prior = None
    if JUDGED.exists():
        prior = pl.read_parquet(JUDGED)
        done = set(prior["review_url"].to_list())
        df = df.filter(~pl.col("review_url").is_in(list(done)))
    if limit:
        df = df.head(limit)
    rows = df.to_dicts()
    print(f"{len(rows)} rows to judge ({len(done)} already done), prompt {PROMPT_HASH}", flush=True)
    if not rows:
        return

    lock = threading.Lock()
    t0 = time.time()
    state = {"n": 0, "tok": {k: [0, 0] for k in JUDGES}}
    out: list[dict] = []

    def work(row: dict) -> dict:
        rec = dict(row)
        rec["prompt_hash"] = PROMPT_HASH
        for name, cfg in JUDGES.items():
            v, s, it, ot = judge(cfg, row["claim_text"], row["verdict"])
            rec[f"{name}_veracity"] = v
            rec[f"{name}_subtype"] = s
            with lock:
                state["tok"][name][0] += it
                state["tok"][name][1] += ot
        with lock:
            state["n"] += 1
            n = state["n"]
            if n % 100 == 0 or n == len(rows):
                el = time.time() - t0
                rate = n / el
                eta = (len(rows) - n) / rate / 60
                print(f"  {n}/{len(rows)}  {rate*60:.0f} rows/min  {el/60:.1f}m elapsed  "
                      f"ETA {eta:.1f}m", flush=True)
        return rec

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(work, rows):
                out.append(rec)
    finally:
        if out:
            new = pl.DataFrame(out)
            full = pl.concat([prior, new], how="diagonal") if prior is not None else new
            OUTDIR.mkdir(parents=True, exist_ok=True)
            full.write_parquet(JUDGED)
            print(f"\nwrote {JUDGED} ({full.height} rows) in {(time.time()-t0)/60:.1f} min",
                  flush=True)
            summarise_judged(full)
    for name, (it, ot) in state["tok"].items():
        print(f"tokens {name}: prompt {it:,} completion {ot:,} total {it+ot:,}")


def summarise_judged(df: pl.DataFrame) -> None:
    for name in JUDGES:
        print(f"\n{name} subtypes:")
        for r in df[f"{name}_subtype"].value_counts(sort=True).to_dicts():
            print(f"  {r[f'{name}_subtype']:<14} {r['count']}")
    a = [side(v) for v in df["gptoss_veracity"].to_list()]
    b = [side(v) for v in df["qwen_veracity"].to_list()]
    agree = sum(x == y for x, y in zip(a, b))
    print(f"\nside agreement: {agree}/{len(a)} ({agree/len(a):.1%})")
    both = [(x, y) for x, y in zip(a, b) if x != y]
    from collections import Counter
    for (x, y), c in Counter(both).most_common(10):
        print(f"  disagree {x} vs {y}: {c}")


def needs_sonnet(df: pl.DataFrame) -> pl.DataFrame:
    g, q = df["gptoss_veracity"].to_list(), df["qwen_veracity"].to_list()
    gs, qs = df["gptoss_subtype"].to_list(), df["qwen_subtype"].to_list()
    flag = [side(x) != side(y)
            or (x is not None and x >= 4) or (y is not None and y >= 4)
            or x is None or y is None or "parse_error" in (a, b)
            for x, y, a, b in zip(g, q, gs, qs)]
    return df.with_columns(pl.Series("_flag", flag))


def emit_sonnet(batch_size: int) -> None:
    df = needs_sonnet(pl.read_parquet(JUDGED))
    sel = df.filter(pl.col("_flag")).drop("_flag")
    BATCHDIR.mkdir(parents=True, exist_ok=True)
    for f in BATCHDIR.glob("batch_*.json"):
        if not f.name.endswith("_sonnet.json"):
            f.unlink()
    rows = sel.to_dicts()
    n_b = 0
    for bi in range(0, len(rows), batch_size):
        chunk = rows[bi:bi + batch_size]
        payload = [{"i": i, "url": r["review_url"], "claim": r["claim_text"],
                    "verdict": r["verdict"], "review_date": r["review_date"]}
                   for i, r in enumerate(chunk)]
        (BATCHDIR / f"batch_{n_b:02d}.json").write_text(json.dumps(payload, indent=1))
        n_b += 1
    print(f"{len(rows)} rows need Sonnet ({len(rows)/df.height:.1%} of {df.height}) "
          f"-> {n_b} batches in {BATCHDIR}")


def merge() -> None:
    df = needs_sonnet(pl.read_parquet(JUDGED))
    ruled: dict[str, tuple] = {}
    for bf in sorted(BATCHDIR.glob("batch_*_sonnet.json")):
        batch = json.loads((BATCHDIR / bf.name.replace("_sonnet", "")).read_text())
        by_i = {r["i"]: r for r in batch}
        for r in json.loads(bf.read_text()):
            src = by_i[r["i"]]
            sub = str(r.get("subtype", "")).strip().lower()
            ruled[src["url"]] = (r.get("veracity"), sub)
    print(f"{len(ruled)} Sonnet rulings from {len(list(BATCHDIR.glob('batch_*_sonnet.json')))} files")

    ver, sub, srcs = [], [], []
    for r in df.to_dicts():
        if r["review_url"] in ruled:
            v, s = ruled[r["review_url"]]
            ver.append(v)
            sub.append(s)
            srcs.append("sonnet")
        else:
            g, q = r["gptoss_veracity"], r["qwen_veracity"]
            ver.append(g if g == q else min(g, q))
            sub.append(r["gptoss_subtype"] if g == q else
                       (r["gptoss_subtype"] if g <= q else r["qwen_subtype"]))
            srcs.append("agreed")
    out = df.drop("_flag").with_columns(
        pl.Series("veracity", ver, dtype=pl.Int64),
        pl.Series("subtype", sub),
        pl.Series("source", srcs))
    out.write_parquet(RATINGS)
    print(f"wrote {RATINGS} ({out.height} rows)\n")
    for col in ("veracity", "subtype", "source"):
        print(f"{col}:")
        for r in out[col].value_counts(sort=True).to_dicts():
            print(f"  {r[col]}: {r['count']}")
    t = out.filter(pl.col("veracity") >= 4).height
    f = out.filter(pl.col("veracity") <= 2).height
    print(f"\ntrues (>=4): {t}   falses (<=2): {f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--emit-sonnet", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()
    if args.judge:
        run_judge(args.workers, args.limit)
    if args.emit_sonnet:
        emit_sonnet(args.batch_size)
    if args.merge:
        merge()


if __name__ == "__main__":
    main()
