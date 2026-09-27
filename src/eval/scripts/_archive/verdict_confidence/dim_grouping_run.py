"""Dimension-grouping + reasoning A/B harness (scores from a FIXED analysis — no retrieval).

Substrate: `data/agreement.parquet` (60 stratified claims) — each row carries `claim_text`, the
fixed `deployment_analysis` (the synthesis output we score), binary `gold` (pass/flag), plus
`modality` and `stratum` for slicing. We score that analysis with DeepSeek V4 Flash under the
new 5-dim prompts (`dim_grouping_prompts.py`).

Two independent experiments, run as cells (config, reasoning):
  Exp 2 (grouping):  (all5, none), (split_2_3, none), (per_dim, none)   — same analysis, no reasoning.
  Exp 1 (reasoning): (all5, none) vs (all5, medium)                      — same analysis, all-5 call.

One row per (claim_id, config, reasoning): the 5 dim scores + summed token/latency cost + gold/
modality/stratum. Resumable via that triple; per-cell errors captured, never aborts.

Run (from src/):
  uv run python -m eval.scripts.verdict_confidence.dim_grouping_run -n 3            # smoke
  uv run python -m eval.scripts.verdict_confidence.dim_grouping_run                 # full (60)
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from config import VERIFICATION_MODEL
from eval.scripts.verdict_confidence.dim_grouping_prompts import (
    DIMS,
    GROUPINGS,
    build_messages,
    parse,
)
from pipeline.verify import _client, _record_usage, get_cache_stats, reset_cache_stats

SUBSTRATE = Path("eval/scripts/verdict_confidence/data/agreement.parquet")
OUT = Path("eval/scripts/verdict_confidence/data/dim_grouping.parquet")
MAX_TOKENS = 8000  # room for reasoning_content + the JSON answer

# (config, reasoning_effort). reasoning="none" explicitly disables thinking (not just default).
CELLS = [("all5", "none"), ("split_2_3", "none"), ("per_dim", "none"), ("all5", "medium")]


def _call(messages: list[dict], reasoning: str) -> tuple[str, dict]:
    """One scoring call. temperature=0 to keep the grouping comparison low-noise. json mode only
    when NOT thinking (it misroutes the answer into reasoning_content under thinking)."""
    thinking = reasoning not in (None, "none")
    kwargs = dict(model=VERIFICATION_MODEL, messages=messages, temperature=0.0,
                  max_tokens=MAX_TOKENS, reasoning_effort=reasoning)
    if not thinking:
        kwargs["response_format"] = {"type": "json_object"}
    r = _client.chat.completions.create(**kwargs)
    _record_usage(r.usage)
    u = r.usage
    det = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(det, "cached_tokens", None) or 0) if det else 0
    msg = r.choices[0].message
    content = msg.content or ""
    if not content and thinking:
        content = getattr(msg, "reasoning_content", None) or ""
    return content, {
        "prompt_tokens": u.prompt_tokens or 0, "cached_tokens": cached,
        "completion_tokens": u.completion_tokens or 0,
    }


def run_cell(row: dict, config: str, reasoning: str) -> dict:
    t0 = time.time()
    base = {"claim_id": row["claim_id"], "config": config, "reasoning": reasoning,
            "gold": row["gold"], "modality": row.get("modality"), "stratum": row.get("stratum")}
    try:
        scores = {d: None for d in DIMS}
        justifs: list[str] = []
        pt = ct = cot = 0
        for subset in GROUPINGS[config]:
            content, usage = _call(build_messages(row["claim_text"], row["deployment_analysis"], subset), reasoning)
            parsed = parse(content, subset)
            justifs.append(parsed["justification"])
            for d in subset:
                scores[d] = parsed[d]
            pt += usage["prompt_tokens"]; ct += usage["cached_tokens"]; cot += usage["completion_tokens"]
        return {**base, **scores, "n_calls": len(GROUPINGS[config]),
                "prompt_tokens": pt, "cached_tokens": ct, "completion_tokens": cot,
                "latency_s": round(time.time() - t0, 3), "justifications": justifs, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {**base, **{d: None for d in DIMS}, "n_calls": 0,
                "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
                "latency_s": round(time.time() - t0, 3), "justifications": [],
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--limit", type=int, default=None, help="cap claims (smoke)")
    ap.add_argument("-w", "--workers", type=int, default=4)
    args = ap.parse_args()

    df = pl.read_parquet(SUBSTRATE)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dicts()

    done: set[tuple] = set()
    existing: list[dict] = []
    if OUT.exists():
        ex = pl.read_parquet(OUT)
        existing = ex.to_dicts()
        done = {(r["claim_id"], r["config"], r["reasoning"]) for r in existing}
        print(f"existing: {len(done)} cells done")

    tasks = [(r, cfg, rea) for r in rows for (cfg, rea) in CELLS if (r["claim_id"], cfg, rea) not in done]
    print(f"substrate: {SUBSTRATE} ({len(rows)} claims) | cells: {CELLS}")
    print(f"to run: {len(tasks)} cells | model: {VERIFICATION_MODEL} | workers: {args.workers}")
    if not tasks:
        print("nothing to do.")
        return

    reset_cache_stats()
    t0 = time.time()
    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, r, cfg, rea): (r["claim_id"], cfg, rea) for (r, cfg, rea) in tasks}
        for j, fut in enumerate(as_completed(futs), 1):
            new_rows.append(fut.result())
            if j % 20 == 0 or j == len(tasks):
                print(f"  [{j}/{len(tasks)}] done")

    all_rows = existing + new_rows
    pl.DataFrame(all_rows).write_parquet(OUT)
    errs = sum(1 for r in new_rows if r["error"])
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT} ({len(all_rows)} rows; {errs} new errors)")

    cs = get_cache_stats()
    print(f"prompt cache: {cs['cached_tokens']}/{cs['prompt_tokens']} tokens cached "
          f"({cs['cached_pct']}%) over {cs['llm_calls']} calls")
    # quick per-cell readout: scale usage (%@2&4) on veracity + mean cost/latency
    res = pl.DataFrame([r for r in new_rows if not r["error"]])
    if res.height:
        for (cfg, rea), g in res.group_by(["config", "reasoning"], maintain_order=True):
            vera = [v for v in g["veracity"].to_list() if v is not None]
            at24 = 100 * sum(1 for v in vera if v in (2, 4)) / len(vera) if vera else 0
            print(f"  {cfg:10s}/{rea:7s}  n={g.height}  veracity %@2|4={at24:4.0f}%  "
                  f"avg out_tok={g['completion_tokens'].mean():.0f}  avg lat={g['latency_s'].mean():.1f}s")


if __name__ == "__main__":
    main()
