"""Screen every E1 claim for ill-posedness BEFORE the urn run (Daniel 2026-08-04).

The smoke audit read 25 of 4,157 rows and found one escaped claim class
(caption-media, where the media reference lives in the extracted context rather
than the claim text). Sweeping that ONE class draw-wide turned up 67 rows, 26 of
them bad — i.e. a 25-row sample understates class prevalence by ~40x, so
"we sampled and it looked fine" is not evidence the draw is clean.

This screens ALL runnable rows on the claim as the system will see it (resolved
claim + usable context + date), with no access to the veracity label or the
fact-check verdict, so it cannot rationalise from the answer.

Taxonomy is exactly the failure classes observed in the audit — no speculative
categories:
  ok                     self-contained, text-checkable world-proposition
  media_locus            truth is a property of an image/video (pixel proposition)
  misattributed_media    the event is real but the judged axis is the media's
                         provenance — a world-recast verifies TRUE against a gold
                         FALSE (the tsunami criterion, 2026-08-04)
  unresolved_referent    demonstrative/definite reference never named anywhere
  fragment               not a proposition: subject-less, appeal, question, or
                         normative recommendation
  artifact_unreproduced  judged text/quote/artifact never reproduced, so there is
                         nothing to check (the McClain class)
  compound               two or more independent load-bearing propositions (ONE
                         query per claim can only chase one of them)
Plus an independent context_leak check: the triage kept a context that named the
verdict mechanism ("AI-generated videos"), so leak detection gets a second pass
on the text that actually reaches the query step.

  uv run python -m eval.scripts.build_eval.screen_claims --smoke
  uv run python -m eval.scripts.build_eval.screen_claims

Writes eval/data/claim_screen_e1.parquet + an HTML page of every non-ok verdict.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import threading
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.evidence_urn_run import CAL, CTX, GATE, RES, exclusion_for, llm

OUT = Path("eval/data/claim_screen_e1.parquet")
PAGE = Path("eval/data/claim_screen_e1.html")

CLASSES = ("ok", "media_locus", "misattributed_media", "unresolved_referent",
           "fragment", "artifact_unreproduced", "compound")

SCREEN_SYS = """You are screening claims for an evidence-retrieval measurement. Each claim will be checked by searching the open web for text documents and judging what they say. Your job is to decide whether the claim, as written, is a well-posed target for that process — NOT whether it is true.

You get the claim, sometimes a short neutral description of how it circulated, and sometimes its date. Judge the claim as a proposition.

Reply with JSON only:
{"verdict": "<class>", "context_leak": <true|false>, "why": "<one short sentence>"}

verdict — ONE class:
  "ok"                     a self-contained proposition about the world that a text document could confirm or contradict.
  "media_locus"            its truth is a property of a specific image or video — whether the footage is authentic, what it depicts, whether it was edited. Text about the world cannot settle it.
  "misattributed_media"    the underlying event is real and would be confirmed by text, but the claim's actual point of dispute is whether a particular piece of media really shows it. Checking the event would return TRUE while the disputed proposition is FALSE.
  "unresolved_referent"    it points at something ("this mosque", "the church", "a pictured couple") that neither the claim nor the description ever names, so there is no determinate subject to check.
  "fragment"               not a proposition: a subject-less predicate, an appeal or instruction, a question, or a recommendation about what someone should do.
  "artifact_unreproduced"  it asserts something about a specific text, quote, sign, or document whose content is never reproduced, so there is nothing to compare against.
  "compound"               it carries two or more independent load-bearing propositions that could come apart, so confirming one leaves the claim's truth unsettled.

Order of precedence when more than one fits: media_locus, then misattributed_media, then unresolved_referent, then artifact_unreproduced, then fragment, then compound. Prefer "ok" whenever a determinate world-proposition survives; a claim that merely mentions a video or photo in passing, while asserting something about the world, is "ok".

context_leak — true only if the description states or strongly implies a verdict on the claim: that it is false, fabricated, AI-generated, doctored, misattributed, a hoax, or that the real facts differ. Neutrally reporting that a claim circulated, who made it, or where it was posted is NOT a leak."""


def user_block(row) -> str:
    parts = [f"CLAIM: {row['claim_for_screen']}"]
    if row["ctx_for_screen"]:
        parts.append(f"DESCRIPTION: {row['ctx_for_screen']}")
    if row["date_for_screen"]:
        parts.append(f"DATE: {row['date_for_screen']}")
    return "\n".join(parts)


def screen(row) -> tuple[dict, float]:
    for _ in range(3):
        try:
            obj, cost, _, _ = llm(
                [{"role": "system", "content": SCREEN_SYS},
                 {"role": "user", "content": user_block(row)}],
                cache_key=f"claimscreen-{row['review_url']}", max_tokens=120)
            v = obj.get("verdict")
            if v in CLASSES:
                return {"verdict": v, "context_leak": bool(obj.get("context_leak")),
                        "why": (obj.get("why") or "")[:200]}, cost
        except Exception:
            time.sleep(3)
    return {"verdict": "screen-failed", "context_leak": False, "why": ""}, 0.0


def build_page(df: pl.DataFrame) -> str:
    bad = df.filter(pl.col("verdict") != "ok").sort(["verdict", "review_url"])
    leaks = df.filter(pl.col("context_leak"))
    counts = dict(zip(*df["verdict"].value_counts().to_dict().values())) if df.height else {}

    def esc(s):
        return html.escape(str(s or ""))

    def block(rows, cls=""):
        return "\n".join(
            f'<div class="case {cls}"><div class="meta">{esc(r["verdict"])}'
            f'{" · CONTEXT-LEAK" if r["context_leak"] else ""} · '
            f'<a href="{esc(r["review_url"])}">review</a></div>'
            f'<div class="claim">{esc(r["claim_for_screen"])}</div>'
            f'<div class="ctx">{esc(r["ctx_for_screen"])}</div>'
            f'<div class="why">{esc(r["why"])}</div></div>'
            for r in rows.iter_rows(named=True))

    return f"""<!doctype html><meta charset="utf-8">
<title>E1 claim screen</title>
<style>
body{{font:14px/1.5 -apple-system,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111}}
h2{{border-bottom:2px solid #111;padding-bottom:.2rem;margin-top:2rem}}
.case{{border-left:3px solid #b00;padding:.4rem .8rem;margin:.7rem 0}}
.case.leak{{border-left-color:#c60}}
.meta{{font-size:12px;color:#666}} .claim{{color:#802;margin:.2rem 0}}
.ctx{{color:#555;font-size:13px}} .why{{color:#062;font-size:13px;margin-top:.2rem}}
code{{background:#eee;padding:0 .3em}}
</style>
<h1>E1 claim screen — every runnable claim</h1>
<p>Screened on the claim as the runner sees it (resolved claim + usable context + date),
blind to veracity and to the fact-check verdict. Counts: <code>{counts}</code></p>
<h2>Non-ok verdicts ({bad.height})</h2>
{block(bad)}
<h2>Context leaks ({leaks.height})</h2>
{block(leaks, "leak")}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="50 claims, cost probe")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ctx = pl.read_parquet(CTX)
    res = pl.read_parquet(RES).select(["review_url", "resolution_status", "claim_resolved"])
    df = (pl.read_parquet(CAL).join(ctx, on="review_url", how="inner")
          .join(res, on="review_url", how="inner").sort("review_url"))
    gate = {d["review_url"]: d for d in json.loads(GATE.read_text())["decisions"]}

    rows = []
    for r in df.iter_rows(named=True):
        if exclusion_for(r, gate):          # already excluded; nothing to screen
            continue
        rd = (r.get("review_date") or "")[:10]
        date = ""
        for cand in ((r.get("claim_date") or "")[:10], (r.get("x_date") or "").strip()):
            if cand and not (rd and cand[:10] >= rd):
                date = cand
                break
        rows.append({
            "review_url": r["review_url"],
            "claim_for_screen": r.get("claim_resolved") or r["claim_text"],
            "ctx_for_screen": (r["x_context"] or "").strip() if r["context_ok"] else "",
            "date_for_screen": date,
        })
    if args.smoke:
        rows = rows[::max(1, len(rows) // 50)][:50]
    print(f"screening {len(rows)} claims | {args.workers} workers", flush=True)

    lock, state = threading.Lock(), {"n": 0, "cost": 0.0}
    out, t0 = [], time.time()

    def work(row):
        v, cost = screen(row)
        with lock:
            state["n"] += 1
            state["cost"] += cost
            out.append({**row, **v})
            if state["n"] % 250 == 0 or state["n"] == len(rows):
                el = (time.time() - t0) / 60
                print(f"  {state['n']}/{len(rows)} | ${state['cost']:.3f} | {el:.1f}m | "
                      f"proj ${state['cost']/state['n']*len(rows):.2f}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, rows))

    res_df = pl.DataFrame(out)
    if not args.smoke:
        res_df.write_parquet(OUT)
        PAGE.write_text(build_page(res_df))
        print(f"\nwrote {OUT} and {PAGE}")
    print(res_df["verdict"].value_counts().sort("count", descending=True))
    print(f"context leaks: {res_df['context_leak'].sum()}")
    print(f"total ${state['cost']:.3f} | {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
