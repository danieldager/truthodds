"""Smoke test: LLM extraction of claim-origination dates + context from FC articles.

The TRUE side of fc_gold is structurally undated (snopes: 3,521 TRUE rows, 0 claim
dates) and some claims need contextualization. Snopes prose usually dates and
attributes the claim, so a firewalled extraction over the ARTICLE text can recover
{claim_date, claimant, context} — never the verdict.

Validation design (2026-08-03, Daniel, spend < $0.50): two arms in one run —
  gold arm    rows where GFC metadata already has claim_date (politifact / AFP /
              factcheck.org): extracted date scores against the known one.
  target arm  snopes TRUE rows (no gold): measures yield; Daniel eyeballs the page.

  uv run python -m eval.scripts.build_eval.extract_claim_dates          # smoke 30+30

Writes eval/data/claim_date_smoke.json + audit page eval/data/claim_date_smoke.html.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.search import scrape                                   # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import llm              # noqa: E402

OUT = Path("eval/data/claim_date_smoke.json")
PAGE = Path("eval/data/claim_date_smoke.html")
MAX_CHARS = 7000

# Firewall: metadata about the claim only — the article's verdict never leaves.
EXTRACT_SYS = """\
You are given the text of a fact-checking article and the claim it examines.
Extract metadata about the CLAIM only — never the article's verdict or rating.

Reply with JSON only:
{"claim_date": "YYYY-MM-DD" | "YYYY-MM" | null,
 "claimant": "<who made or spread the claim>" | null,
 "claim_context": "<up to 40 words of retrieval context>" | null}

claim_date — the date the claim was MADE or first circulated: a speech, post,
broadcast, or publication date ("a post shared on …", "circulating since …",
"said in a speech on …"). NEVER the date of the events the claim describes —
a 2022 attack ad about a 2017 vote is dated 2022, not 2017. Use the earliest
specific date the article gives for the claim's UTTERANCE or circulation. Copy
the year exactly as the article states it; never infer a different year. The
article's own publication or update date is NOT the claim date. null when the
article never dates the claim.
claimant — the person or account the article attributes the claim to; null if
the article names nobody specific.
claim_context — everything that would help a search engine locate independent
evidence about this exact claim: the event or news story it concerns, the place,
the platform or medium where it circulated, and plain-name resolutions of any
vague referents in the claim ("the president" → the name; "my company" → the
company). Facts about the claim's setting ONLY — never the article's verdict,
rating, findings, or any language implying the claim is accurate or inaccurate.
"""

# Output-side leak guard: verdict-flavoured vocabulary has no business in claim
# metadata. Flags for the audit page; flagged rows need a human eye, not silent use.
import re
LEAK = re.compile(
    r"\b(false(?:ly)?|true|fake|hoax|debunk\w*|misleading|mislabel\w*|incorrect"
    r"|inaccurate|unfounded|baseless|no evidence|actually|in fact|satire|scam"
    r"|fabricat\w*|doctored|altered|miscaption\w*)\b", re.I)


def run_one(row):
    txt = scrape(row["review_url"]) or ""
    if len(txt) < 300:
        return {**row, "status": "scrape-fail"}
    try:
        obj, cost, _, _ = llm(
            [{"role": "system", "content": EXTRACT_SYS},
             {"role": "user", "content": f"CLAIM: {row['claim_text']}\n\n"
                                         f"ARTICLE:\n{txt[:MAX_CHARS]}"}],
            cache_key="datex2-" + row["review_url"])
    except Exception as e:
        return {**row, "status": f"llm-fail:{type(e).__name__}", "cost": 0.0}
    ctx = obj.get("claim_context") or ""
    return {**row, "status": "ok", "cost": cost,
            "x_date": obj.get("claim_date"), "x_claimant": obj.get("claimant"),
            "x_context": ctx,
            "leak_flag": bool(LEAK.search(f"{ctx} {obj.get('claimant') or ''}"))}


def main() -> None:
    df = pd.read_parquet("eval/data/fc_gold_v3.parquet", columns=[
        "publisher_site", "review_url", "claim_text", "claim_date", "review_date",
        "veracity", "claim_date_source"])

    gold = df[(df.claim_date.notna()) & (df.claim_date_source == "gfc")
              & (df.publisher_site.isin(["politifact.com", "factcheck.org",
                                         "factcheck.afp.com"]))]
    gold = pd.concat([g.sample(min(10, len(g)), random_state=7)
                      for _, g in gold.groupby("publisher_site")])
    snop = df[(df.publisher_site.str.contains("snopes", na=False))
              & (df.veracity >= 4)].sample(30, random_state=7)
    rows = ([{**r, "arm": "gold"} for r in gold.to_dict("records")]
            + [{**r, "arm": "snopes-true"} for r in snop.to_dict("records")])
    print(f"{len(rows)} articles ({len(gold)} gold-dated / {len(snop)} snopes TRUE)")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        recs = list(ex.map(run_one, rows))
    cost = sum(r.get("cost") or 0 for r in recs)
    print(f"done in {time.time()-t0:.0f}s, LLM cost ${cost:.4f}")

    # ---- score the gold arm -------------------------------------------------
    ok = [r for r in recs if r["status"] == "ok"]
    g = [r for r in ok if r["arm"] == "gold" and r.get("x_date")]
    hits = exact = within7 = 0
    for r in g:
        want = str(r["claim_date"])[:10]
        got = (r["x_date"] or "")[:10]
        if len(got) == 7:                      # month precision
            got_full, want_m = got, want[:7]
            r["delta"] = "month" if got_full == want_m else "month-miss"
            within7 += got_full == want_m
            continue
        try:
            dd = abs((pd.Timestamp(got) - pd.Timestamp(want)).days)
        except Exception:
            r["delta"] = "unparseable"
            continue
        r["delta"] = dd
        exact += dd == 0
        within7 += dd <= 7
    n_gold_ok = len([r for r in ok if r["arm"] == "gold"])
    print(f"gold arm: {n_gold_ok} scraped+extracted, {len(g)} emitted a date; "
          f"exact {exact}, within-7d {within7}")
    sy = [r for r in ok if r["arm"] == "snopes-true"]
    print(f"snopes arm: {len(sy)} ok, dates emitted {sum(bool(r.get('x_date')) for r in sy)}, "
          f"claimants {sum(bool(r.get('x_claimant')) for r in sy)}")
    fails = [r["status"] for r in recs if r["status"] != "ok"]
    if fails:
        print("failures:", pd.Series(fails).value_counts().to_dict())

    OUT.write_text(json.dumps(recs, indent=1, default=str))

    import html as H
    def row_html(r):
        want = str(r.get("claim_date") or "")[:10]
        delta = r.get("delta", "")
        badge = ("" if r["arm"] != "gold" else
                 f' · gold {want} · <b>Δ {delta}</b>')
        leak = ' · <b style="color:#b00">LEAK?</b>' if r.get("leak_flag") else ""
        return (f'<div class="c"><div class="m">{r["arm"]} · '
                f'{H.escape(r["publisher_site"])} · rev {str(r["review_date"])[:10]}'
                f'{badge}{leak} · <a href="{H.escape(r["review_url"])}">article</a></div>'
                f'<div>{H.escape(str(r["claim_text"]))[:220]}</div>'
                f'<div class="x">date <b>{H.escape(str(r.get("x_date")))}</b> · '
                f'claimant {H.escape(str(r.get("x_claimant")))} · '
                f'context {H.escape(str(r.get("x_context")))}</div></div>')
    PAGE.write_text(
        '<!doctype html><meta charset="utf-8"><title>claim-date smoke</title>'
        '<style>body{font:14px/1.4 -apple-system,sans-serif;max-width:900px;margin:2rem auto}'
        '.c{border-left:3px solid #888;padding:.3rem .7rem;margin:.6rem 0}'
        '.m{font-size:12px;color:#666}.x{color:#062}</style>'
        f'<h1>Claim-date extraction smoke — ${cost:.4f}</h1>'
        + "\n".join(row_html(r) for r in ok))
    print(f"wrote {OUT} and {PAGE}")


if __name__ == "__main__":
    main()
