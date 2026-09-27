"""Phase-2 vetting probe: resolve candidate handles AND capture a small tweet sample per handle.

Same SocialData search endpoint as validate_handles.py (1-2 requests/handle — frugal), but keeps
the whole ~20-tweet page instead of only the newest tweet, so one cheap pass yields:
  - liveness/identity: resolved screen_name, account name, followers, latest-post date
  - format evidence: per-handle tweet samples + descriptive stats (text length, image/video
    share, link-only share) for the manual format vetting in docs/survey_5k_roadmap.md Phase 2

  cd src && uv run python eval/scripts/claim_sourcing/sample_handle_tweets.py \
      eval/data/survey_claims/roster_expansion_handles.csv \
      --out-samples eval/data/survey_claims/handle_format_samples.parquet \
      --out-report eval/data/survey_claims/handle_validation_report.csv \
      --out-doc eval/data/survey_claims/handle_format_review.md [--pages 1]

Input CSV needs: band,domain,lean,register,handle. Only rows with a handle are probed.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.textnorm import clean_text               # noqa: E402

KEY = next(l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
           if l.startswith("SOCIALDATA_API_KEY="))
API = "https://api.socialdata.tools/twitter/search"

_URL_RE = re.compile(r"https?://\S+")


def fetch_page(handle: str, cursor: str | None):
    params = {"query": f"from:{handle} -filter:retweets -filter:replies", "type": "Latest"}
    if cursor:
        params["cursor"] = cursor
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}",
                                 headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.loads(r.read().decode())
        return d.get("tweets") or [], d.get("next_cursor")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}", None
    except Exception as e:
        return f"ERR {type(e).__name__}", None


def days_old(iso):
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso.replace("Z", "+00:00"))).days
    except Exception:
        return None


def strip_urls(text: str) -> str:
    return _URL_RE.sub("", text or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("handles_csv")
    ap.add_argument("--pages", type=int, default=1, help="search pages per handle (~20 tweets each)")
    ap.add_argument("--out-samples", default="", help="parquet of all sampled tweets")
    ap.add_argument("--out-report", default="", help="CSV validation report")
    ap.add_argument("--out-doc", default="", help="markdown format-review doc")
    ap.add_argument("--stale-days", type=int, default=30)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.handles_csv)) if (r.get("handle") or "").strip()]
    report, samples = [], []
    print(f"{'domain':28} {'resolved':20} {'followers':>11} {'latest':>11}  status")
    print("-" * 86)
    for r in rows:
        h = r["handle"].strip()
        tweets, cursor = fetch_page(h, None)
        if isinstance(tweets, str):                      # error string
            status, user, page = tweets, {}, []
        elif not tweets:
            status, user, page = "DEAD", {}, []
        else:
            page = list(tweets)
            for _ in range(args.pages - 1):
                if not cursor:
                    break
                time.sleep(0.7)
                more, cursor = fetch_page(h, cursor)
                if isinstance(more, str) or not more:
                    break
                page += more
            user = page[0].get("user") or {}
            status = "LIVE"
            if (user.get("screen_name") or "").lower() != h.lower():
                status = "NAME-MISMATCH"
            elif (d := days_old(page[0].get("tweet_created_at"))) is not None and d > args.stale_days:
                status = f"STALE({d}d)"

        stats = {}
        if page:
            texts = [strip_urls(clean_text(t.get("full_text") or t.get("text") or "")) for t in page]
            n = len(page)
            stats = {
                "n_sampled": n,
                "median_len": int(pd.Series([len(t) for t in texts]).median()),
                "pct_link_only": round(sum(len(t) < 25 for t in texts) / n, 2),
                "pct_media": round(sum(bool((t.get("entities") or {}).get("media")) for t in page) / n, 2),
            }
            for t, raw in zip(texts, page):
                samples.append({**{k: r.get(k) for k in ("band", "domain", "lean", "register")},
                                "handle": h, "text_nourl": t,
                                "text": clean_text(raw.get("full_text") or raw.get("text") or ""),
                                "created_at": raw.get("tweet_created_at"),
                                "post_id": str(raw.get("id_str") or raw.get("id"))})

        report.append({**{k: r.get(k) for k in ("band", "domain", "lean", "register")},
                       "handle": h, "resolved": user.get("screen_name"),
                       "account_name": user.get("name"), "followers": user.get("followers_count"),
                       "latest": (page[0].get("tweet_created_at") if page else None),
                       "status": status, **stats})
        f = user.get("followers_count")
        print(f"{r['domain']:28} @{(user.get('screen_name') or '--'):19} "
              f"{(f'{f:,}' if f else '--'):>11} {((page[0].get('tweet_created_at') or '?')[:10] if page else '--'):>11}  {status}")
        time.sleep(0.7)

    if args.out_report:
        pd.DataFrame(report).to_csv(args.out_report, index=False)
        print(f"\nreport  -> {args.out_report}")
    if args.out_samples and samples:
        pd.DataFrame(samples).to_parquet(args.out_samples, index=False)
        print(f"samples -> {args.out_samples} ({len(samples)} tweets)")
    if args.out_doc:
        lines = ["# Handle format review — Phase 2 vetting samples", ""]
        for rep in report:
            lines.append(f"## {rep['domain']} — @{rep['resolved'] or rep['handle']} "
                         f"[{rep['band']} {rep['lean']}/{rep['register']}] — {rep['status']}, "
                         f"{rep['followers'] or '?'} followers, median {rep.get('median_len', '?')} chars, "
                         f"media {rep.get('pct_media', '?')}, link-only {rep.get('pct_link_only', '?')}")
            for s in [s for s in samples if s["handle"] == rep["handle"]][:8]:
                lines.append(f"- {s['text_nourl'][:220]}")
            lines.append("")
        Path(args.out_doc).write_text("\n".join(lines))
        print(f"doc     -> {args.out_doc}")

    bad = sum(1 for x in report if x["status"] != "LIVE")
    print(f"\n{len(report) - bad}/{len(report)} LIVE" + (f" — {bad} need attention" if bad else ""))


if __name__ == "__main__":
    main()
