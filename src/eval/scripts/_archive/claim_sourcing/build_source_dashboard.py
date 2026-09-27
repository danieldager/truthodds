"""Running source-tracking dashboard: one row per outlet with metadata + harvest/extraction stats.

Reads the roster (source_roster.csv), the harvested corpus (outlet_tweets.parquet), and (optionally) a
claims parquet, fetches live follower counts from SocialData, and renders a self-contained HTML
dashboard. Re-run any time to refresh — this is the "keep track of the sources" doc.

  cd src && uv run python eval/scripts/claim_sourcing/build_source_dashboard.py \
      [--roster source_roster.csv] [--tweets outlet_tweets.parquet] [--claims <claims>.parquet] \
      [-o source_dashboard.html] [--no-followers]
"""
import argparse, csv, json, time, urllib.request, urllib.parse, html
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parents[3]
BASE = SRC / "eval/data/survey_claims"
KEY = next((l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
            if l.startswith("SOCIALDATA_API_KEY=")), None)


def followers(handle):
    url = "https://api.socialdata.tools/twitter/search?" + urllib.parse.urlencode(
        {"query": f"from:{handle} -filter:retweets -filter:replies", "type": "Latest"})
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(
            url, headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"}), timeout=45).read())
        u = (d.get("tweets") or [{}])[0].get("user") or {}
        return u.get("followers_count")
    except Exception:
        return None


def band(s):
    return "0-30" if s < 30 else "30-50" if s < 50 else "50-70" if s < 70 else "70-90" if s < 90 else "90-100"


BAND_COLOR = {"0-30": "#c0392b", "30-50": "#e07b39", "50-70": "#c99a2e", "70-90": "#5a9367", "90-100": "#2f7d4f"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", default=str(BASE / "source_roster.csv"))
    ap.add_argument("--tweets", default=str(BASE / "outlet_tweets.parquet"))
    ap.add_argument("--claims", default="", help="optional claims parquet for extraction counts")
    ap.add_argument("-o", "--output", default=str(BASE / "source_dashboard.html"))
    ap.add_argument("--no-followers", action="store_true")
    args = ap.parse_args()

    roster = list(csv.DictReader(open(args.roster)))
    tw = pd.read_parquet(args.tweets)
    tw["hk"] = tw["handle"].str.lower()
    ts = tw.groupby("hk").agg(n=("post_id", "size"), img=("n_images", lambda s: (s > 0).mean()),
                              dmax=("created_at", "max")).to_dict("index")
    cs, claim_posts = {}, 0
    if args.claims:
        cl = pd.read_parquet(args.claims); cl["hk"] = cl["handle"].str.lower()
        claim_posts = cl["post_id"].nunique()
        cs = cl.groupby("hk").agg(nc=("claim", "size"), posts=("post_id", "nunique")).to_dict("index")

    rows = []
    for r in roster:
        hk = r["handle"].lower()
        s = float(r["ng_score"])
        t = ts.get(hk, {})
        c = cs.get(hk, {})
        rows.append({
            "lean": r["lean"], "handle": r["handle"], "domain": r["domain"], "score": s,
            "band": band(s), "orient": r["orientation"], "desc": r["description"],
            "followers": (None if args.no_followers else followers(r["handle"])),
            "tweets": int(t.get("n", 0)), "imgpct": round(100 * t.get("img", 0)),
            "latest": (str(t.get("dmax", "") or "")[:10]),
            "claims": int(c.get("nc", 0)), "cposts": int(c.get("posts", 0)),
        })
        if not args.no_followers:
            time.sleep(0.4)

    tot_tw = sum(r["tweets"] for r in rows)
    tot_cl = sum(r["claims"] for r in rows)
    nL = sum(1 for r in rows if r["lean"] == "Left")
    esc = html.escape

    def cell(r):
        fol = f"{r['followers']:,}" if r["followers"] else "—"
        cpp = f"{r['claims']/r['cposts']:.1f}" if r["cposts"] else "—"
        cl = f"{r['claims']}" if r["claims"] else "—"
        h = r["handle"]
        return (f"<tr>"
                f"<td><span class='band' style='background:{BAND_COLOR[r['band']]}'>{r['score']:g}</span></td>"
                f"<td><span class='lean {r['lean'][0]}'>{r['lean'][0]}</span></td>"
                f"<td class='src'><a href='https://x.com/{esc(h)}' target='_blank' rel='noopener'>@{esc(h)}</a>"
                f"<div class='dom'>{esc(r['domain'])}</div></td>"
                f"<td class='num'>{fol}</td><td class='num'>{r['tweets'] or '—'}</td>"
                f"<td class='num'>{cl}</td><td class='num'>{cpp}</td>"
                f"<td class='num'>{r['imgpct']}%</td><td class='num'>{r['latest'] or '—'}</td>"
                f"<td class='desc'>{esc(r['desc'])}</td></tr>")

    def section(lean):
        rs = sorted([r for r in rows if r["lean"] == lean], key=lambda r: r["score"])
        return f"<tr class='grp'><td colspan='10'>{lean} ({len(rs)})</td></tr>" + "".join(cell(r) for r in rs)

    sub = (f"{len(rows)} sources · {nL} left / {len(rows)-nL} right · {tot_tw:,} tweets harvested · "
           f"{tot_cl:,} claims" + (f" (from a {claim_posts}-post sample)" if claim_posts else " (none yet)"))
    head = ("<tr><th>NG</th><th>lean</th><th>source</th><th>followers</th><th>tweets</th><th>claims</th>"
            "<th>cl/post</th><th>img</th><th>latest</th><th>tweeting practice</th></tr>")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Source roster</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1a1a19;--muted:#6b6b68;--card:#fff;--line:#e5e4e1;--accent:#3b5bdb;--L:#2f6feb;--R:#c0392b;--grp:#f0f0ee}}
@media(prefers-color-scheme:dark){{:root{{--bg:#17181a;--fg:#e8e8e6;--muted:#9a9a97;--card:#202225;--line:#33353a;--L:#5a8def;--R:#e05a4d;--grp:#26282c}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:22px 16px 70px}}h1{{font-size:20px;margin:0 0 3px}}.sub{{color:var(--muted);margin:0 0 16px}}
table{{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--muted);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--card)}}
tr.grp td{{background:var(--grp);font-weight:700;font-size:12px;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}}
.band{{display:inline-block;min-width:34px;text-align:center;color:#fff;font-weight:700;padding:1px 6px;border-radius:5px}}
.lean{{display:inline-block;width:18px;height:18px;line-height:18px;text-align:center;border-radius:50%;color:#fff;font-weight:700;font-size:11px}}
.lean.L{{background:var(--L)}}.lean.R{{background:var(--R)}}
.src a{{color:var(--accent);text-decoration:none;font-weight:600}}.dom{{color:var(--muted);font-size:12px}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}.desc{{color:var(--muted);font-size:12.5px;max-width:330px}}
</style></head><body><div class="wrap">
<h1>Source roster &amp; harvest tracker</h1><p class="sub">{esc(sub)}</p>
<table>{head}{section("Left")}{section("Right")}</table>
</div></body></html>"""
    Path(args.output).write_text(doc)
    print(f"wrote {args.output} — {len(rows)} sources, {tot_tw:,} tweets, {tot_cl:,} claims")


if __name__ == "__main__":
    main()
