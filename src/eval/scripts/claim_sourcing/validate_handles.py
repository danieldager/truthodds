"""Pre-flight: confirm each roster handle is a LIVE, ACTIVE, correctly-identified X account.

NewsGuard handles are stale (Dec-2024 export) — outlets rebrand / suspend / rename, and some were
wrong-outlet to begin with. ALWAYS run this on a roster CSV (cell,domain,handle) BEFORE committing a
harvest with harvest_outlet_tweets.py: it probes each handle via SocialData and reports the resolved
account identity so dead / dormant / wrong-outlet handles are caught before we spend on them.

  cd src && uv run python eval/scripts/claim_sourcing/validate_handles.py <roster.csv> [-o report.csv] [--stale-days 30]

Per handle -> resolved screen_name, account name, followers, latest-tweet date, and a status:
  LIVE          ok to harvest
  DEAD          from:HANDLE returns nothing (renamed/suspended/wrong handle)
  STALE         resolves but hasn't posted an original in --stale-days (likely dormant/renamed)
  NAME-MISMATCH resolves to a different screen_name than requested (verify it's the right outlet)
Exit code 1 if any handle is not LIVE.
"""
import argparse, csv, json, sys, time, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
KEY = next(l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
           if l.startswith("SOCIALDATA_API_KEY="))


def probe(handle):
    url = "https://api.socialdata.tools/twitter/search?" + urllib.parse.urlencode(
        {"query": f"from:{handle} -filter:retweets -filter:replies", "type": "Latest"})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"status": f"HTTP {e.code}"}
    except Exception as e:
        return {"status": f"ERR {type(e).__name__}"}
    tws = d.get("tweets") or []
    if not tws:
        return {"status": "DEAD"}
    u = tws[0].get("user") or {}
    return {"status": "LIVE", "screen_name": u.get("screen_name"), "name": u.get("name"),
            "followers": u.get("followers_count") or 0, "latest": tws[0].get("tweet_created_at")}


def days_old(iso):
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso.replace("Z", "+00:00"))).days
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roster", help="CSV with columns cell,domain,handle")
    ap.add_argument("-o", "--output", default="", help="write a validation report CSV")
    ap.add_argument("--stale-days", type=int, default=30)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.roster)) if (r.get("handle") or "").strip()]
    out, bad = [], 0
    print(f"{'requested':22} {'resolved':20} {'followers':>11} {'latest':>11}  status")
    print("-" * 78)
    for r in rows:
        h = r["handle"].strip()
        info = probe(h)
        st = info.get("status")
        if st == "LIVE":
            d = days_old(info.get("latest"))
            if info.get("screen_name") and info["screen_name"].lower() != h.lower():
                st = "NAME-MISMATCH"
            elif d is not None and d > args.stale_days:
                st = "STALE"
            note = (f"  {d}d since last post" if st == "STALE"
                    else f"  requested @{h}" if st == "NAME-MISMATCH" else "")
            print(f"@{h:21} @{(info.get('screen_name') or '?'):19} {info['followers']:>11,} "
                  f"{(info.get('latest') or '?')[:10]:>11}  {st}{note}")
        else:
            print(f"@{h:21} {'--':20} {'--':>11} {'--':>11}  {st}")
        if st != "LIVE":
            bad += 1
        out.append({"cell": r.get("cell"), "domain": r.get("domain"), "handle": h,
                    "resolved": info.get("screen_name"), "name": info.get("name"),
                    "followers": info.get("followers"), "latest": info.get("latest"), "status": st})
        time.sleep(0.7)

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print(f"\nwrote {args.output}")
    print(f"\n{len(out) - bad}/{len(out)} LIVE & ok" + (f" — {bad} NEED ATTENTION before harvest" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
