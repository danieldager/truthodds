"""Harvest each outlet's OWN original posts (no retweets/replies) with media, via SocialData.

Reads a handles CSV (cell,domain,handle), pulls up to --target originals per handle from the
SocialData search endpoint (query `from:HANDLE -filter:retweets -filter:replies`), normalizes each
tweet to the canonical posts schema, and writes one parquet. Rate-governed (honors the
X-Ratelimit-Remaining header, ~120/min) and resumable (skips handles that already reached --target
in an existing output, dedups by post_id). Checkpoints after every handle so a long run is safe.

  cd src && uv run python eval/scripts/claim_sourcing/harvest_outlet_tweets.py \
      --handles eval/data/survey_claims/outlet_handles.csv --target 1000 \
      --out eval/data/survey_claims/outlet_tweets.parquet [--only FoxNews,MotherJones] [--max-handles N]

Canonical columns: post_id, cell, domain, handle, created_at, lang, text, is_quote, image_urls,
  video_urls, n_images, n_videos, url, like_count, retweet_count, reply_count, quote_count, view_count.
"""
import argparse, csv, json, sys, time, urllib.request, urllib.parse, urllib.error
from datetime import datetime
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parents[3]          # .../src
sys.path.insert(0, str(SRC))
from eval.textnorm import clean_text               # noqa: E402
ENV = SRC / ".env"
API = "https://api.socialdata.tools/twitter/search"
PAGE_CAP = 120                                       # per-handle pagination backstop (~20/page)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_key():
    for line in ENV.read_text().splitlines():
        if line.startswith("SOCIALDATA_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("no SOCIALDATA_API_KEY in src/.env")


class Governor:
    """Pace under ~120/min; back off hard when the window is nearly spent."""
    def __init__(self):
        self.remaining = None

    def note(self, headers):
        try:
            self.remaining = int(headers.get("x-ratelimit-remaining"))
        except (TypeError, ValueError):
            pass

    def wait(self):
        if self.remaining is not None and self.remaining <= 3:
            log(f"  rate budget low ({self.remaining} left) — sleeping 30s for the window to reset")
            time.sleep(30)
        else:
            time.sleep(0.6)                          # ~100/min steady pace, under the 120 cap


def api_get(key, query, cursor, gov, retries=5):
    params = {"query": query, "type": "Latest"}
    if cursor:
        params["cursor"] = cursor
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                gov.note(r.headers)
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                back = min(90, 5 * 2 ** attempt)
                log(f"  HTTP {e.code} — backoff {back}s (attempt {attempt+1}/{retries})")
                time.sleep(back)
                continue
            log(f"  HTTP {e.code}: {e.read().decode()[:200]}")
            return None
        except Exception as ex:                      # transient network
            back = min(30, 5 * 2 ** attempt)
            log(f"  {type(ex).__name__}: {ex} — retry in {back}s")
            time.sleep(back)
    return None


def media_of(t):
    """Reader-visible image URLs (photo + video poster frames) and best MP4 per video."""
    imgs, vids = [], []
    for m in (t.get("entities") or {}).get("media") or []:
        if m.get("media_url_https"):
            imgs.append(m["media_url_https"])        # photo, or a video/gif poster frame
        variants = [v for v in (m.get("video_info") or {}).get("variants", [])
                    if v.get("content_type") == "video/mp4" and v.get("url")]
        if variants:
            vids.append(max(variants, key=lambda v: v.get("bitrate", 0))["url"])
    return imgs, vids


def normalize(t, cell, domain, handle):
    sn = (t.get("user") or {}).get("screen_name") or handle
    imgs, vids = media_of(t)
    return {
        "post_id": t.get("id_str"),
        "cell": cell, "domain": domain, "handle": sn,
        "created_at": t.get("tweet_created_at"),
        "lang": t.get("lang"),
        # X HTML-escapes & < > in the API text field; decode at ingest so no raw
        # entity reaches the extractor, the read prompts, or any display.
        "text": clean_text(t.get("full_text") or t.get("text") or ""),
        "is_quote": bool(t.get("is_quote_status") or t.get("quoted_status")),
        "image_urls": imgs, "video_urls": vids,
        "n_images": len(imgs), "n_videos": len(vids),
        "url": f"https://x.com/{sn}/status/{t.get('id_str')}",
        "like_count": t.get("favorite_count"), "retweet_count": t.get("retweet_count"),
        "reply_count": t.get("reply_count"), "quote_count": t.get("quote_count"),
        "view_count": t.get("views_count"),
    }


def harvest_handle(key, cell, domain, handle, target, seen, gov, anchor_id=None, mode="deep",
                   time_ops=""):
    """Paginate a handle's originals until target reached / feed exhausted / page cap.

    anchor_id windows the query so resumes/top-ups don't re-fetch what we already hold:
      mode="deep"    -> `max_id:<anchor>`  continue OLDER than our oldest (initial deep-fill / resume)
      mode="refresh" -> `since_id:<anchor>` only NEWER than our newest (later top-ups)
    Snowflake IDs are time-ordered so ID windowing is exact. Post-id dedup (the `seen` set) is the
    hard guarantee against duplicates; the anchor just avoids paying to re-scan already-seen pages.
    time_ops: optional ` since_time:A until_time:B` suffix (see harvest_windowed).
    """
    base = f"from:{handle} -filter:retweets -filter:replies{time_ops}"
    if anchor_id and mode == "refresh":
        query = f"{base} since_id:{anchor_id}"
    elif anchor_id:
        query = f"{base} max_id:{anchor_id}"
    else:
        query = base
    rows, cursor, pages, start = [], None, 0, time.time()
    while len(rows) < target and pages < PAGE_CAP:
        gov.wait()
        d = api_get(key, query, cursor, gov)
        pages += 1
        if not d:
            break
        tws = d.get("tweets") or []
        new = 0
        for t in tws:
            pid = t.get("id_str")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            rows.append(normalize(t, cell, domain, handle))
            new += 1
        cursor = d.get("next_cursor")
        rate = len(rows) / max(0.1, (time.time() - start))
        log(f"  @{handle} {cell}: {len(rows)}/{target}  (+{new}, page {pages}, {rate:.1f} posts/s)")
        if not tws or not cursor:
            log(f"  @{handle}: feed exhausted at {len(rows)} originals")
            break
    return rows


def harvest_windowed(key, cell, domain, handle, target, n_windows, months, existing, seen, gov):
    """Split a handle's target across n equal time windows over the trailing `months`
    (newest first) via since_time/until_time, so high-volume feeds don't collapse into one
    news cycle. Resume-safe per window: rows already held inside a window reduce its quota,
    and the oldest held id in the window anchors max_id so a re-run never re-pays for
    already-scanned pages. Low-volume windows just yield fewer (logged, no redistribution)."""
    now = int(time.time())
    span = int(months * 30.44 * 86400)
    edges = [now - round(span * k / n_windows) for k in range(n_windows + 1)]  # newest -> oldest
    per = [target // n_windows] * n_windows
    per[0] += target - sum(per)
    ex_ts = None
    if existing is not None and len(existing):
        ex_ts = pd.to_datetime(existing["created_at"], errors="coerce", utc=True).astype("int64") // 10**9
    rows = []
    for w in range(n_windows):
        hi, lo = edges[w], edges[w + 1]
        have_w, anchor = 0, None
        if ex_ts is not None:
            in_w = existing[(ex_ts >= lo) & (ex_ts < hi)]
            have_w = len(in_w)
            if have_w:
                ids = pd.to_numeric(in_w["post_id"], errors="coerce").dropna().astype("int64")
                anchor = int(ids.min())
        want = per[w] - have_w
        span_str = f"{datetime.fromtimestamp(lo):%Y-%m-%d}→{datetime.fromtimestamp(hi):%Y-%m-%d}"
        if want <= 0:
            log(f"  @{handle} window {w+1}/{n_windows} [{span_str}]: have {have_w} — skip")
            continue
        got = harvest_handle(key, cell, domain, handle, want, seen, gov, anchor_id=anchor,
                             mode="deep", time_ops=f" since_time:{lo} until_time:{hi}")
        log(f"  @{handle} window {w+1}/{n_windows} [{span_str}]: +{len(got)} (wanted {want})")
        rows.extend(got)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handles", required=True, help="CSV with columns cell,domain,handle")
    ap.add_argument("--target", type=int, default=1000, help="originals to collect per handle")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--only", default="", help="comma-separated handles to restrict to")
    ap.add_argument("--max-handles", type=int, default=0, help="cap number of handles (0 = all)")
    ap.add_argument("--refresh", action="store_true",
                    help="top-up mode: fetch only tweets NEWER than what we already have (since_id)")
    ap.add_argument("--windows", type=int, default=1,
                    help="spread each handle's target across N time windows over --months "
                         "(since_time/until_time; avoids news-cycle concentration on high-volume feeds)")
    ap.add_argument("--months", type=int, default=12,
                    help="trailing months the windows span (only with --windows > 1)")
    args = ap.parse_args()
    if args.refresh and args.windows > 1:
        sys.exit("--refresh and --windows are mutually exclusive (refresh is id-anchored)")

    key = load_key()
    out = Path(args.out)
    only = {h.strip().lower() for h in args.only.split(",") if h.strip()}

    handles = []
    with open(args.handles) as f:
        for row in csv.DictReader(f):
            h = (row.get("handle") or "").strip()
            if not h or h.startswith("#"):
                continue
            if only and h.lower() not in only:
                continue
            handles.append((row["cell"].strip(), (row.get("domain") or "").strip(), h))
    if args.max_handles:
        handles = handles[:args.max_handles]

    mode = "refresh" if args.refresh else "deep"
    # resume: load existing rows, per-handle anchor id + count, global seen post_ids
    existing = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    seen = set(existing["post_id"]) if len(existing) else set()
    by_handle = {h: g for h, g in existing.groupby("handle")} if len(existing) else {}
    all_rows = existing.to_dict("records") if len(existing) else []
    log(f"handles: {len(handles)} | mode={mode} | target {args.target} | resuming from {len(all_rows)} existing rows")

    for i, (cell, domain, handle) in enumerate(handles, 1):
        ex = by_handle.get(handle)
        have = int(len(ex)) if ex is not None else 0
        anchor = None
        if have:
            ids = pd.to_numeric(ex["post_id"], errors="coerce").dropna().astype("int64")
            anchor = int(ids.max()) if mode == "refresh" else int(ids.min())
        if mode == "deep" and have >= args.target:
            log(f"[{i}/{len(handles)}] @{handle} already has {have} >= target — skip")
            continue
        want = args.target if mode == "refresh" else args.target - have
        log(f"[{i}/{len(handles)}] @{handle} ({cell}, {domain}) — have {have}, want {want}, mode {mode}"
            + (f", anchor {anchor}" if anchor else ""))
        if args.windows > 1:
            rows = harvest_windowed(key, cell, domain, handle, args.target, args.windows,
                                    args.months, ex, seen, gov0)
        else:
            rows = harvest_handle(key, cell, domain, handle, want, seen, gov0, anchor_id=anchor, mode=mode)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_parquet(out, index=False)      # checkpoint after each handle
        log(f"[{i}/{len(handles)}] @{handle}: +{len(rows)} → {len(all_rows)} total (checkpointed {out.name})")

    df = pd.DataFrame(all_rows)
    log(f"DONE. {len(df)} posts across {df['handle'].nunique() if len(df) else 0} handles → {out}")
    if len(df):
        img = int((df["n_images"] > 0).sum())
        log(f"  media: {img}/{len(df)} posts have >=1 image ({100*img/len(df):.0f}%)")
        log(f"  per cell:\n{df.groupby('cell').size().to_string()}")
    # Silent zero-yield guard: a requested handle with no rows is almost always a dead/renamed/
    # wrong handle (e.g. an outlet that rebranded) — surface it loudly instead of burying it.
    got = {str(g).lower() for g in (df["handle"] if len(df) else [])}
    zero = [h for _, _, h in handles if h.lower() not in got]
    if zero:
        log(f"  ⚠️  ZERO tweets for {len(zero)} requested handle(s) — likely dead/renamed/wrong: {', '.join(zero)}")


gov0 = Governor()
if __name__ == "__main__":
    main()
