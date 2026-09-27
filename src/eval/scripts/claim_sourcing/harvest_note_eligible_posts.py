"""Harvest the X Community Notes note-eligible feed as a survey post-supply channel.

The AI Note Writer API serves posts that readers have REQUESTED a note on. That is a
free, daily-refreshed stream of posts someone already suspects of being misleading -
the supply the survey pool has been short of. This script drains today's feed and
normalizes it onto the canonical posts schema (same column names as
`harvest_outlet_tweets.py` wherever the fields coincide), keeping the raw JSON beside it.

  cd src && uv run python eval/scripts/claim_sourcing/harvest_note_eligible_posts.py
  cd src && uv run python eval/scripts/claim_sourcing/harvest_note_eligible_posts.py --max-pages 5

Writes under eval/data/community_notes/:
  eligible_posts.parquet   the accumulating canonical table, deduped on post_id
  feed_raw/YYYY-MM-DD.jsonl  one line per API response page, body verbatim + fetch metadata
                           (sibling raw/ holds the unrelated public Community Notes dump)
  state.json               last cursor + last page's rate-limit headers, for resume
  manifest.json            endpoint, window, pulled_at, counts, rate-limit state

Idempotent and resumable: post_id is the dedup key against the existing parquet, the
cursor is checkpointed after every page, and `--resume` restarts from it. Safe to run
daily; each run appends only posts not already held.

Feed semantics (measured, not documented): there is no since_id / start_time. The feed
is a cursor-paginated ranked list of CURRENTLY eligible posts, drained with
`pagination_token` until no next_token. It is not reach-ordered (Spearman feed_rank vs
impressions 0.118) and it is a rolling ~3-day window, not 24h: on the 2026-09-14 pull
72% of posts were <2 days old and 98.3% <7 days, with a thin tail back to 2021.
"Today" is therefore "whatever the feed holds now", deduped on post_id across daily
runs and filtered client-side with --since-days. Rate budget 500 requests / 15 min, free.
test_mode=false is 403 until the note writer earns admission.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.textnorm import clean_text                                      # noqa: E402
from eval.scripts.claim_sourcing.community_notes_api import CNClient, ELIGIBLE_POSTS  # noqa: E402

OUT = SRC / "eval" / "data" / "community_notes"
PARQUET = OUT / "eligible_posts.parquet"
RAW = OUT / "feed_raw"
STATE = OUT / "state.json"
MANIFEST = OUT / "manifest.json"

SNOWFLAKE_EPOCH_MS = 1288834974657     # X's snowflake epoch


def snowflake_time(sid):
    """Note-request suggestion ids are snowflakes, so they carry the request time the API
    does not otherwise expose. Validated on the 2026-09-14 pull: 597/597 posts put the
    earliest request AFTER the post's created_at (median +6.1 h), none in the future."""
    try:
        return datetime.fromtimestamp(((int(sid) >> 22) + SNOWFLAKE_EPOCH_MS) / 1000,
                                      timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        return None


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def media_of(post, media_by_key):
    """(image_urls, video_urls) for one post, from its media_keys and includes.media."""
    imgs, vids, kinds = [], [], []
    keys = (post.get("attachments") or {}).get("media_keys") or []
    keys += [m.get("media_key") for m in (post.get("media_metadata") or []) if m.get("media_key")]
    for k in dict.fromkeys(keys):
        m = media_by_key.get(k)
        if not m:
            continue
        kinds.append(m.get("type"))
        if m.get("type") == "photo":
            if m.get("url"):
                imgs.append(m["url"])
        else:                                        # video / animated_gif
            best = max((v for v in m.get("variants", []) if v.get("bit_rate") is not None),
                       key=lambda v: v["bit_rate"], default=None)
            url = (best or {}).get("url") or m.get("preview_image_url")
            if url:
                vids.append(url)
    return imgs, vids, sorted(set(k for k in kinds if k))


def normalize(post, users_by_id, media_by_key, first_seen_at, feed_rank):
    """One feed post -> one canonical row. Column names match outlet_tweets.parquet
    wherever the two schemas coincide; everything else is CN- or author-specific."""
    author = users_by_id.get(post.get("author_id"), {})
    apm = author.get("public_metrics") or {}
    pm = post.get("public_metrics") or {}
    refs = post.get("referenced_tweets") or []
    ref_types = [r.get("type") for r in refs]
    imgs, vids, kinds = media_of(post, media_by_key)
    urls = [u.get("expanded_url") or u.get("url")
            for u in ((post.get("entities") or {}).get("urls") or [])]
    note_urls = [u.get("expanded_url") or u.get("url")
                 for u in (((post.get("note_tweet") or {}).get("entities") or {}).get("urls") or [])]
    sugg = post.get("note_request_suggestions") or []
    req_times = sorted(t for t in (snowflake_time(x.get("suggestion_id")) for x in sugg) if t)
    links = post.get("suggested_source_links_with_counts") or []
    handle = author.get("username")
    long_text = ((post.get("note_tweet") or {}).get("text")) or ""

    return {
        # --- canonical posts schema (same names as outlet_tweets.parquet) ---
        "post_id": post.get("id"),
        "handle": handle,
        "created_at": post.get("created_at"),
        "lang": post.get("lang"),
        "text": clean_text(long_text or post.get("text") or ""),
        "is_quote": "quoted" in ref_types,
        "image_urls": imgs, "video_urls": vids,
        "n_images": len(imgs), "n_videos": len(vids),
        "url": f"https://x.com/{handle or 'i'}/status/{post.get('id')}",
        "like_count": pm.get("like_count"), "retweet_count": pm.get("retweet_count"),
        "reply_count": pm.get("reply_count"), "quote_count": pm.get("quote_count"),
        "view_count": pm.get("impression_count"),
        # --- reach, beyond the canonical five ---
        "bookmark_count": pm.get("bookmark_count"),
        # --- post form ---
        "conversation_id": post.get("conversation_id"),
        "is_reply": "replied_to" in ref_types or bool(post.get("in_reply_to_user_id")),
        "is_repost": "retweeted" in ref_types,
        "referenced_types": ref_types,
        "referenced_ids": [r.get("id") for r in refs],
        "is_thread_root": post.get("conversation_id") == post.get("id"),
        "is_longform": bool(long_text),
        "text_short": clean_text(post.get("text") or ""),
        "possibly_sensitive": post.get("possibly_sensitive"),
        "media_types": kinds,
        "has_media": bool(kinds),
        "urls": [u for u in (urls + note_urls) if u],
        # --- author, for political-lean inference ---
        "author_id": post.get("author_id"),
        "author_name": author.get("name"),
        "author_description": author.get("description"),
        "author_location": author.get("location"),
        "author_url": author.get("url"),
        "author_created_at": author.get("created_at"),
        "author_verified": author.get("verified"),
        "author_verified_type": author.get("verified_type"),
        "author_protected": author.get("protected"),
        "author_followers": apm.get("followers_count"),
        "author_following": apm.get("following_count"),
        "author_tweet_count": apm.get("tweet_count"),
        "author_listed": apm.get("listed_count"),
        "author_media_count": apm.get("media_count"),
        "author_profile_image_url": author.get("profile_image_url"),
        "author_desc_urls": [u.get("expanded_url") for u in
                             (((author.get("entities") or {}).get("description") or {}).get("urls") or [])],
        # --- note-request metadata ---
        "n_note_requests": len(sugg),
        "first_note_request_at": req_times[0] if req_times else None,
        "last_note_request_at": req_times[-1] if req_times else None,
        "note_request_suggestions": json.dumps(sugg, ensure_ascii=False) if sugg else None,
        "n_suggested_source_links": len(links),
        "suggested_source_link_count": sum(int(l.get("count") or 0) for l in links),
        "suggested_source_links": json.dumps(links, ensure_ascii=False) if links else None,
        # --- provenance ---
        "first_seen_at": first_seen_at,
        "feed_rank": feed_rank,
        "raw_json": json.dumps(post, ensure_ascii=False),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-pages", type=int, default=40, help="page backstop (default 40)")
    ap.add_argument("--max-results", type=int, default=100, help="posts per page (default 100)")
    ap.add_argument("--lang", default="all", help="feed_lang for post_selection (default all)")
    ap.add_argument("--test-mode", default="true", choices=("true", "false"))
    ap.add_argument("--since-days", type=float, default=None,
                    help="keep only posts created within this many days (default: keep all)")
    ap.add_argument("--resume", action="store_true", help="restart from the saved cursor")
    ap.add_argument("--min-remaining", type=int, default=25,
                    help="stop when x-rate-limit-remaining falls to this (default 25)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    test_mode = args.test_mode == "true"

    existing = pd.read_parquet(PARQUET) if PARQUET.exists() else pd.DataFrame()
    seen = set(existing["post_id"]) if len(existing) else set()
    log(f"cn harvest: {len(seen)} posts already held, lang={args.lang}, test_mode={test_mode}")

    cursor = None
    if args.resume and STATE.exists():
        cursor = json.loads(STATE.read_text()).get("cursor")
        log(f"resuming from saved cursor ({'set' if cursor else 'none'})")

    client = CNClient()
    raw_path = RAW / f"{started:%Y-%m-%d}.jsonl"
    rows, pages, api_calls, dupes, t0 = [], 0, 0, 0, time.time()
    cutoff = (started - timedelta(days=args.since_days)) if args.since_days else None

    while pages < args.max_pages:
        fetched_at = datetime.now(timezone.utc)
        try:
            resp = client.eligible_posts(test_mode=test_mode, feed_lang=args.lang,
                                         max_results=args.max_results, pagination_token=cursor)
        except RuntimeError as e:
            log(f"STOP: {e}")
            break
        api_calls += 1
        pages += 1
        with raw_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "fetched_at": fetched_at.isoformat().replace("+00:00", "Z"),
                "endpoint": ELIGIBLE_POSTS, "test_mode": test_mode, "feed_lang": args.lang,
                "page_index": pages - 1, "pagination_token": cursor,
                "rate": client.rate_state(), "response": resp,
            }, ensure_ascii=False) + "\n")

        inc = resp.get("includes", {})
        users_by_id = {u["id"]: u for u in inc.get("users", [])}
        media_by_key = {m["media_key"]: m for m in inc.get("media", [])}
        new = 0
        for i, post in enumerate(resp.get("data", [])):
            pid = post.get("id")
            if not pid or pid in seen:
                dupes += 1
                continue
            if cutoff and post.get("created_at") and \
                    datetime.fromisoformat(post["created_at"].replace("Z", "+00:00")) < cutoff:
                continue
            seen.add(pid)
            rows.append(normalize(post, users_by_id, media_by_key,
                                  fetched_at.isoformat().replace("+00:00", "Z"),
                                  (pages - 1) * args.max_results + i))
            new += 1

        cursor = resp.get("meta", {}).get("next_token")
        rate = client.rate_state()
        remaining = rate.get("x-rate-limit-remaining")
        STATE.write_text(json.dumps({"cursor": cursor, "rate": rate,
                                     "updated_at": fetched_at.isoformat()}, indent=1))
        if rows:                                      # checkpoint every page
            pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_parquet(PARQUET, index=False)
        el = time.time() - t0
        log(f"  page {pages}: {resp.get('meta',{}).get('result_count',0)} returned, +{new} new "
            f"(total {len(rows)}, {dupes} dup) | {len(rows)/max(el,0.1):.1f} posts/s, "
            f"{api_calls/max(el/60,0.01):.1f} req/min | rate-limit-remaining {remaining}")
        if not cursor:
            log("  feed exhausted (no next_token)")
            break
        if remaining is not None and int(remaining) <= args.min_remaining:
            log(f"  STOP: rate budget down to {remaining}")
            break
        time.sleep(0.5)

    df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    if len(df):
        df = df.drop_duplicates(subset="post_id", keep="first")
        df.to_parquet(PARQUET, index=False)
    finished = datetime.now(timezone.utc)
    ca = pd.to_datetime(df["created_at"], errors="coerce", utc=True) if len(df) else None
    new_ca = pd.to_datetime(pd.DataFrame(rows)["created_at"], errors="coerce", utc=True) if rows else None
    manifest = {
        "endpoint": f"https://api.x.com{ELIGIBLE_POSTS}",
        "auth": "OAuth 1.0a user context (X_API_KEY / X_ACCESS_TOKEN family)",
        "window": {
            "expressed_as": "cursor pagination over the currently-eligible feed; "
                            "no since_id / start_time parameter exists",
            "post_selection": f"feed_lang:{args.lang}", "test_mode": test_mode,
            "created_at_min": str(ca.min()) if ca is not None and len(ca) else None,
            "created_at_max": str(ca.max()) if ca is not None and len(ca) else None,
            "this_run_created_at_min": str(new_ca.min()) if new_ca is not None and len(new_ca) else None,
            "this_run_created_at_max": str(new_ca.max()) if new_ca is not None and len(new_ca) else None,
            "since_days": args.since_days,
        },
        "pulled_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": finished.isoformat().replace("+00:00", "Z"),
        "api_calls": api_calls, "pages": pages,
        "new_posts": len(rows), "duplicates_skipped": dupes, "total_posts": int(len(df)),
        "rate_limit": client.rate_state(),
        "rate_budget": "500 requests / 15 min (x-rate-limit-limit), no per-request charge",
        "cursor": cursor, "raw": str(raw_path.relative_to(SRC)),
        "parquet": str(PARQUET.relative_to(SRC)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    log(f"done: +{len(rows)} new / {len(df)} held in {(finished-started).total_seconds():.0f}s, "
        f"{api_calls} API calls -> {PARQUET.relative_to(SRC)}")


if __name__ == "__main__":
    main()
