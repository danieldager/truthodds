# `capture/` — X timeline capture (Chrome MV3)

Ported from the standalone `zeerover` repo on 2026-08-25 (TH2). This is the
**upstream half of the ingest path** and the first piece of the production
browser tool: it captures the posts you browse on X into NDJSON, which
`ingest_timeline_capture.py` carries to a claims parquet.

Only the capture path was ported. Left behind in `zeerover`: `gen_icons.py`,
the repo's own README/LICENSE/clog/HUB, and its local `*.ndjson` samples.

## Install (load unpacked)

1. `chrome://extensions` → enable **Developer mode**.
2. **Load unpacked** → select this `capture/` folder.
3. Pin the toolbar icon. Chrome/Chromium 111+.
4. After editing any file here: press ↻ on the extension card **and** reload the
   X tab — the content scripts run at `document_start`.

## Capture

Browse X logged in; `injected.js` hooks `fetch`/`XHR` in the MAIN world and
posts are read out of the GraphQL responses the page already fetches. Click the
icon for the running count, **Export NDJSON**, or **Reset**.

**Autopilot** (optional, off by default) drives the page so passive capture sees
more. For timeline collection use **Source → Home feed only**, set *Run for*,
leave *Feed requests / min* at 20, Start. It scrolls at one steady beat
(*Scroll every* seconds, default 1) and the ONLY thing that ever holds it is the
request cap: X meters timeline fetches, not scroll style, so when the last minute
already spent its budget it waits for the oldest request to age out — under a
minute, never the multi-minute idles the old posts/hour pacer produced. X's own
response headers report HomeTimeline at **500 requests / 15 min** (33/min) and a
real capture session uses **0.2-0.5/min**, so the cap is headroom: scroll cadence,
not the cap, sets the collection rate. (SearchTimeline is only 50/15 min — trending
and manual modes are the constrained ones.) Tab-bound;
halts on any Cloudflare/Arkose challenge. This is automation — X's ToS restricts
it; use a dedicated research account.

### Collection protocol (the urn = what a user's eyes land on)

- **For You tab, scrolled naturally.** Nothing else goes in the urn — no search,
  no trending, no hunting accounts. Records from the For You feed carry
  `operation: "HomeTimeline"`; the Following tab is `HomeLatestTimeline`. Check
  which tab is selected before starting — autopilot opens `/home`, which is
  whichever tab was last used.
- **Several 1–2k sessions across days and times of day**, not one long dump —
  one sitting is one news cycle.
- **Per account: Export → Reset → switch account → capture.** Every record
  carries `account` (the logged-in handle, read from the page), so files never
  blend — but dedup is per browser profile, so WITHOUT the Reset a post already
  seen on account A is silently not captured on account B, which biases B's
  sample. Reset clears that.
- Trending/search captures, if any, are a tagged side pool (`operation`,
  `topic`), never part of the fit.

## Run order (capture → claims)

```bash
# 1. Export NDJSON from the popup (lands in ~/Downloads)
# 2. Plan — lands the export in the corpus, prints counts + cost. No paid calls.
cd src && uv run python -m eval.scripts.claim_sourcing.ingest_timeline_capture

# 3. Smoke — sample-screen 200 + chain 20 posts, print the claims (~$0.01)
uv run python -m eval.scripts.claim_sourcing.ingest_timeline_capture --smoke

# 4. Full — screen all + chain every not-yet-chained qualifying post
#    (the STRICT screen: public affairs + stakes; anecdote/mockery/rhetoric/self-promo
#     are out — see general_pool_screen.SYS_STRICT; ~1/3 of feed posts qualify)
uv run python -m eval.scripts.claim_sourcing.ingest_timeline_capture --go
```

Output: `eval/data/tweet_corpus/general_verify_input.parquet` (the claims
parquet). The eval-stratum build (`build_eval/general_pool_build.py`) imports
the same screen + chain stages and adds the scoring/report stages on top.

## Schema

`extractor.js` `projectSimplified` emits one JSON object per post: `id`,
`full_text` (note-tweet aware, retweets resolved to the original), `screen_name`,
`created_at`, `lang`, `conversation_id`, `is_reply`, `reply_to`, `retweeted_by`,
`promoted` (true for ads — the ingest drops them), `account` (the logged-in
handle whose feed this was), `like_count`/`retweet_count`/`reply_count`/
`quote_count`/`view_count` (engagement at capture time), `followers` (author),
`image_urls` (photos only — video/GIF excluded), `captured_at`, `source_url`,
`operation`, `topic`.

**Retweets** are exported as the ORIGINAL post — its id, author, text, media,
date — with `retweeted_by` = the handle whose retweet surfaced it (`null`
otherwise). The wrapper's "RT @user: …" form never appears. **Quote tweets** are
the quoter's own record, with `quotes` (the quoted post's id), `quoted_handle`
and `quoted_text` carried as context; the quoted original is also captured
separately under its own id. Downstream, quote posts go through the
`--voice quote` extraction prompt (claims from the quoter's text only; the
quoted post resolves referents). Captures before 2026-08-25 lack these fields
(June captures also lack `image_urls`/`topic`); the ingest reads them with
defaults.

## Tests

`npm test` runs the extractor unit tests (`test/extractor.test.js` against
`fixtures/`). Node was not available in the porting session — the files were
syntax-checked only, so **run `npm test` once before trusting a capture run**.
