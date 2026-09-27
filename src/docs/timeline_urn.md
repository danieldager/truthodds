# Timeline urn — where it lives, what's in it, how to read it

The production claim set drawn from X timeline captures. Live and growing: Daniel
captures ~1 h sessions, each export is landed, screened and chained the same day.
Safe to read at any time — every write is a whole-file replace, never a partial one.

## Path

```
src/eval/data/tweet_corpus/timeline_urn.parquet
```

Gitignored (like all of `eval/data/**/*.parquet`), so it is a local artifact, not a
committed one. One row per CLAIM; post fields are denormalized onto every row, so no
join is needed for post-level analysis — group by `post_id`.

```python
import polars as pl
u = pl.read_parquet("eval/data/tweet_corpus/timeline_urn.parquet")   # run from src/
cw = u.filter("checkworthy")
```

## Contents (2026-08-26)

**3,910 claims / 2,699 checkworthy / 2,511 verify-eligible / 1,084 posts**, in two
frames — always analyse with `frame` as a covariate, never pooled blind:

| frame | claims | cw | posts | post age | what it is |
|---|---|---|---|---|---|
| `feed` | 2,800 | 1,974 | 804 | ~0 days | For You captures, tagged `account` |
| `search_june` | 1,110 | 725 | 280 | ~85 days | June search draw, untagged |

**One file, one instrument.** Every claim was produced by the current gate + extract
+ normalize; the base was a single clean run on 2026-08-26 (provenance `built_at`
13:24-13:34) with nothing carried over from an earlier extraction, and later capture
batches are appended incrementally through the identical chain. The cumulative sources are
`urn_{quote,user,outlet}_verify_input.parquet` and nothing else.
`general_*_verify_input.parquet` is FROZEN: it holds Aug-24 claims off an older
chain that the eval build still reads. Never mix the two.

Voices: user 2,607, quote 211, outlet 89. Accounts: danieldager1 998 cw,
mr_misinfo 251 cw, untagged (June) 725 cw.
Checkworthy topics: politics 658, business 509, technology 251, health 111,
crime 98, sports 80, science 72, entertainment 40, disaster 29.
Languages: en 1,687, fr 287. Under EN-assertions-only parity: **1,473 claims**,
511 posts, 333 handles, top-20 posts 19%, top-10 handles 19%.

## What is and isn't in it

IN, as `frame="feed"`: posts an account's **For You feed served**
(`operation == "HomeTimeline"` with a tagged `account`) that passed the STRICT
qualify screen (public affairs AND stakes), chained through extract → normalize.

IN, as `frame="search_june"`: the June search-derived corpus, re-screened under
STRICT and chained in the same run as the feed half. Folded in 2026-08-26 for age
and topic balance (~85 days vs ~0), which also gives the urn an internal age control
on a fixed instrument.

OUT, deliberately:
- **Browsing** — profile timelines, opened threads, search, Explore. Captured and
  screened (so nothing is lost), never chained. They qualify at 38% / 54% / 12% vs
  the feed's 17%, which is exactly why they can't join silently: it would change the
  sampling frame, not just the yield. `URN_OPS` in `ingest_timeline_capture.py`.
- Non-en/fr posts, posts under 15 chars, ads (`promoted`).

## Columns that matter

`claim`, `claim_id`, `type` (assertion|attribution), `checkworthy`,
`verify_eligible`, `category` (checkable|opinion|trivial|unresolved|artifact),
`topic`, `voice` (user|quote|outlet) · post: `post_id`, `post_text`, `handle`,
`created_at`, `lang`, `is_quote`, `quoted_handle`, `quoted_text`, `retweeted_by`,
`is_reply`, `conversation_id`, `image_urls` · provenance: `account`, `operation`,
`captured_at` · engagement: `like_count`, `retweet_count`, `reply_count`,
`quote_count`, `view_count`, `followers`.

## Known caveats

- **Verification has not been run.** These are extracted + normalized claims, not
  adjudicated ones. No veracity label exists yet.
- **Quote-voice noise**: the quote chain has a 5-run CV of ~25% (user chain 1.9%).
  136 of 942 claims are quote-voice; treat quote-only cuts as noisy.
- **`promo_ad` over-fires on third-party news** in finance/tech-heavy feeds — a news
  outlet reporting someone else's funding/launch/price is read as the author's own
  promo. Costs recall in one genre, does not hurt precision of what's kept. Unfixed
  (Daniel: precision high, missing some is fine).
- `engagement` fields are captured at capture time, not post time. `followers` is
  null for everything captured before 2026-08-26 12:00 (extractor bug, since fixed).

## Refreshing

```bash
cd src && uv run python -m eval.scripts.claim_sourcing.ingest_timeline_capture        # plan, free
cd src && uv run python -m eval.scripts.claim_sourcing.ingest_timeline_capture --go   # screen + chain
```
Rewrites `timeline_urn.parquet` at the end of every `--go`. ~$0.01–0.03 per session.
Per-batch human-readable review: `general_{user,quote}_20260826_normalized.html`.
