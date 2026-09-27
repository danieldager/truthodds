# Fixtures

These JSON files are **synthetic** but mirror the real shape of X / Twitter
GraphQL responses (`TimelineAddEntries` → `entries[]` →
`content.itemContent.tweet_results.result`). They exist so `extractor.js` can be
unit-tested offline (`npm test`).

They deliberately exercise the cases the extractor must get right:

| File | Covers |
|------|--------|
| `home_timeline.json` | plain tweet (screen_name in `legacy`), long **note tweet** (full text preferred over truncated `legacy.full_text`, screen_name in `core`), `TweetWithVisibilityResults` wrapper with emoji + non-Latin script, who-to-follow module + cursor (must be ignored) |
| `user_tweets.json` | **retweet** wrapper (text resolved to the embedded original, not the truncated `RT @user: …`) + the inner original; **quote tweet** + the quoted original |
| `search_timeline.json` | normal tweet, **tombstone** (unavailable post, ignored), tweet with `rest_id` but no `__typename` (still matched) |

## Replacing with real captures (recommended before trusting field names)

X reshapes these responses periodically. To validate against the live API:

1. Open `x.com` logged in → DevTools → **Network** → filter `graphql`.
2. Scroll the timeline / open a profile / run a search / open a thread.
3. Click a `HomeTimeline` / `UserTweets` / `SearchTimeline` / `TweetDetail`
   response → right-click → **Copy response** → save it here as e.g.
   `real_home_timeline.json`.
4. Add assertions in `test/extractor.test.js` against the real file and run
   `npm test`. If field names drifted, update the predicate / paths in
   `extractor.js` (see the maintenance note in the top-level README).

Keeping a couple of real responses here makes the extractor quick to
re-validate after any X change.
