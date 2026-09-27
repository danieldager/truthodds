// Unit tests for the extractor, runnable with `npm test` (node --test).
// Fixtures are synthetic but mirror real X GraphQL response shapes — see
// fixtures/README.md for how to replace them with real captures.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  extractTweets, extractRaw, projectSimplified,
  opName, topicFromUrl, unwrapTweet, resolveText, screenName, imageUrls, engagement,
} = require("../extractor.js");

function load(name) {
  const p = path.join(__dirname, "..", "fixtures", name);
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function byId(tweets) {
  const m = new Map();
  for (const t of tweets) m.set(t.id, t);
  return m;
}

test("home timeline: plain, note, and visibility-wrapped tweets; noise ignored", () => {
  const tweets = extractTweets(load("home_timeline.json"));
  const m = byId(tweets);

  // exactly the three real tweets — not the who-to-follow user, not the cursor
  assert.equal(tweets.length, 3);
  assert.deepEqual([...m.keys()].sort(), ["1001", "1002", "1003"]);

  // plain tweet, screen_name from legacy (older location)
  assert.equal(m.get("1001").full_text, "Hello world from the timeline");
  assert.equal(m.get("1001").screen_name, "alice");
  assert.equal(m.get("1001").lang, "en");
  assert.equal(m.get("1001").conversation_id, "1001");
  assert.equal(m.get("1001").created_at, "Wed Mar 12 09:00:00 +0000 2025");
  // PHOTOS only: alice's tweet carries 2 photos + a video + an animated_gif;
  // only the two photo URLs survive, in order (video/gif excluded).
  assert.deepEqual(m.get("1001").image_urls, [
    "https://pbs.twimg.com/media/photo1.jpg",
    "https://pbs.twimg.com/media/photo2.jpg",
  ]);
  // no-media tweets → empty array (never null/undefined)
  assert.deepEqual(m.get("1002").image_urls, []);
  assert.deepEqual(m.get("1003").image_urls, []);

  // note tweet: must prefer the FULL note text, not the truncated legacy text
  const bob = m.get("1002");
  assert.match(bob.full_text, /^This is the FULL/);
  assert.match(bob.full_text, /legacy boundary\.$/); // full sentence, no ellipsis
  assert.ok(!bob.full_text.includes("…"), "note text must not be truncated");
  assert.equal(bob.screen_name, "bob"); // screen_name from core (newer location)

  // visibility-wrapped tweet with emoji + non-Latin script, fully intact
  const carol = m.get("1003");
  assert.equal(carol.full_text, "你好 🌏 café — emoji and non-Latin script test");
  assert.equal(carol.screen_name, "carol");
  assert.equal(carol.lang, "zh");
});

test("user tweets: a retweet folds into its original + retweeted_by; quotes captured", () => {
  const raws = extractRaw(load("user_tweets.json"));
  // raw walk: RT wrapper + inner original + quote wrapper + quoted original = 4
  assert.equal(raws.length, 4);

  // the RT wrapper (2001, dave) projects to the ORIGINAL post with retweeted_by
  const wrapper = projectSimplified(raws.find((r) => r.rest_id === "2001"));
  assert.equal(wrapper.id, "2000");
  assert.equal(wrapper.screen_name, "origauthor");
  assert.equal(wrapper.retweeted_by, "dave");

  const tweets = extractTweets(load("user_tweets.json"));
  const m = byId(tweets);
  // after projection the wrapper and the inner original are ONE record
  assert.equal(tweets.length, 3);
  assert.deepEqual([...m.keys()].sort(), ["1999", "2000", "2002"]);

  const FULL =
    "This is the original but it is going to be cut off because retweets are truncated. Except now it is shown in full.";

  const orig = m.get("2000");
  assert.equal(orig.full_text, FULL);
  assert.ok(!orig.full_text.startsWith("RT @"), "RT prefix must never survive");
  assert.equal(orig.screen_name, "origauthor"); // the real author, never the retweeter
  assert.equal(orig.retweeted_by, "dave"); // merged in from the wrapper copy
  assert.deepEqual(orig.image_urls, ["https://pbs.twimg.com/media/orig_photo.jpg"]);

  // quote tweet and the quoted original are both captured, distinctly; the
  // quote links to its original and carries its text/author as context
  const quote = m.get("2002");
  assert.equal(quote.screen_name, "eve");
  assert.equal(quote.retweeted_by, null); // a quote is the quoter's own post
  assert.equal(quote.quotes, "1999");
  assert.equal(quote.quoted_handle, "frank");
  assert.equal(quote.quoted_text, "The quoted message that eve is reacting to.");
  assert.equal(m.get("1999").quotes, null);
  assert.equal(orig.quotes, null);
  assert.equal(m.get("2002").full_text, "Look at this take 👀 https://t.co/abc123");
  assert.equal(m.get("1999").screen_name, "frank");
  assert.equal(m.get("1999").full_text, "The quoted message that eve is reacting to.");
});

test("engagement counts + author followers; absent fields are null", () => {
  const t = { legacy: { favorite_count: 12, retweet_count: 3, reply_count: 0, quote_count: "2" },
              views: { count: "1500" },
              core: { user_results: { result: { legacy: { screen_name: "a", followers_count: 987 } } } } };
  assert.deepEqual(engagement(t), { like_count: 12, retweet_count: 3, reply_count: 0,
                                    quote_count: 2, view_count: 1500, followers: 987 });
  assert.deepEqual(engagement({ legacy: {} }), { like_count: null, retweet_count: null,
    reply_count: null, quote_count: null, view_count: null, followers: null });
  const m = byId(extractTweets(load("home_timeline.json")));
  assert.ok("like_count" in m.get("1001") && "followers" in m.get("1001"));

  // The live user object resolves screen_name from `.core`, and legacy.followers_count
  // came back null for all 1,161 posts of the first three captures — so the count is
  // read from whichever of the known places carries it.
  const from = (u) => engagement({ legacy: {}, core: { user_results: { result: u } } }).followers;
  assert.equal(from({ core: { screen_name: "a", followers_count: 40 } }), 40);
  assert.equal(from({ relationship_counts: { followers: 41 } }), 41);
  assert.equal(from({ followers_count: 42 }), 42);
  assert.equal(from({ core: { screen_name: "a" } }), null);
});

test("promoted timeline items are flagged; organic ones are not", () => {
  const tweet = (id, text) => ({
    __typename: "Tweet", rest_id: id,
    core: { user_results: { result: { legacy: { screen_name: "adv" } } } },
    legacy: { id_str: id, full_text: text, lang: "en" },
  });
  const root = { data: { home: { instructions: [{ entries: [
    { entryId: "promoted-tweet-1", content: { itemContent: {
      itemType: "TimelineTweet",
      tweet_results: { result: tweet("3001", "Buy our thing") },
      promotedMetadata: { advertiser_results: { result: { rest_id: "9" } } },
    } } },
    { entryId: "tweet-2", content: { itemContent: {
      itemType: "TimelineTweet",
      tweet_results: { result: { __typename: "TweetWithVisibilityResults",
                                 tweet: tweet("3002", "Organic post") } },
    } } },
  ] }] } } };
  const m = byId(extractTweets(root));
  assert.equal(m.size, 2);
  assert.equal(m.get("3001").promoted, true);
  assert.equal(m.get("3002").promoted, false);
  // the fixtures carry no ads
  assert.ok(extractTweets(load("home_timeline.json")).every((t) => t.promoted === false));
});

test("search timeline: tombstones skipped; rest_id-only tweets captured", () => {
  const tweets = extractTweets(load("search_timeline.json"));
  const m = byId(tweets);

  assert.equal(tweets.length, 2);
  assert.deepEqual([...m.keys()].sort(), ["3001", "3003"]);
  assert.equal(m.get("3001").screen_name, "grace");
  // matched via rest_id even though __typename is absent
  assert.equal(m.get("3003").screen_name, "heidi");
});

test("thread detail: pin entry + module items[] are all captured", () => {
  // Validates the recursive walk against TimelinePinEntry and the
  // TimelineTimelineModule `items[].item.itemContent` path (which differs from
  // the normal `content.itemContent` path), plus a cursor inside the module.
  const tweets = extractTweets(load("tweet_detail.json"));
  const m = byId(tweets);

  assert.equal(tweets.length, 3);
  assert.deepEqual([...m.keys()].sort(), ["4000", "4001", "4002"]);
  assert.equal(m.get("4000").screen_name, "ivan"); // pin entry, core handle
  assert.equal(m.get("4001").screen_name, "judy"); // normal entry, legacy handle
  assert.equal(m.get("4002").screen_name, "mallory"); // module items[] path
  assert.match(m.get("4002").full_text, /module items\[\] array/);
});

test("extractRaw returns the full raw tweet nodes, deduped", () => {
  const raws = extractRaw(load("home_timeline.json"));
  assert.equal(raws.length, 3); // alice, bob, carol — not the user/cursor
  const alice = raws.find((n) => n.rest_id === "1001");
  assert.ok(alice.legacy && alice.core, "raw node retains legacy + core (for 4CAT)");
  assert.equal(alice.legacy.full_text, "Hello world from the timeline");
});

test("projectSimplified derives the schema + reads __import_meta provenance", () => {
  const raws = extractRaw(load("home_timeline.json"));
  const node = raws.find((n) => n.rest_id === "1001");
  node.__import_meta = {
    source_platform_url: "https://x.com/search?q=Gaza",
    captured_at: "2026-06-02T00:00:00.000Z",
    operation: "SearchTimeline",
    topic: "Gaza",
    account: "research_main",
  };
  const s = projectSimplified(node);
  assert.equal(s.id, "1001");
  assert.equal(s.full_text, "Hello world from the timeline");
  assert.equal(s.topic, "Gaza");
  assert.equal(s.source_url, "https://x.com/search?q=Gaza");
  assert.equal(s.operation, "SearchTimeline");
  assert.equal(s.captured_at, "2026-06-02T00:00:00.000Z");
  assert.equal(s.account, "research_main");

  // no meta → provenance nulls, core fields still derived
  const carol = projectSimplified(raws.find((n) => n.rest_id === "1003"));
  assert.equal(carol.topic, null);
  assert.equal(carol.source_url, null);
  assert.equal(carol.account, null);
  assert.equal(carol.screen_name, "carol");
});

test("dedup within a single response", () => {
  const dup = {
    a: { __typename: "Tweet", rest_id: "9", legacy: { full_text: "once" } },
    b: { __typename: "Tweet", rest_id: "9", legacy: { full_text: "once" } },
  };
  const tweets = extractTweets(dup);
  assert.equal(tweets.length, 1);
  assert.equal(tweets[0].id, "9");
});

test("robust to junk input", () => {
  assert.deepEqual(extractTweets(null), []);
  assert.deepEqual(extractTweets(undefined), []);
  assert.deepEqual(extractTweets(42), []);
  assert.deepEqual(extractTweets("a string"), []);
  assert.deepEqual(extractTweets({}), []);
  assert.deepEqual(extractTweets([]), []);
});

test("opName parses the GraphQL operation name", () => {
  assert.equal(
    opName("https://x.com/i/api/graphql/aBc123-XyZ/HomeTimeline?variables=%7B%7D"),
    "HomeTimeline"
  );
  assert.equal(opName("https://x.com/i/api/graphql/hash/UserTweets"), "UserTweets");
  assert.equal(
    opName("https://x.com/i/api/graphql/hash/SearchTimeline?q=1#frag"),
    "SearchTimeline"
  );
  assert.equal(opName(""), null);
  assert.equal(opName(undefined), null);
  assert.equal(opName("https://x.com/home"), null);
});

test("flags replies/comments via in_reply_to_status_id_str", () => {
  const root = {
    a: { __typename: "Tweet", rest_id: "5", legacy: { full_text: "a reply", in_reply_to_status_id_str: "4" } },
    b: { __typename: "Tweet", rest_id: "6", legacy: { full_text: "top-level post" } },
  };
  const m = byId(extractTweets(root));
  assert.equal(m.get("5").is_reply, true);
  assert.equal(m.get("5").reply_to, "4");
  assert.equal(m.get("6").is_reply, false);
  assert.equal(m.get("6").reply_to, null);
});

test("topicFromUrl extracts the search query a post was collected under", () => {
  assert.equal(topicFromUrl("https://x.com/search?q=Israel&src=trend_click"), "Israel");
  assert.equal(topicFromUrl("https://x.com/search?q=State%20of%20Play&f=live"), "State of Play");
  assert.equal(topicFromUrl("https://x.com/search?q=%23climate"), "#climate");
  assert.equal(topicFromUrl("https://x.com/home"), null);
  assert.equal(topicFromUrl("https://x.com/someuser/status/123"), null);
  assert.equal(topicFromUrl("https://x.com/search?q="), null);
  assert.equal(topicFromUrl(""), null);
  assert.equal(topicFromUrl(undefined), null);
});

test("helpers: unwrapTweet, resolveText, screenName", () => {
  const inner = { __typename: "Tweet", rest_id: "1", legacy: { full_text: "x" } };
  assert.equal(
    unwrapTweet({ __typename: "TweetWithVisibilityResults", tweet: inner }),
    inner
  );
  assert.equal(unwrapTweet(inner), inner);
  assert.equal(unwrapTweet(null), null);

  assert.equal(resolveText({ legacy: { full_text: "short" } }), "short");
  assert.equal(
    resolveText({
      legacy: { full_text: "short" },
      note_tweet: { note_tweet_results: { result: { text: "long" } } },
    }),
    "long"
  );
  assert.equal(resolveText({}), null);

  assert.equal(
    screenName({ core: { user_results: { result: { core: { screen_name: "a" } } } } }),
    "a"
  );
  assert.equal(
    screenName({ core: { user_results: { result: { legacy: { screen_name: "b" } } } } }),
    "b"
  );
  assert.equal(screenName({}), null);
});

test("imageUrls: photos only, video/gif excluded, deduped, falls back to entities", () => {
  // extended_entities preferred; photos kept in order, video + gif dropped, dup URL deduped
  assert.deepEqual(
    imageUrls({
      legacy: {
        extended_entities: {
          media: [
            { type: "photo", media_url_https: "https://x/a.jpg" },
            { type: "video", media_url_https: "https://x/vthumb.jpg" },
            { type: "photo", media_url_https: "https://x/b.jpg" },
            { type: "animated_gif", media_url_https: "https://x/gthumb.jpg" },
            { type: "photo", media_url_https: "https://x/a.jpg" },
          ],
        },
      },
    }),
    ["https://x/a.jpg", "https://x/b.jpg"]
  );
  // no extended_entities → fall back to entities.media
  assert.deepEqual(
    imageUrls({ legacy: { entities: { media: [{ type: "photo", media_url_https: "https://x/c.jpg" }] } } }),
    ["https://x/c.jpg"]
  );
  // no media / junk → []
  assert.deepEqual(imageUrls({ legacy: {} }), []);
  assert.deepEqual(imageUrls({ legacy: { entities: {} } }), []);
  assert.deepEqual(imageUrls(null), []);
  assert.deepEqual(imageUrls({}), []);
  // a video-only tweet yields no images
  assert.deepEqual(
    imageUrls({ legacy: { extended_entities: { media: [{ type: "video", media_url_https: "https://x/v.jpg" }] } } }),
    []
  );
});
