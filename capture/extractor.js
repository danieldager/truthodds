// extractor.js — path-agnostic recursive tweet extraction.
//
// X reshapes the WRAPPER around its data (instruction types, entry kinds,
// cursors, modules, promoted content) far more often than it reshapes the
// Tweet object itself. So instead of hardcoding the path from the root, we
// recursively walk the parsed JSON and pick out any node that looks like a
// Tweet. The Tweet's own shape (rest_id/__typename + legacy.full_text) is the
// stable anchor.
//
// This file is loaded two ways:
//   • in the service worker via importScripts("extractor.js") — functions
//     become worker globals used by background.js;
//   • in Node via require("./extractor.js") for unit tests — see the
//     module.exports guard at the bottom.

// Some results are wrapped: { __typename: "TweetWithVisibilityResults",
// tweet: <Tweet> }. Peel that to reach the real Tweet. (The recursive walk
// would find the inner Tweet anyway; this helper is for resolving embedded
// retweet/quote originals where we hold the wrapper directly.)
function unwrapTweet(result) {
  if (!result || typeof result !== "object") return null;
  if (result.__typename === "TweetWithVisibilityResults" && result.tweet) {
    return result.tweet;
  }
  return result;
}

// Prefer the full "note tweet" text (long posts) over the truncated
// legacy.full_text. Returns null if neither is a usable string.
function resolveText(tweet) {
  if (!tweet || typeof tweet !== "object" || !tweet.legacy) return null;
  const note =
    tweet.note_tweet &&
    tweet.note_tweet.note_tweet_results &&
    tweet.note_tweet.note_tweet_results.result &&
    tweet.note_tweet.note_tweet_results.result.text;
  if (typeof note === "string" && note.length) return note;
  return typeof tweet.legacy.full_text === "string" ? tweet.legacy.full_text : null;
}

// Author handle. X has been migrating user fields out of `legacy` into `core`,
// so screen_name may live in either place depending on deployment — check both.
function screenName(tweet) {
  const r =
    tweet.core && tweet.core.user_results && tweet.core.user_results.result;
  if (!r || typeof r !== "object") return null;
  if (r.core && typeof r.core.screen_name === "string") return r.core.screen_name;
  if (r.legacy && typeof r.legacy.screen_name === "string") return r.legacy.screen_name;
  return null;
}

// Photo image URLs on a tweet — PHOTOS ONLY. Video and animated-GIF media are
// intentionally excluded (images yes, videos no). Reads legacy.extended_entities
// .media (the complete media list, incl. multi-photo) and falls back to
// legacy.entities.media. Returns media_url_https for type === "photo", deduped
// and order-preserved; [] when the tweet carries no photos.
// Follower count of the author. Same `legacy` -> `core` migration as
// screenName: the live timeline shape resolves screen_name from `.core`, and
// `legacy.followers_count` came back null for every post of the first 1,161
// captured, so try each place the count is known to sit rather than one path.
function followerCount(u) {
  if (!u || typeof u !== "object") return null;
  const cands = [
    u.legacy && u.legacy.followers_count,
    u.core && u.core.followers_count,
    u.relationship_counts && u.relationship_counts.followers,
    u.followers_count,
  ];
  for (const c of cands) if (typeof c === "number") return c;
  return null;
}

// Engagement at capture time + author size. X's `views.count` is a string.
function engagement(tweet) {
  const l = tweet.legacy || {};
  const u = tweet.core && tweet.core.user_results && tweet.core.user_results.result;
  const n = (v) => (v == null || v === "" || isNaN(Number(v)) ? null : Number(v));
  return {
    like_count: n(l.favorite_count),
    retweet_count: n(l.retweet_count),
    reply_count: n(l.reply_count),
    quote_count: n(l.quote_count),
    view_count: n(tweet.views && tweet.views.count),
    followers: followerCount(u),
  };
}

function imageUrls(tweet) {
  if (!tweet || typeof tweet !== "object" || !tweet.legacy) return [];
  const ext = tweet.legacy.extended_entities;
  const ent = tweet.legacy.entities;
  const media =
    (ext && Array.isArray(ext.media) && ext.media) ||
    (ent && Array.isArray(ent.media) && ent.media) ||
    [];
  const out = [];
  const seen = new Set();
  for (const m of media) {
    if (!m || m.type !== "photo") continue; // skip "video" + "animated_gif"
    const u = m.media_url_https;
    if (typeof u === "string" && u && !seen.has(u)) {
      seen.add(u);
      out.push(u);
    }
  }
  return out;
}

// Walk the parsed JSON and collect the RAW Tweet objects (deduped by id within
// this response). Returns references to the full tweet nodes — this is what
// gets stored and what 4CAT's importer needs.
function extractRaw(root) {
  const out = [];
  const seenLocal = new Set();

  (function walk(node) {
    if (!node || typeof node !== "object") return;

    const looksLikeTweet =
      (node.__typename === "Tweet" || typeof node.rest_id === "string") &&
      node.legacy &&
      typeof node.legacy.full_text === "string";

    // Ads: X marks a promoted timeline item with `promotedMetadata` beside its
    // `tweet_results`. The tweet node itself carries no marker, so tag it here,
    // before descending into it. Downstream drops promoted posts.
    if (node.promotedMetadata && node.tweet_results) {
      const t = unwrapTweet(node.tweet_results.result);
      if (t && typeof t === "object") t.__promoted = true;
    }

    if (looksLikeTweet) {
      const id = node.rest_id || node.legacy.id_str;
      if (id && !seenLocal.has(id)) {
        seenLocal.add(id);
        out.push(node);
      }
    }

    if (Array.isArray(node)) {
      for (let i = 0; i < node.length; i++) walk(node[i]);
    } else {
      for (const k in node) walk(node[k]);
    }
  })(root);

  return out;
}

// Project a stored record (a raw Tweet node, optionally carrying __import_meta
// provenance) down to the simplified, analysis-friendly schema.
function projectSimplified(node) {
  if (!node || typeof node !== "object" || !node.legacy) return null;
  const id = node.rest_id || node.legacy.id_str;
  if (!id) return null;

  // Retweet wrapper: the record IS the original post (its id, author, text,
  // media, date), with the retweeter noted in `retweeted_by` (Daniel 2026-08-25).
  // The wrapper's own "RT @user: …" form, id and author are dropped — a retweet
  // adds no content, only who amplified it. Quote tweets are left as-is: the
  // quoter's post is its own record and the quoted original is captured
  // separately under its own id.
  const rtInner =
    node.legacy.retweeted_status_result &&
    unwrapTweet(node.legacy.retweeted_status_result.result);
  if (rtInner) {
    const rec = projectSimplified({ ...rtInner, __import_meta: node.__import_meta });
    if (rec) rec.retweeted_by = screenName(node);
    return rec;
  }

  // Quote tweet: the author's own post with another post embedded. The quoted
  // post is captured separately under its own id; here we only LINK to it and
  // carry its text/author as context for claim extraction (Daniel 2026-08-25).
  // X puts quoted_status_result at the tweet's top level (legacy fallback kept).
  const qWrap = node.quoted_status_result || node.legacy.quoted_status_result;
  const quoted = qWrap && unwrapTweet(qWrap.result);
  const quotes = (quoted && (quoted.rest_id || (quoted.legacy && quoted.legacy.id_str))) || null;

  const replyTo = node.legacy.in_reply_to_status_id_str || null;
  const meta = node.__import_meta || {};
  return {
    id,
    full_text: resolveText(node),
    screen_name: screenName(node),
    created_at: node.legacy.created_at || null,
    lang: node.legacy.lang || null,
    conversation_id: node.legacy.conversation_id_str || null,
    is_reply: !!replyTo,
    reply_to: replyTo,
    retweeted_by: null, // handle of the account whose retweet surfaced this post
    promoted: !!node.__promoted, // an ad (promotedMetadata on the timeline item)
    quotes, // id of the quoted post (null unless this is a quote tweet)
    quoted_handle: quotes ? screenName(quoted) : null,
    quoted_text: quotes ? resolveText(quoted) : null,
    image_urls: imageUrls(node),
    ...engagement(node),
    captured_at: meta.captured_at || null,
    source_url: meta.source_platform_url || null,
    operation: meta.operation || null,
    topic: meta.topic || null,
    account: meta.account || null, // handle logged in when captured (the feed's owner)
  };
}

// Project a batch of stored raw records to the simplified schema, one record
// per post. A retweet wrapper and its inner original both project to the
// original's id; they merge here, keeping `retweeted_by` from whichever copy
// carries it. First copy wins for everything else.
function projectAll(records) {
  const byId = new Map();
  for (const raw of records) {
    const rec = projectSimplified(raw);
    if (!rec) continue;
    const prev = byId.get(rec.id);
    if (!prev) byId.set(rec.id, rec);
    else if (!prev.retweeted_by && rec.retweeted_by) prev.retweeted_by = rec.retweeted_by;
  }
  return [...byId.values()];
}

// Convenience: raw extraction + projection in one (used by the unit tests).
function extractTweets(root) {
  return projectAll(extractRaw(root));
}

// Pull the GraphQL OperationName from the request URL:
//   /i/api/graphql/<query-id-hash>/<OperationName>?variables=...
function opName(url) {
  const m = /\/graphql\/[^/]+\/([^/?#]+)/.exec(url || "");
  return m ? m[1] : null;
}

// The search query a post was collected under — the `q` of a /search page URL
// (for autopilot trending, exactly the trend term we navigated to). null for
// non-search browsing (home, profiles, threads). Decoding is automatic.
function topicFromUrl(url) {
  try {
    const u = new URL(url);
    if (u.pathname === "/search") {
      const q = u.searchParams.get("q");
      return q && q.trim() ? q.trim() : null;
    }
  } catch (_) {}
  return null;
}

// Dual environment: CommonJS (Node tests) gets named exports; the service
// worker (classic importScripts) just sees these as globals and skips this.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    extractTweets, extractRaw, projectSimplified, projectAll,
    opName, topicFromUrl, unwrapTweet, resolveText, screenName, imageUrls, engagement,
  };
}
