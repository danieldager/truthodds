// content.js — isolated-world bridge.
// Receives MAIN-world messages from injected.js and forwards the raw JSON
// to the background service worker. Runs in the extension's isolated world,
// so it CAN call chrome.runtime.* (the page cannot).

const MARKER = "__XCAP__";

// GraphQL ops that fetch a page of the feed. Pacing is a cap on how many of
// these go out per minute (X limits requests, not scroll style), so autopilot
// needs to see the request stream — it reads the ledger below directly, since
// every content script of this extension shares one isolated world per frame.
const FEED_OP = /\/graphql\/[^/]+\/(\w*Timeline\w*|UserTweets\w*|TweetDetail)/;

// Which X account this tab is logged in as — the sampling frame is one
// account's feed, so every record carries it (Daniel's main vs dummy account
// must never blend silently). Handle from the account-switcher button in the
// left nav; if the DOM changes, the numeric user id from the `twid` cookie.
function detectAccount() {
  try {
    const el = document.querySelector('[data-testid="SideNav_AccountSwitcher_Button"]');
    const m = el && /@([A-Za-z0-9_]{1,15})/.exec(el.textContent || "");
    if (m) return m[1];
    const c = /(?:^|;\s*)twid=u%3D(\d+)/.exec(document.cookie || "");
    if (c) return "id:" + c[1];
  } catch (_) {}
  return null;
}

window.addEventListener("message", (event) => {
  // Only trust messages this window posted to itself (injected.js uses
  // window.postMessage). event.source === window rejects messages from
  // iframes/other windows; matching origin rejects cross-origin posts; the
  // marker rejects unrelated page chatter.
  //
  // Caveat: injected.js shares the page's JS world, so a hostile script
  // already running on x.com could in principle post a spoofed payload bearing
  // the marker. The downstream effect is at worst noise in the dataset (the
  // background only parses it as tweet JSON; no privileged action), so for this
  // passive-research threat model that residual risk is accepted.
  if (event.source !== window) return;
  if (event.origin !== window.location.origin) return;
  const d = event.data;
  if (!d || d.source !== MARKER || typeof d.body !== "string") return;

  try {
    if (FEED_OP.test(d.url || "")) {
      const log = (window.__xcapFeedReqs = window.__xcapFeedReqs || []);
      log.push(Date.now());
      if (log.length > 300) log.splice(0, log.length - 300);
    }
  } catch (_) {}

  try {
    const p = chrome.runtime.sendMessage({
      type: "xcap_payload",
      url: d.url,
      body: d.body,
      rl: d.rl || null, // per-endpoint rate-limit headers, for the sweep governor
      account: detectAccount(),
    });
    // In MV3 sendMessage returns a promise; swallow rejections (e.g. the
    // service worker briefly unavailable) so they don't surface as unhandled.
    if (p && typeof p.catch === "function") p.catch(() => {});
  } catch (_) {
    // Extension context can be invalidated (e.g. the extension was reloaded
    // while this page stayed open). Nothing to do but ignore — the next page
    // load re-establishes the bridge.
  }
});

// ---------------------------------------------------------------------------
// Fallback tag-injection. NOT used by default: the manifest declares
// injected.js as a `world: "MAIN"` content script (Chrome 111+), which is the
// clean path and runs at document_start before X's app code. If MAIN-world
// timing ever proves unreliable on a target Chromium build, remove the second
// content-script entry from manifest.json and uncomment the block below; the
// web_accessible_resources entry (already in the manifest) makes it loadable.
// ---------------------------------------------------------------------------
//
// (function injectFallback() {
//   try {
//     const s = document.createElement("script");
//     s.src = chrome.runtime.getURL("injected.js");
//     s.onload = () => s.remove();
//     (document.head || document.documentElement).prepend(s);
//   } catch (_) {}
// })();
