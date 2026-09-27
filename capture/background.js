// background.js — MV3 service worker: parse, extract, dedup, persist.
//
// Classic worker (not a module) so it can importScripts the shared extractor,
// which also doubles as a Node-testable CommonJS module.
importScripts("extractor.js"); // provides extractTweets() and opName()

const SEEN_KEY = "seen_ids";
const DATA_KEY = "captured";
const RUN_KEY = "autopilot_run"; // autopilot runtime state (for topic attribution)
const CFG_KEY = "autopilot_cfg"; // autopilot settings (for session-end actions)
const RATELIMIT_KEY = "ratelimits"; // per-endpoint {remaining,reset,limit,at} from response headers
const DEBUG = false; // flip true to log every capture to the SW console

// ---------------------------------------------------------------------------
// Serialized read-modify-write.
//
// Rapid scrolling fires many payloads nearly simultaneously. Each handler does
// read → modify → write against chrome.storage.local; if two interleave at the
// async read, the second write clobbers the first and freshly-captured tweets
// are LOST (not merely duplicated). A single promise chain forces the handlers
// to run one at a time. The chain lives only for this service-worker lifetime —
// which is exactly the window in which the race exists — and every task still
// reads fresh from storage, so a teardown mid-chain loses nothing.
// ---------------------------------------------------------------------------
let queue = Promise.resolve();
function enqueue(task) {
  queue = queue.then(task).catch((e) => console.error("[xcap]", e));
  return queue;
}

// Confirmation line so you can tell the worker reloaded with logging on, even
// before any new post is captured.
if (DEBUG) console.log("[xcap] service worker active — capture logging ON");

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || typeof msg.type !== "string") return false;

  if (msg.type === "xcap_payload") {
    enqueue(() => handlePayload(msg.body, msg.url, sender.tab && sender.tab.url, msg.rl,
                                msg.account || null));
    return false; // fire-and-forget; no response needed
  }

  if (msg.type === "xcap_reset") {
    // Ack only after the clear has actually been written, so the popup's
    // refresh reads the emptied store rather than racing it.
    enqueue(() => chrome.storage.local.set({ [SEEN_KEY]: [], [DATA_KEY]: [] }))
      .then(() => sendResponse({ ok: true }));
    return true; // keep the message channel open for the async sendResponse
  }

  // autopilot.js (a content script) can't read its own tab id; hand it back so a
  // run stays bound to the single tab that started it.
  if (msg.type === "autopilot_whoami") {
    sendResponse({ tabId: sender.tab && sender.tab.id });
    return true;
  }

  // A session finished (post target / time budget): run the end-of-session
  // actions (export → reset → restart). Serialized so it doesn't race captures.
  if (msg.type === "autopilot_session_end") {
    enqueue(() => handleSessionEnd());
    return false;
  }

  return false;
});

async function handlePayload(bodyText, apiUrl, pageUrl, rl, account) {
  // Stamp the live per-endpoint rate-limit budget FIRST, straight from the
  // response headers — independent of the body, so it lands even on a thin or
  // 429 response that carries no tweets. The profiles sweep governor reads this
  // (RATELIMIT_KEY[UserTweets]) to pause before it burns through the window.
  if (rl && typeof rl.remaining === "number") {
    const op0 = opName(apiUrl);
    if (op0) {
      const cur = (await chrome.storage.local.get(RATELIMIT_KEY))[RATELIMIT_KEY] || {};
      cur[op0] = { remaining: rl.remaining, reset: rl.reset || 0, limit: rl.limit || 0, at: Date.now() };
      await chrome.storage.local.set({ [RATELIMIT_KEY]: cur });
    }
  }

  let json;
  try {
    json = JSON.parse(bodyText);
  } catch {
    return; // not JSON / partial body — ignore
  }

  let raws;
  try {
    raws = extractRaw(json);
  } catch (e) {
    if (DEBUG) console.error("[xcap] extract failed", e);
    return;
  }
  if (!raws || !raws.length) return;

  const store = await chrome.storage.local.get([SEEN_KEY, DATA_KEY, RUN_KEY]);
  const seen = new Set(store[SEEN_KEY] || []);
  const captured = store[DATA_KEY] || [];

  const op = opName(apiUrl);
  // Attribution, most-specific first:
  //  1. /search pages carry the exact query in the URL.
  //  2. Home-timeline fetches are self-identifying by GraphQL op — timing-proof
  //     (no dependence on when the driver stamps run.context after a nav).
  //  3. Anything else during a run (notably thread-dive replies, op=TweetDetail,
  //     no `q` in the /status URL) inherits the driver's stamped context — so a
  //     dive off the home feed stays "(home feed)", not the last trend topic.
  let topic = topicFromUrl(pageUrl);
  const run = store[RUN_KEY];
  if (!topic && /^Home(Latest)?Timeline$/.test(op || "")) topic = "(home feed)";
  if (!topic && run && run.running && run.context) topic = run.context;
  const now = Date.now();
  const nowIso = new Date(now).toISOString();

  const fresh = [];
  for (const node of raws) {
    const id = node.rest_id || (node.legacy && node.legacy.id_str);
    if (!id || seen.has(id)) continue;
    seen.add(id);
    // Store the raw tweet object + Zeeschuimer-style provenance. This is the
    // 4CAT-native record; the simplified schema is derived from it on export.
    fresh.push({
      ...node,
      id,
      __import_meta: {
        source_platform: "twitter",
        source_platform_url: pageUrl || "",
        timestamp_collected: now,
        captured_at: nowIso,
        operation: op,
        topic: topic || null,
        account: account || null, // the logged-in handle this feed belongs to
      },
    });
  }

  if (!fresh.length) return;

  await chrome.storage.local.set({
    [SEEN_KEY]: [...seen],
    [DATA_KEY]: captured.concat(fresh),
  });

  if (DEBUG) {
    console.log(
      `[xcap] ${op || "?"} +${fresh.length} (total ${captured.length + fresh.length})`
    );
  }
}

// ---------------------------------------------------------------------------
// Session-end actions (auto-export → reset → restart), run in the worker so
// they fire even with the popup closed.
// ---------------------------------------------------------------------------
async function handleSessionEnd() {
  const store = await chrome.storage.local.get([CFG_KEY, DATA_KEY, RUN_KEY]);
  const cfg = store[CFG_KEY] || {};
  const records = dedupById(store[DATA_KEY] || []);
  const wantExport = !!cfg.autoExport && records.length > 0;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");

  let exportOk = !wantExport; // nothing to export → treat as success
  let note = "";

  if (wantExport) {
    try {
      if (cfg.exportDest === "4cat" && cfg.fourcatUrl) {
        // 4CAT gets the raw Zeeschuimer-format records as-is.
        const raw = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
        const res = await uploadTo4cat(cfg.fourcatUrl, raw);
        if (DEBUG) console.log("[xcap] 4CAT upload result:", res);
      } else {
        // Downloaded file = the simplified, derived schema.
        const simple = projectAll(records).map((r) => JSON.stringify(r)).join("\n") + "\n";
        await downloadNdjson(simple, `x_capture_${stamp}.ndjson`);
      }
      exportOk = true;
    } catch (e) {
      // Fail gracefully: keep the data, and save a local fallback file so a
      // session is never lost just because (e.g.) the 4CAT server is unreachable.
      console.error("[xcap] session-end export failed:", e);
      note = "export failed — data kept";
      try {
        const raw = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
        await downloadNdjson(raw, `x_capture_${stamp}_export-failed.ndjson`);
        note = "export failed (server unreachable?) — saved a fallback file; data kept";
      } catch (_) {}
    }
  }

  // Reset is manual-only (the Reset button) — a finished session never auto-wipes the
  // store, so a completed sweep waits for you to Export. autoRestart still guards on
  // exportOk so it can't restart over unsaved data.
  if (cfg.autoRestart && exportOk) {
    const run = store[RUN_KEY] || {};
    const tabId = run.tabId;
    const cur = (await chrome.storage.local.get(DATA_KEY))[DATA_KEY];
    await chrome.storage.local.set({
      [RUN_KEY]: {
        running: true,
        startedAt: Date.now(),
        startCount: Array.isArray(cur) ? cur.length : 0,
        tabId: tabId != null ? tabId : null,
        mode: cfg.mode,
        queue:
          cfg.mode === "manual"
            ? (cfg.manualTopics || []).map((t) => ({
                label: t,
                url: `https://x.com/search?q=${encodeURIComponent(t)}&src=typed_query&f=live`,
              }))
            : cfg.mode === "profiles"
              ? (cfg.profileHandles || []).map((h) => ({
                  handle: h, label: "@" + h, url: "https://x.com/" + h,
                }))
              : [],
        idx: 0,
        visited: 0,
        reason: null,
        seededFrom: false,
      },
    });
    if (tabId != null) {
      try {
        await chrome.tabs.sendMessage(tabId, { type: "autopilot_start" });
      } catch (_) {}
    }
  } else if (note) {
    // Surface the failure in the reason the popup shows.
    const cur = (await chrome.storage.local.get(RUN_KEY))[RUN_KEY] || {};
    cur.reason = `${cur.reason || "session ended"} — ${note}`;
    await chrome.storage.local.set({ [RUN_KEY]: cur });
  }
}

function downloadNdjson(text, filename) {
  const url = "data:application/x-ndjson;base64," + b64utf8(text);
  return chrome.downloads.download({ url, filename, saveAs: false });
}

function dedupById(records) {
  const seen = new Set();
  const out = [];
  for (const r of records) {
    if (!r || seen.has(r.id)) continue;
    seen.add(r.id);
    out.push(r);
  }
  return out;
}

// UTF-8-safe base64 for the data: URL (btoa alone corrupts non-Latin text).
function b64utf8(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

// Upload to a 4CAT instance using Zeeschuimer's protocol: POST the NDJSON to
// /api/import-dataset/ with the platform header, then poll /api/check-query/.
// Requires host permission for the 4CAT origin (localhost is in the manifest).
async function uploadTo4cat(baseUrl, ndjson) {
  const root = baseUrl.replace(/\/+$/, "");
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 15000); // don't hang on a dead server
  let res;
  try {
    res = await fetch(root + "/api/import-dataset/", {
      method: "POST",
      headers: { "X-Zeeschuimer-Platform": "twitter" },
      body: ndjson,
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) throw new Error(`4CAT responded ${res.status}`);
  const data = await res.json().catch(() => ({}));
  if (data && data.key) {
    const pollUrl = root + "/api/check-query/?key=" + encodeURIComponent(data.key);
    let fails = 0;
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      const p = await fetch(pollUrl).then((x) => x.json()).catch(() => null);
      if (p && p.done) return p;
      if (p === null && ++fails >= 3) break; // server went away mid-poll
    }
  }
  return data;
}
