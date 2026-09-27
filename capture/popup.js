// popup.js — capture controls + autopilot + session-end settings.
// The export Blob is built HERE (the MV3 service worker lacks createObjectURL).

const DATA_KEY = "captured";
const CFG_KEY = "autopilot_cfg";
const RUN_KEY = "autopilot_run";

// Study default: the NewsGuard-grid outlet handles (top 10 by Tranco traffic per
// cell) from the fact-checking repo — reliability (NewsGuard score >=/< 60) x bias
// (NewsGuard orientation): Reliable-Left, Reliable-Right, Unreliable-Left,
// Unreliable-Right. Interleaved RL/RR/UL/UR so any prefix (a stopped-early or
// chunked sweep) stays balanced across the four cells. Only a textarea pre-fill —
// the user edits before starting; the extension never reads the repo at runtime.
// Two grid outlets can't be swept as-is: naturalnews.com (no live X account —
// banned 2019, omitted) and virginiamercury.com (its NewsGuard handle is the
// parent network @statesnewsroom, used here as a stand-in). 39 handles.
const DEFAULT_PROFILE_HANDLES = [
  "huffpost", "FoxNews", "MSNBC", "BreitbartNews",
  "NewYorker", "foxbusiness", "dailykos", "EpochTimes",
  "vice", "NRO", "WSWS_Updates", "newsmax",
  "voxdotcom", "DCExaminer", "Consortiumnews", "gatewaypundit",
  "Slate", "reason", "MintPressNews", "CBNOnline",
  "rollingstone", "DailyCaller", "Blavity", "TheBlaze",
  "thedailybeast", "townhallcom", "PressenzaIPA", "realdailywire",
  "NYDailyNews", "CTmagazine", "statesnewsroom", "worldnetdaily",
  "Salon", "FreeBeacon", "DemocracyDocket", "FDRLST",
  "MotherJones", "DailySignal", "TheGrayzoneNews"
];

// Normalize a pasted profile line to a bare lowercase handle: accepts "@handle",
// a full x.com / twitter.com URL, or a bare handle.
function normHandle(s) {
  let h = String(s || "").trim();
  if (!h) return "";
  const m = /(?:x|twitter)\.com\/+([^/?#\s]+)/i.exec(h);
  if (m) h = m[1];
  h = h.replace(/^@+/, "").split(/[/?#]/)[0];
  return h.toLowerCase();
}

// A resume is valid only if the queued handles still match the requested list.
const sameHandles = (queue, handles) =>
  Array.isArray(queue) && Array.isArray(handles) && queue.length === handles.length &&
  queue.every((it, i) => it && it.handle === handles[i]);

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Capture: count / export / reset
// ---------------------------------------------------------------------------
const $count = $("count");
const $ops = $("ops");
const $status = $("status");
const $export = $("export");
const $reset = $("reset");

const setStatus = (m) => { $status.textContent = m || ""; };

function dedup(records) {
  const seen = new Set();
  const out = [];
  for (const r of records) {
    if (!r || seen.has(r.id)) continue;
    seen.add(r.id);
    out.push(r);
  }
  return out;
}

async function refresh() {
  const { [DATA_KEY]: captured = [] } = await chrome.storage.local.get(DATA_KEY);
  const unique = dedup(captured);
  $count.textContent = unique.length;
  $export.disabled = unique.length === 0;
  const byOp = {};
  for (const r of unique) {
    const o = (r.__import_meta && r.__import_meta.operation) || "unknown";
    byOp[o] = (byOp[o] || 0) + 1;
  }
  $ops.textContent = Object.entries(byOp)
    .sort((a, b) => b[1] - a[1])
    .map(([o, n]) => `${o}: ${n}`)
    .join("  ·  ");
  return unique;
}

$export.addEventListener("click", async () => {
  setStatus("");
  const unique = await refresh();
  if (!unique.length) { setStatus("Nothing to export yet."); return; }
  // Download = the simplified, analysis-friendly schema (4CAT gets raw instead).
  const posts = projectAll(unique); // one record per post (retweets fold into their original)
  const ndjson = posts.map((r) => JSON.stringify(r)).join("\n") + "\n";
  const blob = new Blob([ndjson], { type: "application/x-ndjson" });
  const url = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  try {
    await chrome.downloads.download({ url, filename: `x_capture_${stamp}.ndjson`, saveAs: false });
    setStatus(`Exported ${posts.length} posts.`);
  } catch (e) {
    setStatus(`Export failed: ${e && e.message ? e.message : e}`);
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 15000);
  }
});

$reset.addEventListener("click", async () => {
  if (!confirm("Clear all captured posts? This cannot be undone.")) return;
  await chrome.runtime.sendMessage({ type: "xcap_reset" });
  setStatus("Cleared.");
  refresh();
});

// ---------------------------------------------------------------------------
// Autopilot + session-end controls
// ---------------------------------------------------------------------------
const ap = {
  mode: $("ap-mode"), manualWrap: $("ap-manual-wrap"), manual: $("ap-manual"),
  profilesWrap: $("ap-profiles-wrap"), profiles: $("ap-profiles"), perTarget: $("ap-pertarget"),
  time: $("ap-time"), target: $("ap-target"),
  threads: $("ap-threads"), threadDwell: $("ap-threaddwell"),
  dwell: $("ap-dwell"), cadence: $("ap-cadence"), rate: $("ap-rate"),
  feasibility: $("ap-feasibility"),
  toggle: $("ap-toggle"), status: $("ap-status"),
};

const MAX_RATE = 100; // hard cap on posts/min (matches autopilot.js)

// Live sanity check on the run-until settings: required pace, over-cap, > 2 h.
function checkFeasibility() {
  const time = clampNum(ap.time.value, 1, 600, 30);
  const target = clampNum(ap.target.value, 0, 1000000, 0);
  const reqPerMin = clampNum(ap.rate.value, 0, 60, 20);
  const warn = [];
  // ~30 posts arrive per feed request, so the request cap sets a rough ceiling
  // on posts/hour; it only bites when the scroll outruns it.
  const ceilPerHour = reqPerMin > 0 ? reqPerMin * 30 * 60 : 0;
  if (ceilPerHour > 0 && target > 0 && target / (time / 60) > ceilPerHour)
    warn.push(`${target} posts in ${time} min needs ~${Math.round(target / (time / 60))}/h, over the ~${ceilPerHour}/h the ${reqPerMin} req/min cap allows — it'll run long or fall short.`);
  if (time > 120) warn.push(`Session over 2 h — long automated runs raise account risk.`);
  if (target > 0) {
    const required = target / time;
    const minTime = Math.ceil(target / MAX_RATE);
    if (minTime > 120) {
      warn.push(`${target} posts may exceed the 2-hour cap even at a brisk pace — it'll stop at 2 h.`);
    } else if (required > MAX_RATE) {
      warn.push(`~${Math.round(required)}/min needed for ${time} min (over ${MAX_RATE}/min) — it'll keep going past ${time} min to reach ${target} (up to 2 h).`);
    }
  }
  if (warn.length) {
    ap.feasibility.textContent = "⚠ " + warn.join(" ");
    ap.feasibility.className = "warn";
  } else if (target > 0) {
    ap.feasibility.textContent = `Runs until ${target} posts (~${time} min at ${Math.round(target / time)}/min; extends up to 2 h if the feed is slower).`;
    ap.feasibility.className = "hint";
  } else {
    ap.feasibility.textContent = reqPerMin > 0
      ? `No post target — steady scroll for ${time} min, capped at ${reqPerMin} feed requests/min (~${ceilPerHour} posts/h ceiling).`
      : `No post target — steady scroll for ${time} min (request rate uncapped).`;
    ap.feasibility.className = "hint";
  }
}
const se = {
  exportOn: $("se-export"), destWrap: $("se-dest-wrap"),
  destFile: $("se-dest-file"), dest4cat: $("se-dest-4cat"),
  fourcatWrap: $("se-4cat-wrap"), fourcatUrl: $("se-4caturl"),
  restart: $("se-restart"),
};

const clampNum = (v, lo, hi, d) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d;
};
const toggleModeWraps = () => {
  ap.manualWrap.style.display = ap.mode.value === "manual" ? "block" : "none";
  ap.profilesWrap.style.display = ap.mode.value === "profiles" ? "block" : "none";
};
const toggle4cat = () => { se.fourcatWrap.style.display = se.dest4cat.checked ? "block" : "none"; };
const toggleDest = () => { se.destWrap.style.display = se.exportOn.checked ? "block" : "none"; toggle4cat(); };

function readCfg() {
  return {
    mode: ap.mode.value,
    manualTopics: ap.manual.value.split(/[\n,]/).map((s) => s.trim()).filter(Boolean),
    profileHandles: [...new Set(ap.profiles.value.split(/[\n,]/).map(normHandle).filter(Boolean))],
    perProfileTarget: clampNum(ap.perTarget.value, 1, 100000, 100),
    pacingVersion: PACING_VERSION,
    feedReqPerMin: clampNum(ap.rate.value, 0, 60, 20),
    dwellSec: clampNum(ap.dwell.value, 10, 3600, 90),
    cadenceSec: clampNum(ap.cadence.value, 0.5, 30, 1),
    sessionMaxMin: clampNum(ap.time.value, 1, 600, 30),
    targetPosts: clampNum(ap.target.value, 0, 1000000, 0),
    threadDives: ap.threads.checked,
    threadDwellSec: clampNum(ap.threadDwell.value, 5, 300, 20),
    autoExport: se.exportOn.checked,
    exportDest: se.dest4cat.checked ? "4cat" : "file",
    fourcatUrl: se.fourcatUrl.value.trim() || "http://localhost:4444",
    autoRestart: se.restart.checked,
  };
}

// Pacing defaults are versioned. A saved config otherwise pins whatever the
// popup was last showing, so shipping a new default (1 s cadence, 20 req/min)
// would silently do nothing until the fields were edited by hand. Bump
// PACING_VERSION whenever a pacing default changes: the stored values for those
// fields are dropped once, the rest of the config is left alone.
const PACING_VERSION = 2;
const PACING_FIELDS = ["cadenceSec", "feedReqPerMin"];

function migrateCfg(cfg = {}) {
  if (cfg.pacingVersion === PACING_VERSION) return cfg;
  const out = { ...cfg, pacingVersion: PACING_VERSION };
  for (const k of PACING_FIELDS) delete out[k];
  chrome.storage.local.set({ [CFG_KEY]: out });
  return out;
}

function applyCfg(cfg = {}) {
  if (cfg.mode) ap.mode.value = cfg.mode;
  ap.manual.value = (cfg.manualTopics || []).join("\n");
  ap.profiles.value = (cfg.profileHandles && cfg.profileHandles.length
    ? cfg.profileHandles : DEFAULT_PROFILE_HANDLES).join("\n");
  if (cfg.perProfileTarget != null) ap.perTarget.value = cfg.perProfileTarget;
  if (cfg.feedReqPerMin != null) ap.rate.value = cfg.feedReqPerMin;
  if (cfg.dwellSec) ap.dwell.value = cfg.dwellSec;
  if (cfg.cadenceSec) ap.cadence.value = cfg.cadenceSec;
  if (cfg.sessionMaxMin) ap.time.value = cfg.sessionMaxMin;
  if (cfg.targetPosts != null) ap.target.value = cfg.targetPosts;
  ap.threads.checked = cfg.threadDives === true; // off unless explicitly turned on
  if (cfg.threadDwellSec) ap.threadDwell.value = cfg.threadDwellSec;
  se.exportOn.checked = !!cfg.autoExport;
  if (cfg.exportDest === "4cat") se.dest4cat.checked = true;
  else se.destFile.checked = true;
  if (cfg.fourcatUrl) se.fourcatUrl.value = cfg.fourcatUrl;
  se.restart.checked = !!cfg.autoRestart;
  toggleModeWraps();
  toggleDest();
  checkFeasibility();
}

async function activeTabId() {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    return tabs[0] && tabs[0].id;
  } catch {
    return null;
  }
}

async function renderAutopilot() {
  const { [RUN_KEY]: run } = await chrome.storage.local.get(RUN_KEY);
  const running = !!(run && run.running);
  ap.toggle.textContent = running ? "■ Stop autopilot" : "▶ Start autopilot";
  ap.toggle.classList.toggle("running", running);
  [ap.mode, ap.manual, ap.profiles, ap.perTarget, ap.time, ap.target, ap.threads, ap.threadDwell, ap.skim, ap.dwell, ap.cadence].forEach(
    (e) => (e.disabled = running)
  );
  if (running) {
    const cur = run.queue && run.queue[run.idx];
    ap.status.textContent =
      `Running · ${run.visited || 0} topic(s)` + (cur ? ` · now: ${cur.label}` : "");
  } else if (run && run.reason) {
    ap.status.textContent = (run.completed ? "Finished" : "Stopped") + `: ${run.reason}`;
  } else {
    ap.status.textContent = "";
  }
}

async function startAutopilot() {
  const cfg = readCfg();
  if (cfg.mode === "manual" && !cfg.manualTopics.length) {
    ap.status.textContent = "Add at least one topic.";
    return;
  }
  if (cfg.mode === "profiles" && !cfg.profileHandles.length) {
    ap.status.textContent = "Add at least one profile handle.";
    return;
  }

  // Need an x.com / twitter.com tab whose content script is present.
  let tab = null;
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    tab = tabs[0] || null;
  } catch {}
  const onX = tab && tab.url && /^https?:\/\/(x|twitter)\.com\//.test(tab.url);
  if (!onX) {
    ap.status.textContent = "Open an x.com tab first, then Start.";
    return;
  }

  await chrome.storage.local.set({ [CFG_KEY]: cfg });

  // Resume where the last run left off when the mode is unchanged (keep the
  // topic queue + position); otherwise start fresh. New params (dwell, target,
  // time, threads, …) apply automatically. If already over the post target, the
  // first drive tick will export & end the session.
  const prev = (await chrome.storage.local.get(RUN_KEY))[RUN_KEY];
  // Resume an interrupted run when the mode (and, for profiles, the handle list)
  // is unchanged. Profiles compare the queued handles so EDITING the list — e.g.
  // chunking the sweep across sessions — starts a fresh pass instead of replaying
  // the old list; a completed sweep also starts over rather than resuming.
  const resume =
    cfg.mode === "profiles"
      ? !!(prev && prev.mode === "profiles" && !prev.completed &&
           sameHandles(prev.queue, cfg.profileHandles))
      : !!(prev && prev.mode === cfg.mode && Array.isArray(prev.queue) && prev.queue.length > 0);
  const queue = resume
    ? prev.queue
    : cfg.mode === "manual"
      ? cfg.manualTopics.map((t) => ({
          label: t,
          url: `https://x.com/search?q=${encodeURIComponent(t)}&src=typed_query&f=live`,
        }))
      : cfg.mode === "profiles"
        ? cfg.profileHandles.map((h) => ({
            handle: h, label: "@" + h, url: "https://x.com/" + h,
          }))
        : [];
  const { [DATA_KEY]: have = [] } = await chrome.storage.local.get(DATA_KEY);
  const run = {
    running: true,
    startedAt: Date.now(),
    startCount: Array.isArray(have) ? have.length : 0, // baseline for posts/min
    tabId: tab.id,
    mode: cfg.mode,
    queue,
    idx: resume ? prev.idx || 0 : 0,
    visited: resume ? prev.visited || 0 : 0,
    reason: null,
    seededFrom: resume ? !!prev.seededFrom : false,
  };
  await chrome.storage.local.set({ [RUN_KEY]: run });

  try {
    await chrome.tabs.sendMessage(tab.id, { type: "autopilot_start" });
  } catch {
    // The content script isn't live in this tab (typically the extension was
    // updated but the tab wasn't reloaded). Reload it — the run state we just
    // saved makes autopilot resume on its own once the fresh scripts load.
    try {
      ap.status.textContent = "Reloading the tab to start…";
      await chrome.tabs.reload(tab.id);
    } catch {
      run.running = false;
      run.reason = "couldn't start — reload the x.com tab and try again";
      await chrome.storage.local.set({ [RUN_KEY]: run });
      ap.status.textContent = "Reload the x.com tab, then Start.";
    }
  }
  renderAutopilot();
}

async function stopAutopilot() {
  const { [RUN_KEY]: run = {} } = await chrome.storage.local.get(RUN_KEY);
  run.running = false;
  run.reason = "stopped from popup";
  await chrome.storage.local.set({ [RUN_KEY]: run });
  const tabId = await activeTabId();
  try {
    if (tabId != null) await chrome.tabs.sendMessage(tabId, { type: "autopilot_stop" });
  } catch {
    /* tab may not be x.com — the flag is already cleared */
  }
  renderAutopilot();
}

ap.mode.addEventListener("change", toggleModeWraps);
ap.time.addEventListener("input", checkFeasibility);
ap.target.addEventListener("input", checkFeasibility);
ap.rate.addEventListener("input", checkFeasibility);
se.exportOn.addEventListener("change", toggleDest);
se.destFile.addEventListener("change", toggle4cat);
se.dest4cat.addEventListener("change", toggle4cat);
ap.toggle.addEventListener("click", async () => {
  const { [RUN_KEY]: run } = await chrome.storage.local.get(RUN_KEY);
  if (run && run.running) stopAutopilot();
  else startAutopilot();
});

// Persist session-end settings live, so a session already running picks up the
// latest choices when it finishes.
[se.exportOn, se.destFile, se.dest4cat, se.fourcatUrl, se.restart].forEach((el) => {
  el.addEventListener("change", async () => {
    await chrome.storage.local.set({ [CFG_KEY]: readCfg() });
  });
});

// ---------------------------------------------------------------------------
// Init + live updates
// ---------------------------------------------------------------------------
chrome.storage.local.get(CFG_KEY).then((o) => applyCfg(migrateCfg(o[CFG_KEY] || {})));
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  if (changes[DATA_KEY]) refresh();
  if (changes[RUN_KEY]) renderAutopilot();
});
refresh();
renderAutopilot();
