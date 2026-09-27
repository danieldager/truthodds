// autopilot.js — OPTIONAL active-collection driver (isolated world, document_idle).
//
// ⚠️ This is the one part of Zeezeef that is NOT passive. When started, it
// scrolls the page and navigates between topics on its own to drive the feed so
// the passive capture pipeline (injected.js → content.js → background.js) sees
// more posts. That is automation; X's ToS restricts it and it raises the risk of
// rate-limits / suspension on the logged-in account. It is OFF by default and
// fully separate from the passive core — start it explicitly from the popup.
//
// Design: a resumable state machine. Switching topics is a real navigation, so
// the content script re-injects each time; runtime state lives in
// chrome.storage.local and the driver resumes on load. The run is bound to one
// tab id so it can't hijack the user's other x.com tabs.

(() => {
  if (window.top !== window) return; // top frame only — never drive iframes

  const RUN_KEY = "autopilot_run"; // runtime state (running, queue, idx, counters)
  const CFG_KEY = "autopilot_cfg"; // user settings
  const BADGE_ID = "__xcap_autopilot_badge";
  const TRENDING_URL = "https://x.com/explore/tabs/trending";
  const HOME_URL = "https://x.com/home";
  const HOME_CHANCE = 0.5; // odds of detouring to the home feed between topics
  // Home mode scrolls in passes: scroll down for a pass, then reload /home to
  // reset the feed to the top (fresh posts; global dedup drops what we've seen).
  const HOME_PASS_MIN_MS = 8 * 60000;
  const HOME_PASS_MAX_MS = 12 * 60000;

  const DEFAULTS = {
    mode: "home", // "home" | "trending" | "manual" | "current" | "profiles"
    manualTopics: [],
    profileHandles: [], // "profiles" mode: X handles to sweep (bare, normalized)
    perProfileTarget: 1000, // "profiles" mode: own original posts to collect / profile
    feedReqPerMin: 20, // REQUEST-RATE CEILING — feed fetches/min (0 = uncapped). X's own
    // headers report HomeTimeline at 500 per 15-min window (33/min); observed use
    // during a real session is 0.2-0.5/min, so this is headroom, not a throttle.
    dwellSec: 90, // scroll each topic this long
    cadenceSec: 1, // base seconds between scroll steps (the real rate lever)
    sessionMaxMin: 30, // time budget for the whole run
    targetPosts: 0, // finish once this many posts are captured (0 = no count target)
    threadDives: false, // occasionally dip into a comment thread, then return Off: the urn is what a feed-scroller sees, not thread replies.
    threadDwellSec: 20, // how long to scroll inside a thread
    // session-end actions (performed by background.js)
    autoExport: false, // dump captured posts when a session finishes
    exportDest: "file", // "file" | "4cat"
    fourcatUrl: "http://localhost:4444",
    autoReset: false, // manual Reset button only — a finished session never auto-wipes the store
    autoRestart: false, // start another session automatically
  };
  const TRENDS_PER_CYCLE = 20; // trends to read per trending cycle
  const DATA_KEY = "captured"; // capture store (read for the post target)
  const MIN_THREAD_REPLIES = 10; // only dive into posts with at least this many replies
  const PROFILE_STALL_CHECKS = 6; // profiles: end a profile after this many no-progress polls
  // Profiles pacing is RATE control, not behavioral mimicry. X caps the
  // UserTweets GraphQL endpoint per rolling 15-min window (~50 req), so the sweep
  // is throttled by request rate: a slow gap between pagination scrolls, a rest
  // between profiles, and a live governor that reads X's own x-rate-limit-remaining
  // header (captured by injected.js) and pauses until the window resets. The
  // header governor is the real safeguard — it reacts to X's ACTUAL budget rather
  // than trusting a hardcoded cap X churns every few weeks; the gap/rest below are
  // the fallback that keeps us well under the cap when a header is unavailable.
  const PROFILE_STEP_MIN_MS = 2500; // profiles: min gap between pagination scrolls
  const PROFILE_STEP_MAX_MS = 5000; // profiles: max gap (jittered so it's not one fixed rate)
  const PROFILE_MAX_STEPS = 200; // profiles: per-profile scroll backstop (target/stall/governor stop sooner; ~200 reaches a 1000-post target)
  const PROFILE_COOLDOWN_MIN_MS = 45000; // profiles: rest between profiles (min)
  const PROFILE_COOLDOWN_MAX_MS = 90000; // profiles: rest between profiles (max)
  const RATELIMIT_KEY = "ratelimits"; // per-endpoint {remaining,reset,limit,at}, written by background from headers
  const RL_ENDPOINT = "UserTweets"; // the profile-timeline endpoint we pace against
  const RL_MIN_REMAINING = 10; // pause the sweep when this few UserTweets requests remain in the window

  let aborted = false; // module-local instant abort (badge / stop message)
  let busy = false; // guard against overlapping drive() runs
  let paceFactor = 1; // adaptive-speed multiplier on pauses (<1 = faster when behind)

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const jitter = (ms, f = 0.4) => Math.max(250, ms * (1 + (Math.random() * 2 - 1) * f));
  const getLocal = (k) =>
    chrome.storage.local.get(k).then((o) => o[k]).catch(() => undefined);
  const setRun = (run) => chrome.storage.local.set({ [RUN_KEY]: run }).catch(() => {});

  // --- rate-limit governor ---------------------------------------------------
  // The master pacing control. injected.js reads X's x-rate-limit-remaining /
  // -reset headers off the UserTweets responses and background stamps them into
  // RATELIMIT_KEY. When the window is nearly spent, hold the sweep until it
  // resets (polling so a mid-wait Stop still aborts). Returns "abort" or null.
  async function waitForRateBudget() {
    for (;;) {
      if (aborted) return "abort";
      const rl = (await getLocal(RATELIMIT_KEY)) || {};
      const b = rl[RL_ENDPOINT];
      if (!b || typeof b.remaining !== "number" || b.remaining > RL_MIN_REMAINING) return null;
      const waitMs = (b.reset || 0) * 1000 - Date.now();
      if (waitMs <= 0) return null; // window already reset — proceed
      showBadge(`rate-limit cooldown · ${Math.ceil(Math.min(waitMs, 15 * 60000) / 1000)}s`);
      await sleep(Math.min(waitMs + 1500, 5000)); // re-check at least every 5 s
    }
  }

  // A rest between profiles so per-profile request bursts don't stack inside one
  // rate-limit window. Abortable; the badge counts the rest down.
  async function interProfileCooldown() {
    const end = Date.now() + PROFILE_COOLDOWN_MIN_MS +
      Math.random() * (PROFILE_COOLDOWN_MAX_MS - PROFILE_COOLDOWN_MIN_MS);
    while (Date.now() < end) {
      if (aborted) return;
      showBadge(`cooling down · ${Math.ceil((end - Date.now()) / 1000)}s`);
      await sleep(1000);
    }
  }

  // --- request-rate pacer ----------------------------------------------------
  // The only throttle. X meters the NUMBER of timeline GraphQL calls per window,
  // not how human the scrolling looks, so the run is capped on requests: hold
  // whenever cfg.feedReqPerMin fetches have already gone out in the last 60 s,
  // and only until the oldest one ages out. A hold is therefore under a minute by
  // construction — no multi-minute idles. content.js keeps the ledger (shared
  // isolated world); no ledger means no requests seen, so no hold.
  async function waitForReqBudget(cfg) {
    const cap = cfg.feedReqPerMin || 0;
    if (cap <= 0) return null;
    for (;;) {
      const now = Date.now();
      const log = (window.__xcapFeedReqs || []).filter((t) => now - t < 60000);
      if (log.length < cap) break;
      if (aborted) return "abort";
      if (isChallenged()) {
        stop("hit a verification / rate-limit challenge — stopped to protect the account");
        return "abort";
      }
      badgeIdle = Math.max(1, Math.ceil((60000 - (now - log[0])) / 1000));
      drawBadge();
      await sleep(1000);
    }
    if (badgeIdle) { badgeIdle = 0; drawBadge(); }
    return null;
  }

  // --- human-like motion -----------------------------------------------------
  // Standard normal (Box–Muller) → log-normal pause sampler. Human inter-action
  // delays follow a log-normal distribution (a long right tail of reading
  // pauses), NOT the uniform jitter a naive bot uses — a documented behavioral
  // tell. Median tracks cadenceSec; clamped to sane bounds.
  function gaussian() {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function humanPause(cfg) {
    const median = Math.max(400, cfg.cadenceSec * 1000);
    return Math.min(30000, Math.max(300, median * Math.exp(0.6 * gaussian())));
  }

  // Eased scroll (accelerate → peak → decelerate) as many small steps, so the
  // scroll-event stream has a human velocity profile rather than one jump.
  async function smoothScrollBy(dy, opts = {}) {
    const fast = !!opts.fast;
    const steps = fast ? 5 + Math.floor(Math.random() * 4) : 10 + Math.floor(Math.random() * 10);
    const dur = fast ? 90 + Math.random() * 160 : 220 + Math.random() * 500; // total ms
    let moved = 0;
    for (let i = 1; i <= steps; i++) {
      if (aborted) return;
      const t = i / steps;
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // easeInOutQuad
      const target = Math.round(dy * eased);
      window.scrollBy(0, target - moved);
      moved = target;
      await sleep((dur / steps) * (0.6 + Math.random() * 0.8));
    }
  }

  // Ask the background worker for our own tab id (content scripts can't read it).
  function myTabId() {
    return chrome.runtime
      .sendMessage({ type: "autopilot_whoami" })
      .then((r) => (r && r.tabId != null ? r.tabId : null))
      .catch(() => null);
  }

  // React to popup start/stop without needing a reload.
  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg) return false;
    if (msg.type === "autopilot_start") { aborted = false; drive(); }
    if (msg.type === "autopilot_stop") { aborted = true; stop("stopped from popup"); }
    return false;
  });

  // Resume after each topic navigation.
  drive();

  async function drive() {
    if (busy) return;
    busy = true;
    try {
      await step();
    } catch (_) {
      /* never throw into the page */
    } finally {
      busy = false;
    }
  }

  async function step() {
    const run = await getLocal(RUN_KEY);
    if (!run || !run.running) { removeBadge(); return; }
    const cfg = { ...DEFAULTS, ...((await getLocal(CFG_KEY)) || {}) };

    // Only the tab that started the run drives (avoid hijacking other tabs).
    const myId = await myTabId();
    if (run.tabId != null && myId != null && run.tabId !== myId) return;

    // Seed the badge's session count from storage (the content script re-injects
    // on every navigation, so the in-memory count resets each topic).
    badgeCount = Math.max(0, (await capturedCount()) - (run.startCount || 0));

    // Show the Stop badge immediately on the driving tab — including during the
    // navigate-to-Trends step, before any scrolling starts.
    showBadge(currentLabel(run, cfg));

    // Stop conditions + stale-run guard. With a post target the time extends to
    // a 2 h safety cap, so it can keep going (steadily) until the target is met.
    const capMin = cfg.targetPosts > 0 ? Math.max(cfg.sessionMaxMin, 120) : cfg.sessionMaxMin;
    if (Date.now() - run.startedAt > capMin * 60000)
      return finishSession(cfg.targetPosts > 0 ? "safety cap (2 h) reached" : "time budget reached");
    if (cfg.targetPosts > 0 && (await capturedCount()) >= cfg.targetPosts)
      return finishSession("post target reached");
    if (isChallenged()) return stop("hit a verification / rate-limit challenge — stopped to protect the account");

    // Profiles sweep: a dedicated single-pass driver (navigate each profile,
    // scroll to a per-account target, advance). It owns its own nav + finish.
    if (cfg.mode === "profiles") return profilesStep(run, cfg);

    // Trending: seed the queue from the Explore → Trending tab (a dense list of
    // short trend terms). Navigate there once — `seededFrom` guards against a
    // redirect loop — read the cells, then cycle each as a plain /search?q=term.
    if (cfg.mode === "trending" && (!run.queue || run.queue.length === 0)) {
      if (!onTrendingTab() && !run.seededFrom) {
        run.seededFrom = true;
        await setRun(run);
        return go(TRENDING_URL);
      }
      const trends = await readTrends(TRENDS_PER_CYCLE);
      if (!trends.length)
        return stop("couldn't read the Trending tab — reload x.com and retry, or use Manual mode");
      run.queue = trends;
      run.idx = 0;
      run.seededFrom = false;
      await setRun(run);
      return go(trends[0].url);
    }

    // Home mode: stay on the home timeline and scroll/dive, looping until the
    // session ends. Navigate there once if we're not already on it.
    if (cfg.mode === "home" && !onHome()) return go(HOME_URL);

    showBadge(currentLabel(run, cfg));

    // Stamp what we're currently driving so the capture pipeline attributes posts
    // correctly. Thread-dive replies have no `q` in their /status URL and inherit
    // this context — the dive is a nested SPA nav that never re-runs step(), so
    // whatever is stamped here holds for the whole dwell (e.g. dives off /home
    // stay "(home feed)" rather than leaking the last trend topic).
    run.context = contextTopic(run, cfg);
    await setRun(run);

    // Home mode scrolls in passes: drift down into older posts for 8-12 min, then
    // reload /home to reset the feed to the top and start the next pass. Long
    // enough to reach genuinely older content, short enough that fresh top-of-feed
    // posts keep arriving. Other modes use the per-topic dwell (deeper, ~4 min,
    // with a post target).
    const remainMs = Math.max(0, capMin * 60000 - (Date.now() - run.startedAt));
    const homePassMs = HOME_PASS_MIN_MS + Math.random() * (HOME_PASS_MAX_MS - HOME_PASS_MIN_MS);
    const dwellMs = cfg.mode === "home"
      ? Math.min(remainMs, homePassMs)
      : (cfg.targetPosts > 0 ? Math.max(cfg.dwellSec, 240) : cfg.dwellSec) * 1000;
    // Keep dives happening across a long home pass (≈2 per 90 s of scrolling)
    // instead of the fixed 2-per-call used for short topic dwells.
    const maxDives = cfg.mode === "home" ? Math.max(2, Math.round(dwellMs / 90000) * 2) : 2;
    const res = await scrollPage(dwellMs, cfg, { allowDives: true, checkRun: true, maxDives });
    if (aborted || res === "abort" || res === "stop") return;

    // Re-read: the user may have stopped mid-dwell.
    const r = await getLocal(RUN_KEY);
    if (!r || !r.running) { removeBadge(); return; }
    r.visited = (r.visited || 0) + 1;

    if (cfg.mode === "current") return finishSession("done (single page)");

    // Home mode: reload the home feed and keep scrolling (fresh top-of-feed posts
    // each pass; global dedup drops repeats). The stop checks at the top of the
    // next step() end the session on time / target / challenge.
    if (cfg.mode === "home") { await setRun(r); return go(HOME_URL); }

    // Home-feed interleave: between topics, ~50% of the time detour to the home
    // timeline (the most representative feed) before moving on, unless we just
    // scrolled it. idx is left untouched so the topic queue resumes where it was
    // after the home visit. Global dedup means re-seen tweets never double-count.
    if (!onHome() && Math.random() < HOME_CHANCE) {
      await setRun(r); // persists the bumped visit count; idx unchanged
      return go(HOME_URL);
    }

    let next = r.idx + 1;
    if (next >= r.queue.length) {
      // Loop: keep collecting until the time budget or post target is reached.
      if (cfg.mode === "trending") {
        r.queue = []; r.idx = 0; r.seededFrom = false; await setRun(r);
        return go(TRENDING_URL); // refresh trends and sweep again
      }
      next = 0; // manual: restart the list
    }
    r.idx = next;
    await setRun(r);
    go(r.queue[next].url);
  }

  // Profiles mode: a SINGLE PASS over run.queue = [{url, handle, label}]. For
  // each profile, land on its page and scroll (reusing the human-paced
  // scrollPage) until ~perProfileTarget of THAT handle's own original posts
  // (screen_name == handle, is_reply false) are captured, or the feed ends /
  // stalls / a per-profile dwell cap is hit; then advance. After the last
  // profile, finishSession. Unlike "manual", the list is NOT looped.
  async function profilesStep(run, cfg) {
    if (!run.queue || run.queue.length === 0)
      return stop("no profiles to sweep — add handles in the popup and restart");
    const item = run.queue[run.idx];
    if (!item) return finishSession("done (all profiles)");
    const handle = item.handle;

    // Stamp the attribution context first so even the initial UserTweets fetch
    // (which can fire before the dwell starts) is bucketed to this handle, then
    // navigate to the profile if we're not already on it.
    run.context = "@" + handle;
    await setRun(run);
    if (!onProfile(handle)) return go(item.url);

    showBadge(item.label);
    const target = Math.max(1, cfg.perProfileTarget || 100);

    // Per-profile dwell cap so a low-volume / rate-limited account can't hang the
    // whole sweep; the stall check below usually ends a dead profile sooner. Deep
    // (1000-post) targets with 15-min rate cooldowns need a generous ceiling —
    // fast profiles still advance early on the target/stall gate, so this only
    // bounds a genuinely rate-limited deep profile (resumed on a later chunk).
    const dwellMs = Math.max(60, cfg.dwellSec) * 1000 * 12;

    // Target + stall gate, polled from inside scrollPage's periodic check. The
    // badge shows live per-profile progress (N / target).
    let lastCount = await handleCount(handle), stall = 0;
    const gate = async () => {
      const c = await handleCount(handle);
      showBadge(`${item.label} · ${c}/${target}`);
      if (c >= target) return "target";
      if (c > lastCount) { lastCount = c; stall = 0; }
      else if (++stall >= PROFILE_STALL_CHECKS) return "stalled";
      return null;
    };

    // Paced sweep, not a crawl: fastScroll skips the reading-pause beat (request
    // RATE is the lever, not behavioural mimicry — per the X
    // detection research) but spaces pagination scrolls slowly and obeys the rate
    // governor inside scrollPage. No thread dives (they capture OTHER users'
    // replies, off-target for a per-account set). Hold here first if the rate
    // window is already nearly spent.
    if ((await waitForRateBudget()) === "abort") return;
    const res = await scrollPage(dwellMs, cfg, {
      allowDives: false, checkRun: true, profileGate: gate, fastScroll: true,
    });
    if (aborted || res === "abort" || res === "stop") return;

    // Re-read (the user may have stopped mid-dwell) and advance. Any non-abort
    // result — target / stalled / exhausted / done — means this profile is done.
    const r = await getLocal(RUN_KEY);
    if (!r || !r.running) { removeBadge(); return; }
    r.visited = (r.visited || 0) + 1;
    const next = r.idx + 1;
    if (next >= r.queue.length) { await setRun(r); return finishSession("done (all profiles)"); }
    r.idx = next;
    await setRun(r);
    // Pause if the rate window is spent, then rest before the next profile so
    // per-profile request bursts don't stack inside one 15-min window.
    if ((await waitForRateBudget()) === "abort") return;
    await interProfileCooldown();
    if (aborted) return;
    go(r.queue[next].url);
  }

  // Scroll the current page for `durationMs`. Every beat is the same STEADY scan
  // (smooth step of ~0.6 viewport + a cadenceSec gap → continuous downward
  // motion); the run is held only by the per-minute feed-request cap. The one
  // exception is a STOP beat, which exists solely to open a comment thread and
  // only when thread dives are on. Periodic abort / challenge / target checks.
  // Returns "abort" | "stop" | "exhausted" | "done".
  async function scrollPage(durationMs, cfg, opts = {}) {
    let end = Date.now() + durationMs;
    const scroller = document.scrollingElement || document.documentElement;
    let lastH = 0, stale = 0, n = 0, dives = 0;
    const maxDives = opts.maxDives || 2;

    // Steady drift only. The old mix of reading stops, skim bursts and back-
    // nudges was behavioural camouflage; it bought nothing against a
    // request-count limiter and made the collection rate lumpy and slow. The
    // one surviving branch exists because thread dives hang off it.
    const P_STOP = cfg.threadDives ? 0.12 : 0; // pause to read / enter a thread

    // End-of-feed tracking after a downward move; true once exhausted.
    const atEnd = () => {
      const h = scroller.scrollHeight;
      if (window.scrollY + window.innerHeight >= h - 200 && h === lastH) stale++;
      else stale = 0;
      lastH = h;
      return stale >= 4;
    };

    while (Date.now() < end) {
      if (aborted) return "abort";

      // Hold here if the last minute already spent cfg.feedReqPerMin fetches.
      // Idle time doesn't count against the dwell — this paces the run, it
      // shouldn't shorten the scroll.
      {
        const t0 = Date.now();
        if ((await waitForReqBudget(cfg)) === "abort") return "abort";
        end += Date.now() - t0;
      }

      if (n % 6 === 0) {
        if (isChallenged()) {
          stop("hit a verification / rate-limit challenge — stopped to protect the account");
          return "stop";
        }
        if (opts.checkRun) {
          const r = await getLocal(RUN_KEY);
          if (!r || !r.running) return "stop";
          const capMin = cfg.targetPosts > 0 ? Math.max(cfg.sessionMaxMin, 120) : cfg.sessionMaxMin;
          if (Date.now() - r.startedAt > capMin * 60000) {
            finishSession(cfg.targetPosts > 0 ? "safety cap (2 h) reached" : "time budget reached");
            return "stop";
          }
          const have = await capturedCount();
          if (cfg.targetPosts > 0 && have >= cfg.targetPosts) {
            finishSession("post target reached");
            return "stop";
          }
          // posts/min so far → adapt pace toward the target, never above the cap.
          const elapsedMin = Math.max(0.2, (Date.now() - r.startedAt) / 60000);
          const sessionCount = Math.max(0, have - (r.startCount || 0)); // captured this session
          const rate = sessionCount / elapsedMin; // posts/min this session
          if (cfg.targetPosts > 0) {
            const remainMin = Math.max(0.2, cfg.sessionMaxMin - elapsedMin);
            const required = (cfg.targetPosts - have) / remainMin; // posts/min still needed
            // Stay near a natural pace — catch up by scrolling longer/deeper
            // (longer dwell, more topics), not by scrolling unnaturally fast.
            paceFactor = required > 0 ? Math.min(1.4, Math.max(0.7, rate / required)) : 1.2;
          } else {
            paceFactor = 1; // no target → steady human pace
          }
          badgeCount = sessionCount;
          updateBadgeRate(rate);
        }
        // Profiles mode: per-profile target / stall gate. "target" | "stalled"
        // ends this profile so the driver advances to the next one.
        if (opts.profileGate) {
          const g = await opts.profileGate();
          if (g) return g;
        }
      }

      // Profiles PACED SWEEP: scroll to paginate the timeline, but throttled by
      // request RATE, not human mimicry — no log-normal pauses, skim bursts, back-
      // nudges, or thread dives. Each step waits a slow jittered gap; the rate
      // governor holds until the UserTweets window resets when it's nearly spent;
      // PROFILE_MAX_STEPS backstops a runaway feed. Challenge / rate-limit halt
      // still fires immediately to protect the account.
      if (opts.fastScroll) {
        if (isChallenged()) {
          stop("hit a verification / rate-limit challenge — stopped to protect the account");
          return "stop";
        }
        await smoothScrollBy(window.innerHeight * 1.1, { fast: true });
        n++;
        if (atEnd()) return "exhausted";
        if (n >= PROFILE_MAX_STEPS) return "exhausted"; // per-profile pagination backstop
        if ((await waitForRateBudget()) === "abort") return "abort";
        await sleep(PROFILE_STEP_MIN_MS + Math.random() * (PROFILE_STEP_MAX_MS - PROFILE_STEP_MIN_MS));
        continue;
      }

      const roll = Math.random();

      // STOP — pause to read; sometimes drop into a comment thread.
      if (roll < P_STOP) {
        const canDive =
          opts.allowDives && cfg.threadDives && dives < maxDives &&
          end - Date.now() > cfg.threadDwellSec * 1000 + 5000;
        if (canDive && Math.random() < 0.5) {
          const t0 = Date.now();
          const dove = await threadDive(cfg);
          if (aborted) return "abort";
          if (dove) { dives++; end += Date.now() - t0; } // don't let it eat scroll time
        } else {
          await sleep(humanPause(cfg) * paceFactor); // reading stop
        }
        n++;
        continue;
      }

      // STEADY scan — the whole loop, now: one smooth step of a bit over half a
      // viewport, then a gap of cfg.cadenceSec (+/-15% so it isn't a metronome).
      // Even and predictable; how many posts that yields is whatever the feed
      // gives, bounded only by the request cap above.
      await smoothScrollBy(window.innerHeight * (0.5 + Math.random() * 0.2));
      n++;
      if (atEnd()) return "exhausted";
      await sleep(cfg.cadenceSec * 1000 * (0.85 + Math.random() * 0.3) * paceFactor);
    }
    return "done";
  }

  // Pop into a comment thread: open a visible post's conversation (SPA — so the
  // capture hook keeps running and the feed's scroll is restored on back), scroll
  // the replies briefly, then return. Best-effort: any failure just resumes the
  // feed. The replies are flagged downstream via `is_reply` and tagged with the
  // current topic (background falls back to the run's topic on /status/ pages).
  async function threadDive(cfg) {
    try {
      const link = pickThreadLink(MIN_THREAD_REPLIES);
      if (!link) return false; // no visible post with enough replies — skip the dive
      const fromPath = location.pathname;
      link.click(); // X's router handles the timestamp permalink → conversation
      const opened = await waitForCondition(() => /\/status\/\d+/.test(location.pathname), 4000);
      if (!opened) return false;
      await scrollPage(cfg.threadDwellSec * 1000, cfg); // scroll the replies
      if (aborted) return true;
      history.back(); // SPA back → feed, scroll position restored
      await waitForCondition(() => location.pathname === fromPath, 4000);
      await sleep(jitter(900));
      return true;
    } catch (_) {
      return false;
    }
  }

  // Pick a visible post with at least `minReplies` replies (worth diving into a
  // real conversation), and return its timestamp permalink.
  function pickThreadLink(minReplies) {
    const arts = Array.from(document.querySelectorAll('article[data-testid="tweet"]'));
    const cands = [];
    for (const art of arts) {
      const r = art.getBoundingClientRect();
      if (!(r.top > 80 && r.bottom < window.innerHeight - 80)) continue; // in view
      if (replyCount(art) < minReplies) continue;
      const t = art.querySelector('a[href*="/status/"] time');
      const link = t ? t.closest("a") : art.querySelector('a[href*="/status/"]');
      if (link) cands.push(link);
    }
    return cands.length ? cands[Math.floor(Math.random() * cands.length)] : null;
  }

  // Reply count from a post's reply button (handles "1.2K"-style counts).
  function replyCount(article) {
    const btn = article.querySelector('[data-testid="reply"]');
    if (!btn) return 0;
    const txt = ((btn.getAttribute("aria-label") || "") + " " + (btn.textContent || "")).replace(/,/g, "");
    const m = /([\d.]+)\s*([km])?/i.exec(txt);
    if (!m) return 0;
    let n = parseFloat(m[1]) || 0;
    if (/k/i.test(m[2] || "")) n *= 1e3;
    if (/m/i.test(m[2] || "")) n *= 1e6;
    return n;
  }

  function waitForCondition(fn, timeoutMs) {
    return new Promise((res) => {
      if (fn()) return res(true);
      const t0 = Date.now();
      const iv = setInterval(() => {
        if (fn() || Date.now() - t0 > timeoutMs) {
          clearInterval(iv);
          res(!!fn());
        }
      }, 200);
    });
  }

  function go(url) {
    try { location.assign(url); } catch (_) {}
  }

  // Stop the instant X challenges the session. Pushing through a captcha /
  // verification is exactly what escalates a flag into a suspension, so we halt
  // on the first sign: a challenge URL, or a Cloudflare/Arkose human-check frame.
  function isChallenged() {
    if (/^\/(i\/flow|account\/access|login|logout|suspended)/.test(location.pathname)) return true;
    try {
      if (
        document.querySelector(
          'iframe[src*="challenges.cloudflare.com"], iframe[src*="arkoselabs"], iframe[title*="human" i]'
        )
      )
        return true;
    } catch (_) {}
    return false;
  }

  function onHome() {
    return location.pathname === "/home";
  }

  // On this handle's profile page (the main /<handle> Posts tab we navigate to).
  function onProfile(handle) {
    const p = location.pathname.replace(/^\/+|\/+$/g, "").toLowerCase();
    return p === String(handle).toLowerCase();
  }

  // Count stored ORIGINAL posts authored by `handle` — screen_name == handle
  // (case-insensitive), not a reply, and not a retweet (a retweet keeps the outlet
  // as author but isn't its own post). So the per-profile target counts genuine
  // originals and heavy-retweet accounts get scrolled deeper to reach it. Matches
  // projectSimplified's is_reply / is_retweet rules; reuses the shared screenName().
  async function handleCount(handle) {
    const store = await getLocal(DATA_KEY);
    if (!Array.isArray(store)) return 0;
    const h = String(handle).toLowerCase();
    let n = 0;
    for (const node of store) {
      if (!node || !node.legacy || node.legacy.in_reply_to_status_id_str) continue;
      if (node.legacy.retweeted_status_result) continue; // exclude retweets — originals only
      const sn = screenName(node);
      if (sn && sn.toLowerCase() === h) n++;
    }
    return n;
  }

  function onTrendingTab() {
    return location.pathname.startsWith("/explore");
  }

  // Read the rendered trend terms (the Explore → Trending cells). Markup-
  // dependent — this is a maintenance point (like the GraphQL extractor). Each
  // becomes a plain search, exactly like clicking the trend: /search?q=<term>.
  async function readTrends(limit) {
    await waitFor('[data-testid="trend"]', 8000);
    const out = [], seen = new Set();
    document.querySelectorAll('[data-testid="trend"]').forEach((el) => {
      const name = pickTrendName((el.innerText || "").split("\n").map((s) => s.trim()).filter(Boolean));
      if (!name) return;
      const url = `https://x.com/search?q=${encodeURIComponent(name)}&src=trend_click`;
      if (seen.has(url)) return;
      seen.add(url);
      out.push({ label: name, url });
    });
    return out.slice(0, limit);
  }

  // A trend cell shows ~3 lines: a category ("… · Trending" / "Trending in …"),
  // the short trend term, and a post count. Drop the boilerplate, then take the
  // first SHORT remaining line — the term. The length cap rejects the long
  // headlines of curated news/event cards (which aren't usable as searches).
  function pickTrendName(lines) {
    const drop = /(trending|^promoted$|posts$|^\d[\d.,]*\s*(k|m)?$|^·|^\d+$)/i;
    const cand = lines.filter((l) => l && !drop.test(l));
    return cand.find((l) => l.length <= 50) || null;
  }

  function waitFor(sel, timeoutMs) {
    return new Promise((res) => {
      if (document.querySelector(sel)) return res(true);
      const t0 = Date.now();
      const iv = setInterval(() => {
        if (document.querySelector(sel) || Date.now() - t0 > timeoutMs) {
          clearInterval(iv);
          res(!!document.querySelector(sel));
        }
      }, 400);
    });
  }

  async function stop(reason) {
    aborted = true;
    const run = (await getLocal(RUN_KEY)) || {};
    run.running = false;
    run.reason = reason;
    run.stoppedAt = Date.now();
    await setRun(run);
    removeBadge();
  }

  // Natural end of a session (post target / time budget / single page done).
  // Stops, then hands the session-end actions (auto-export / reset / restart)
  // to the background worker.
  async function finishSession(reason) {
    aborted = true;
    const run = (await getLocal(RUN_KEY)) || {};
    run.running = false;
    run.reason = reason;
    run.completed = true;
    run.stoppedAt = Date.now();
    await setRun(run);
    removeBadge();
    try {
      chrome.runtime.sendMessage({ type: "autopilot_session_end" });
    } catch (_) {}
  }

  async function capturedCount() {
    const c = await getLocal(DATA_KEY);
    return Array.isArray(c) ? c.length : 0;
  }

  function currentLabel(run, cfg) {
    if (cfg.mode === "current") return "this page";
    if (onHome()) return "home feed";
    const item = run.queue && run.queue[run.idx];
    return item ? item.label : "…";
  }

  // The data topic a captured post is attributed to (distinct from the badge
  // label): the home feed is its own bucket; on a /search topic it's the trend /
  // query term; "current" mode lets the URL decide (null → background uses none).
  function contextTopic(run, cfg) {
    if (onHome()) return "(home feed)";
    if (cfg.mode === "current") return null;
    const item = run.queue && run.queue[run.idx];
    return item ? item.label : null;
  }

  // Always-visible abort affordance (the popup closes on each navigation).
  let badgeLabel = "";
  let badgeRate = 0;
  let badgeCount = 0; // tweets captured this session (shown on the badge)
  let badgeIdle = 0; // seconds left in a pacer idle (0 = scrolling)
  function showBadge(label) {
    badgeLabel = label;
    let b = document.getElementById(BADGE_ID);
    if (!b) {
      b = document.createElement("div");
      b.id = BADGE_ID;
      b.style.cssText =
        "position:fixed;z-index:2147483647;top:10px;right:10px;background:#1d9bf0;" +
        "color:#fff;font:600 12px/1 -apple-system,BlinkMacSystemFont,sans-serif;" +
        "padding:9px 13px;border-radius:999px;box-shadow:0 2px 10px rgba(0,0,0,.35);" +
        "cursor:pointer;user-select:none;max-width:60vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      b.title = "Click to stop Zeezeef autopilot";
      b.addEventListener("click", () => { aborted = true; stop("stopped from on-page button"); });
      document.documentElement.appendChild(b);
    }
    drawBadge();
  }
  function updateBadgeRate(rate) {
    badgeRate = rate;
    drawBadge();
  }
  function drawBadge() {
    const b = document.getElementById(BADGE_ID);
    if (!b) return;
    b.textContent =
      `■ Stop autopilot — ${badgeLabel} · ${badgeCount} captured` +
      (badgeRate ? ` · ~${Math.round(badgeRate)}/min` : "") +
      (badgeIdle ? ` · idling ${badgeIdle}s` : "");
  }
  function removeBadge() {
    badgeRate = 0;
    badgeIdle = 0;
    const b = document.getElementById(BADGE_ID);
    if (b) b.remove();
  }
})();
