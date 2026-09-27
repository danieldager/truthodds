// injected.js — MAIN-world fetch/XHR hook.
//
// Runs in the PAGE's own JS context (manifest `world: "MAIN"`, document_start),
// so it can see and wrap the page's window.fetch / XMLHttpRequest — which an
// MV3 isolated-world content script cannot. It monkeypatches both, watches for
// X's GraphQL responses, and posts the raw JSON text to the content-script
// bridge via window.postMessage.
//
// Hard rules (a misbehaving hook breaks x.com for the user):
//   1. NEVER throw into page code — every path swallows its own errors.
//   2. NEVER consume the body the page needs — clone() it BEFORE the page can.
//   3. Install exactly once, even if injected twice.
//
// Note: MAIN-world injection at document_start is best-effort, not a contract —
// a page inline <head> script can theoretically grab the original fetch first.
// There is no MV3 API to truly preempt it; the residual miss is accepted.

(() => {
  const MARKER = "__XCAP__";
  const MAX_BYTES = 8 * 1024 * 1024; // skip pathologically large bodies
  // Post only to our own origin, never "*", so the payload isn't broadcast.
  const ORIGIN = (window.location && window.location.origin) || "/";

  // Idempotent: if a prior copy already patched this context, do nothing.
  if (window.__XCAP_INSTALLED__) return;
  window.__XCAP_INSTALLED__ = true;

  const isTarget = (url) =>
    typeof url === "string" && url.indexOf("/i/api/graphql/") !== -1;

  // The fetch first-arg may be a string, a URL, or a Request. Normalize.
  function urlOf(input) {
    try {
      if (typeof input === "string") return input;
      if (input instanceof Request) return input.url;
      if (typeof URL !== "undefined" && input instanceof URL) return input.href;
      if (input && typeof input.url === "string") return input.url; // Request-like
      if (input != null) return String(input); // URL-like / stringifiable
    } catch (_) {}
    return "";
  }

  function emit(rawText, url, rl) {
    try {
      if (typeof rawText !== "string" || rawText.length === 0) return;
      window.postMessage({ source: MARKER, url: url || "", body: rawText, rl: rl || null }, ORIGIN);
    } catch (_) {
      /* never break the page */
    }
  }

  // Pull X's per-endpoint rate-limit headers off a GraphQL response so the sweep
  // driver can pace against the ACTUAL remaining budget. `get` is a header
  // accessor: res.headers.get for fetch, xhr.getResponseHeader for XHR. Returns
  // {remaining, reset(unix s), limit} or null when the headers aren't present.
  function readRL(get) {
    try {
      const rem = get("x-rate-limit-remaining");
      if (rem == null || rem === "") return null;
      const remaining = Number(rem);
      if (!Number.isFinite(remaining)) return null;
      return {
        remaining,
        reset: Number(get("x-rate-limit-reset")) || 0,
        limit: Number(get("x-rate-limit-limit")) || 0,
      };
    } catch (_) {
      return null;
    }
  }

  // Make a patched function report native-ish source, so any integrity check
  // the page runs on fetch.toString() is less likely to trip. Best-effort.
  function masquerade(patched, original) {
    try {
      Object.defineProperty(patched, "name", {
        value: original.name,
        configurable: true,
      });
      Object.defineProperty(patched, "length", {
        value: original.length,
        configurable: true,
      });
      const nativeStr = "function " + original.name + "() { [native code] }";
      patched.toString = () => nativeStr;
    } catch (_) {}
  }

  // ---- fetch ---------------------------------------------------------------
  // Guard on a sentinel as well as the window flag: never double-wrap fetch.
  if (typeof window.fetch === "function" && !window.fetch.__xcap) {
    const origFetch = window.fetch;
    const patchedFetch = function (...args) {
      let url = "";
      let target = false;
      try {
        url = urlOf(args[0]);
        target = isTarget(url);
      } catch (_) {}

      const p = origFetch.apply(this, args);
      if (!target) return p; // pass non-graphql calls straight through

      // Own the promise chain so the clone happens BEFORE the page's own
      // handlers can read (and lock) the body, then hand the page back the
      // SAME Response. A no-op rejection path lets failures pass through.
      return p.then((res) => {
        try {
          if (res && res.type !== "opaque" && res.status !== 0) {
            const rl = readRL((h) => res.headers && res.headers.get(h));
            const len = Number(res.headers && res.headers.get("content-length"));
            if (!(len > MAX_BYTES)) {
              const clone = res.clone(); // synchronous, before the page reads
              clone.text().then((t) => emit(t, url, rl)).catch(() => {});
            }
          }
        } catch (_) {}
        return res;
      });
    };
    masquerade(patchedFetch, origFetch);
    try {
      patchedFetch.__xcap = true;
    } catch (_) {}
    window.fetch = patchedFetch;
  }

  // ---- XMLHttpRequest ------------------------------------------------------
  // Patch the prototype methods in place (don't replace the constructor), which
  // preserves the prototype chain the page relies on.
  try {
    const XHR = XMLHttpRequest.prototype;
    if (!XHR.open.__xcap) {
      const origOpen = XHR.open;
      const origSend = XHR.send;

      XHR.open = function (method, url, ...rest) {
        try {
          this.__xcap_url = typeof url === "string" ? url : String(url || "");
        } catch (_) {
          this.__xcap_url = "";
        }
        return origOpen.call(this, method, url, ...rest);
      };

      XHR.send = function (...args) {
        try {
          if (isTarget(this.__xcap_url)) {
            this.addEventListener("load", () => {
              try {
                const rt = this.responseType;
                let text = "";
                if (rt === "" || rt === "text") {
                  text = this.responseText;
                } else if (rt === "json") {
                  // responseText throws for non-text responseType; re-serialize.
                  text = JSON.stringify(this.response);
                } else {
                  return; // arraybuffer/blob/document — not our JSON path
                }
                // responseURL reflects redirects; fall back to the open() URL.
                const rl = readRL((h) => this.getResponseHeader(h));
                emit(text, this.responseURL || this.__xcap_url, rl);
              } catch (_) {}
            });
          }
        } catch (_) {}
        return origSend.apply(this, args);
      };

      try {
        XHR.open.__xcap = true;
      } catch (_) {}
    }
  } catch (_) {}
})();
