"""Second-pass AUDIT (v3): correct the first-pass extraction — merge only logically-connected
claims, fix types (binding in text), REMOVE with recorded reasons (opinion/duplicate/merged/
not_a_claim; nothing silent), topic-label each kept claim. Checkworthy is NOT judged here —
it is a code rule at the handoff (build_verify_input). Reads the claims-review HTML payload
produced by extract_tweet_claims.py --html, normalizes each post's claims with a fast instruct
model, and writes a new payload HTML (feed to build_review_doc.py) plus a flat parquet.

  cd src && uv run python eval/scripts/claim_sourcing/normalize_tweet_claims.py \
      --payload <r5 review>.html -o <out payload>.html [--parquet <out>.parquet] [--model ...]
"""
import argparse, json, re, time, urllib.request

# Daniel 2026-07-19: an unnamed study/report is not an identifiable speaker — the content
# axis dominates (checking the content settles whether such a source exists). Attribution
# claims sourced this way fold into their content members (or become plain assertions).
_UNNAMED_SRC = re.compile(
    r"^(a|an|one|another)\s+(new\s+|recent\s+)?(study|report|survey|poll|analysis|paper)\b",
    re.I)
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.prompt_hash import prompt_hash  # noqa: E402

BASE = "https://api.deepinfra.com/v1/openai"
KEY = next(l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
           if l.startswith("DEEPINFRA_API_KEY="))
USAGE = {"in": 0, "out": 0, "cost": 0.0}

SYSTEM = """You are the claims filter. A first-pass extractor pulled claims from a news outlet's social-media POST; you receive both. Trust the extractor's recall — NEVER add a new claim. Return every input claim — kept, corrected, or dismissed with a bookkeeping category — each with exactly one category and one topic. Nothing may disappear.

Fix or dismiss what the extractor found:
1. Manufactured claims — built from promotion, captions, questions, satire, or links that state nothing: mark them "artifact".
2. Duplicates and welds — the same proposition twice (mark the lesser one "duplicate"); several independently checkable propositions in one claim (split them; a fact riding inside another claim as an aside is its own claim). Never mark a claim that belongs to a pair group as "duplicate" or "merged": pair members always stay, each judged with its own category — the attribution and its content claims answer different questions.
3. Broken pairs — one quote split across several attribution claims (one statement by one speaker is ONE attribution); a "Y, according to X" hybrid (the attribution keeps its "X said" form, content claims stay bare); a pair built on an anonymous or unnamed source, including an unnamed study or report (keep only the content as an assertion).
4. Dangling references — pronouns or descriptions a reader of the claim list cannot resolve: name them from the post, full name and role on first mention. Never invert who does what to whom; add no facts.
5. Wrong types — reported speech from an identifiable speaker typed assertion, or the outlet's own voice typed attribution.

Category, exactly one per claim, judged in this order — assign the FIRST that applies. Judge every claim individually: categories apply to claims, never to whole posts — a promotional post can still carry a checkable reported event. Judge each pair member on its own: the bare content claim inherits nothing from its attribution — a speaker's spin, characterization, prediction, or self-description is opinion even though the saying is checkable, and a content claim is checkable only if it is checkable entirely on its own. When in doubt between checkable and any other category, choose the other: the gate is strict, and only checkable claims are verified.
- "promo": the outlet promoting itself — subscriptions, merch, tune-ins, its own shows and products.
- "trivial": true or false, checking it would tell the public nothing worth knowing — inconsequential detail, trivially true or false, too general or absolute to mean anything precise, or the scaffolding of another claim (who spoke where and when, that someone responded or commented).
- "opinion": evaluation, characterization, exhortation, or prediction — a view of reality evidence cannot refute. A headline's framing or thesis is opinion when its load-bearing words are judgments rather than observable facts. Quoted opinion CONTENT belongs here — but never the attribution itself: "X said Y" with a named speaker is checkable (the saying happened or it did not) even when Y is pure opinion. Charged wording, sarcasm, and hedges never make a checkable substance opinion: judge whether evidence could refute the underlying proposition, not its tone — a cause-and-effect statement stays checkable however loaded its wording, and a hedged universal is refuted by a single counterexample.
- "unresolved": checkable in principle, but even with the post in hand a searcher could not pin down WHICH person, event, or document it is about — an event or actor with no name, place, or date. Missing detail alone is not unresolved.
- "checkable": everything that survives — a proposition evidence could confirm or refute, with real stakes and an identifiable subject. Only checkable claims are verified. A real reported event is checkable however terse; a reported death is always checkable.
Plus the bookkeeping categories for dismissals only: "duplicate", "merged", "artifact". Watch for: predictions marked checkable; an attribution treated as opinion because its content is opinion (the saying itself is checkable); a claim whose subject cannot be pinned down even with the post (unresolved, not checkable).

Topic every claim, exactly one of: "politics", "crime", "health", "education", "business", "science", "technology", "entertainment", "sports", "disaster", "lifestyle", "other" — the subject of the proposition, not the venue; health effects and nutrition are health.

Return only JSON:
{"claims": [{"claim": "...", "type": "assertion" or "attribution", "group": integer or null, "category": "<one of the categories>", "topic": "<one of the topics>"}]}"""


def _anchor_tokens(text):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", (text or "").lower()) if len(w) > 2}


def anchored(claim_text, post_text, p1_claims, quoted_text=None):
    """Text-anchoring guard: a pass-2 claim must be built from the post's (or pass-1's) own
    words — low overlap means the normalizer drifted or corrupted the text (observed: injected
    CJK tokens, 2026-07-13). True if >= 60% of the claim's content tokens appear in the pool.
    For a quote post the quoted text joins the pool: claims spell out what the author is
    endorsing or rejecting, in the quoted post's words."""
    ct = _anchor_tokens(claim_text)
    if not ct:
        return False
    pool = _anchor_tokens(post_text) | _anchor_tokens(quoted_text)
    for c in p1_claims or []:
        pool |= _anchor_tokens(c.get("c"))
    return len(ct & pool) / len(ct) >= 0.6


def _obj(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def normalize(model, post_text, claims, quoted=None):
    lines = "\n".join(
        f"{i+1}. [{c.get('t')}{' | group ' + str(c.get('g')) if c.get('g') else ''}] {c.get('c')}"
        for i, c in enumerate(claims))
    user = f"POST:\n{post_text}"
    if quoted and quoted.get("text"):
        user += (f"\n\nQUOTED POST by @{quoted.get('handle') or 'unknown'} "
                 f"(context only — the author is reacting to this):\n{quoted['text']}")
    user += f"\n\nEXTRACTED CLAIMS:\n{lines}"
    return _normalize_call(model, user)


# --voice user: ordinary-user posts (C2 CN corpus). Identical contract to SYSTEM; only the
# author framing changes — "news outlet" becomes an ordinary account, promo becomes
# self-promotion, and "the outlet's own voice" becomes the author's.
SYSTEM_USER = SYSTEM.replace(
    "A first-pass extractor pulled claims from a news outlet's social-media POST",
    "A first-pass extractor pulled claims from an ordinary user's social-media POST "
    "(an original post, reply, rant, or personal story)"
).replace(
    '- "promo": the outlet promoting itself — subscriptions, merch, tune-ins, its own shows and products.',
    '- "promo": the author promoting themselves — their products, accounts, streams, merch, or fundraising.'
).replace(
    "the outlet's own voice typed attribution",
    "the author's own voice typed attribution"
)

# --voice quote (Daniel 2026-08-25): SYSTEM_USER with the framing changed to a quote post. The
# categories, order and topics are untouched; the only added rule is that a claim belonging
# to the QUOTED post alone, which the author takes no position on, is an artifact here.
SYSTEM_QUOTE = SYSTEM_USER.replace(
    "A first-pass extractor pulled claims from an ordinary user's social-media POST "
    "(an original post, reply, rant, or personal story)",
    "A first-pass extractor pulled claims from an ordinary user's QUOTE POST — the author's "
    "own text written on top of another account's post, shown as QUOTED POST. The quoted post "
    "is context only: claims are the author's own assertions, including what the author "
    "endorses, disputes, corrects, or adds about the quoted post, spelled out. A claim that "
    "belongs to the quoted post alone, which the author takes no position on, is an "
    "\"artifact\" here — the quoted post is judged separately under its own author"
)
assert SYSTEM_QUOTE != SYSTEM_USER

PROMPT_HASH = prompt_hash(SYSTEM)   # reset in main() once --voice picks the prompt

EMPTY_SYSTEM = """The claims extractor found NO claims in this post. Explain why with ONE reason:
- "promo": the post only promotes the outlet — subscriptions, shows, merch, tune-ins.
- "opinion": the post states only the outlet's view, rhetoric, or exhortation.
- "artifact": the post states nothing — a caption, question, joke, teaser, or bare link.
- "unresolved": the post states something but no person, event, or document can be identified even with the post in hand.
- "missed-claims": the post plainly states checkable facts the extractor should have found.
Return only JSON: {"no_claim_reason": "<one of the reasons>"}"""


def judge_empty(model, post_text):
    payload = {"model": model, "temperature": 0, "max_tokens": 60,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": EMPTY_SYSTEM},
                            {"role": "user", "content": f"POST:\n{post_text}"}]}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read().decode())
                u = d.get("usage") or {}
                USAGE["in"] += u.get("prompt_tokens", 0); USAGE["out"] += u.get("completion_tokens", 0)
                USAGE["cost"] += u.get("estimated_cost") or 0
                nr = (_obj(d["choices"][0]["message"]["content"]) or {}).get("no_claim_reason")
                return nr if nr in ("promo", "opinion", "artifact", "unresolved", "missed-claims") else None
        except Exception:
            time.sleep(2 * 2 ** a)
    return None


def _normalize_call(model, user):
    thinking = "Thinking" in model
    payload = {"model": model, "temperature": 0, "max_tokens": 8000 if thinking else 1500,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    if not thinking:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read().decode())
                u = d.get("usage") or {}
                USAGE["in"] += u.get("prompt_tokens", 0); USAGE["out"] += u.get("completion_tokens", 0)
                USAGE["cost"] += u.get("estimated_cost") or 0
                return _obj(d["choices"][0]["message"]["content"])
        except Exception:
            time.sleep(2 * 2 ** a)
    return {}


def load_payload(path):
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', Path(path).read_text(), re.S)
    return json.loads(m.group(1))["posts"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", required=True, help="first-pass review HTML from extract_tweet_claims.py --html")
    ap.add_argument("-o", "--output", required=True, help="output payload HTML (feed to build_review_doc.py)")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--voice", default="outlet", choices=["outlet", "user", "quote"],
                    help="system-prompt variant: outlet (default, E2/production), "
                         "user (ordinary-user posts, C2 CN corpus), or quote (a user's "
                         "quote post — quoted post shown as context)")
    ap.add_argument("--resume", default="",
                    help="checkpoint JSONL (default <output>.partial.jsonl). Pass 2 was "
                         "single-write-at-end: a kill at minute 44 of 45 lost everything "
                         "(Daniel 2026-08-04). Completed posts are now appended per post "
                         "and skipped on restart.")
    args = ap.parse_args()
    global PROMPT_HASH
    if args.voice != "outlet":
        global SYSTEM
        SYSTEM = {"user": SYSTEM_USER, "quote": SYSTEM_QUOTE}[args.voice]
    PROMPT_HASH = prompt_hash(SYSTEM)

    posts = load_payload(args.payload)
    # v4.4 (Daniel): pass 1's recall is trusted — the filter never adds claims, so
    # posts the extractor left empty skip it entirely
    todo = [p for p in posts if p.get("claims")]

    # ---- resume ----------------------------------------------------------
    ckpt = Path(args.resume or (args.output + ".partial.jsonl"))
    done = {}
    if ckpt.exists():
        for line in ckpt.open():
            try:
                d = json.loads(line)
                done[d["url"]] = d
            except Exception:
                pass                      # a torn final line is expected after a kill
    by_url = {p["url"]: p for p in posts}
    for u, d in done.items():
        if u in by_url:
            if d.get("claims") is not None:
                by_url[u]["claims"] = d["claims"]
            by_url[u]["nr"] = d.get("nr")
    n_before = len(todo)
    todo = [p for p in todo if p["url"] not in done]
    if done:
        print(f"resume: {len(done)} posts already done -> {len(todo)} of {n_before} remain",
              flush=True)
    _ck = ckpt.open("a")
    _cklock = threading.Lock()

    def _save(p, with_claims=True):
        with _cklock:
            _ck.write(json.dumps({"url": p["url"],
                                  "claims": p["claims"] if with_claims else None,
                                  "nr": p.get("nr")}, ensure_ascii=False) + "\n")
            _ck.flush()
    # ----------------------------------------------------------------------
    print(f"normalizing {len(todo)} posts-with-claims of {len(posts)} ({args.model})...", flush=True)
    t0 = time.time()

    stats = {"anchor_fixes": 0}

    def run(p):
        p1 = list(p["claims"])
        q = p.get("quoted") if args.voice == "quote" else None
        qt = (q or {}).get("text")
        o = normalize(args.model, p.get("text") or "", p1, q)
        corr = [c for c in (o.get("claims") or []) if isinstance(c, dict) and c.get("claim")]
        CATS = {"artifact", "promo", "duplicate", "merged", "trivial", "opinion",
                "unresolved", "checkable"}
        kept = []
        for c in corr:
            text = c.get("claim")
            if not anchored(text, p.get("text"), p1, qt):
                cand = max(p1, key=lambda x: len(_anchor_tokens(x.get("c")) & _anchor_tokens(text)),
                           default=None) if p1 else None
                if cand and anchored(cand.get("c"), p.get("text"), p1, qt):
                    text = cand.get("c")
                stats["anchor_fixes"] += 1
            cat = c.get("category") if c.get("category") in CATS else "artifact"
            kept.append({"c": text, "t": c.get("type"),
                         "g": int(c["group"]) if isinstance(c.get("group"), (int, float)) else None,
                         "cat": cat, "topic": c.get("topic") or "other"})
        # never-drop safety: any pass-1 claim the filter left unaccounted comes back,
        # category "unaccounted" (an audit failure signal, counted and visible)
        kept_tok = [_anchor_tokens(k["c"]) for k in kept]
        for c1 in p1:
            t1 = _anchor_tokens(c1.get("c"))
            if not t1:
                continue
            if not any(len(t1 & kt) / len(t1) >= 0.5 for kt in kept_tok):
                kept.append({"c": c1.get("c"), "t": c1.get("t"), "g": c1.get("g"),
                             "cat": "unaccounted", "topic": "other"})
        # code rule: unnamed-source attributions never stand as their own claim
        for k in kept:
            if k.get("t") == "attribution" and _UNNAMED_SRC.match(k["c"] or "") \
                    and k.get("cat") not in ("duplicate", "merged", "artifact"):
                has_content = any(x.get("g") == k.get("g") and x.get("t") == "assertion"
                                  for x in kept if k.get("g") is not None)
                if has_content:
                    k["cat"] = "merged"
                    k["_fold"] = True   # deliberate code fold — pair_guard must not restore
                    stats["unnamed_src_folds"] = stats.get("unnamed_src_folds", 0) + 1
                else:
                    k["t"] = "assertion"
                    k["g"] = None
                    stats["unnamed_src_folds"] = stats.get("unnamed_src_folds", 0) + 1
        p["claims"] = kept
        # post-level reason is CODE-computed: modal filtered category when nothing checkable
        if any(k["cat"] == "checkable" for k in kept):
            p["nr"] = None
        elif kept:
            from collections import Counter as _C
            p["nr"] = _C(k["cat"] for k in kept).most_common(1)[0][0]
        else:
            p["nr"] = "empty"
        return p

    # Periodic progress: this stage was silent from start to finish, so a wedged
    # job and a working one looked identical (Daniel 2026-08-04).
    _n = [0]
    def _run_p(p_):
        r = run(p_)
        _save(p_)
        _n[0] += 1
        if _n[0] % 100 == 0 or _n[0] == len(todo):
            el = time.time() - t0
            rate = _n[0] / max(el, 1e-9)
            print(f"  {_n[0]}/{len(todo)} | {rate*60:.0f} posts/min | "
                  f"${USAGE['cost']:.3f} | {el/60:.1f}m elapsed | "
                  f"ETA {(len(todo)-_n[0])/max(rate,1e-9)/60:.0f}m", flush=True)
        return r
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(_run_p, todo))
    # reason-only judgment for posts the extractor left EMPTY: the schema cannot contain
    # claims, so no manufacture risk; "missed-claims" doubles as a pass-1 recall alarm
    empties = [p for p in posts if (p.get("text") or "").strip() and not p.get("claims")
               and p["url"] not in done]
    def judge(p):
        p["nr"] = judge_empty(args.model, p.get("text") or "") or "empty"
        _save(p, with_claims=False)
        return p
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(judge, empties))
    if empties:
        from collections import Counter as _C
        print(f"  empty-post reasons ({len(empties)}): {dict(_C(p['nr'] for p in empties).most_common())}")
        alarms = [p["handle"] for p in empties if p["nr"] == "missed-claims"]
        if alarms:
            print(f"  RECALL ALARMS (missed-claims): {alarms}")
    _ck.close()
    dt = time.time() - t0

    payload = json.dumps({"posts": posts, "prompt_hash": PROMPT_HASH},
                         ensure_ascii=False).replace("</", "<\\/")
    Path(args.output).write_text(f'<!doctype html><meta charset="utf-8">'
                                 f'<script type="application/json" id="data">{payload}</script>')
    from collections import Counter
    cats = Counter(c["cat"] for p in posts for c in p["claims"])
    topics = Counter(c.get("topic") for p in posts for c in p["claims"] if c["cat"] == "checkable")
    n_claims = sum(len(p["claims"]) for p in posts)
    print(f"DONE in {dt:.0f}s ({dt/max(1,len(todo)):.1f}s/post). {n_claims} claims, "
          f"categories {dict(cats.most_common())}; anchor fixes {stats['anchor_fixes']} -> {args.output}")
    print(f"  checkable topics: {dict(topics.most_common())}")
    print(f"  tokens: {USAGE['in']:,} in / {USAGE['out']:,} out; cost ${USAGE['cost']:.3f} (DeepInfra)")
    if args.parquet:
        rows = [{"handle": p["handle"], "cell": p["cell"], "domain": p["domain"], "url": p["url"],
                 "post_text": p["text"], "claim": c["c"], "type": c["t"], "group": c.get("g"),
                 "category": c["cat"], "topic": c.get("topic"),
                 "checkworthy": c["cat"] == "checkable", "prompt_hash": PROMPT_HASH}
                for p in posts for c in p["claims"]]
        pd.DataFrame(rows).to_parquet(args.parquet, index=False)
        print(f"  wrote {args.parquet}")


if __name__ == "__main__":
    main()
