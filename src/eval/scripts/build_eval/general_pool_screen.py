"""On-topic screen smoke over the zeerover general-tweet captures (trivially-true pool scout).

Estimates what share of Daniel's captured general tweets qualify for claim
extraction — i.e. are about public, checkable subject matter (politics, current
events, public figures, economy, science/health, notable sports/culture facts)
rather than personal chatter, jokes, promo, or bare media links. Sample-based
(default 200, stratified by language), DeepSeek-V4-Flash, one short call per
tweet. Output: eval/data/urn_runs/general_pool/screen_smoke.jsonl + summary.
"""
import json, glob, random, re, sys, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.prompt_hash import prompt_hash  # noqa: E402

POOL = SRC / "eval/data/tweet_corpus/general_pool"
OUT = SRC / "eval/data/urn_runs/general_pool"
BASE = "https://api.deepinfra.com/v1/openai"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
_KEY = None


def _key() -> str:
    """Read the API key on first use, never at import time."""
    global _KEY
    if _KEY is None:
        for line in (SRC / ".env").read_text().splitlines():
            if line.startswith("DEEPINFRA_API_KEY="):
                _KEY = line.split("=", 1)[1].strip()
        if not _KEY:
            sys.exit("no DEEPINFRA_API_KEY in src/.env")
    return _KEY


SYS = """You judge whether a social-media post qualifies for factual claim extraction.

A post QUALIFIES when its text asserts or reports something about public, checkable
subject matter: politics, government, current events, public figures or organizations,
the economy, science, health, or notable public facts (including major sports or
cultural outcomes). The assertion must be about the world, not only about the author.

A post does NOT qualify when it is personal chatter or opinion with no public factual
assertion, a joke or meme reaction, self-promotion or advertising, a bare link or
media share with no assertion in the text, or a question that asserts nothing.

Reply with JSON only: {"qualifies": true/false, "category": "<one of: politics,
current_events, public_figure, economy, science_health, sports_culture, personal,
joke_meme, promo_ad, bare_link, question, other>", "reason": "<one short sentence>"}"""


# Strict screen (2026-08-26, Daniel: "a more strict gate; I don't mind missing
# stuff"): adds a stakes test and anecdote / mockery / rhetoric exclusions, and
# treats the author's own promotion as promo. On the first For You capture it
# passed 61/194 vs 104/194 for SYS; the 43 drops were hand-read — 2 wanted posts
# lost (a viral claim about Altman read as watch promo; a 2015 Ellison clip read
# as anecdote), the rest noise. The production ingest uses this; the eval-stratum
# build keeps SYS so its pool stays reproducible.
SYS_STRICT = """You are the intake gate of a fact-checking tool that flags misleading posts in a user's social-media feed. Decide whether a post is worth passing to claim extraction. Be strict: the cost of passing noise is high, the cost of missing a borderline post is low.

A post QUALIFIES only when BOTH hold:
1. Its own text asserts something about public affairs — politics, government, current events, public figures or institutions acting in their public role, the economy, science, health, or notable public facts (including major sports or cultural outcomes). For a quote post (author's text followed by the quoted post), what the author endorses or adds counts as asserted; what they merely react to does not.
2. Stakes: a reader who believed it would be misled about a matter of public concern if it were false.

A post does NOT qualify when it is any of:
- personal: the author's life, feelings, or opinions with no public factual assertion.
- joke_meme: humor, memes, reaction content.
- promo_ad: the author promoting their OWN product, company, project, event, or content, or a company announcing its own results. A claim made by an outsider ABOUT a company or public figure is not promo — it qualifies if it has stakes.
- bare_link: a link or media share whose text asserts nothing.
- question: asks and asserts nothing.
- anecdote: storytelling — personal or biographical narrative, human-interest stories, celebrity anecdotes, history lessons, "here is what happened to me/them" accounts — even when the details are checkable, because nothing of public concern turns on them. NOT anecdote: a specific act by a public figure or institution in their public role (a politician's, official's, or executive's decision, statement, deal, or intervention), however old.
- mockery: the author quotes, transcribes, or screenshots what someone else said (an interviewee, a random account, a public figure) in order to mock, marvel at, or dunk on it, without asserting the content themselves.
- rhetoric: commentary, evaluation, prediction, or exhortation whose only factual content is common knowledge.

Reply with JSON only: {"qualifies": true/false, "category": "<one of: politics, current_events, public_figure, economy, science_health, sports_culture, personal, joke_meme, promo_ad, bare_link, question, anecdote, mockery, rhetoric, other>", "reason": "<one short sentence>"}"""


PROMPT_HASH = {False: prompt_hash(SYS), True: prompt_hash(SYS_STRICT)}


def call(text: str, strict: bool = False) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS_STRICT if strict else SYS},
                     {"role": "user", "content": text[:1600 if strict else 1200]}],
        "temperature": 0.0, "max_tokens": 150,
    }).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {_key()}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        out = json.load(r)
    txt = out["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    d = json.loads(m.group(0)) if m else {"qualifies": None, "category": "parse_error",
                                          "reason": txt[:120]}
    d["usage"] = out.get("usage", {})
    d["prompt_name"] = "screen-strict" if strict else "screen"
    d["prompt_hash"] = PROMPT_HASH[strict]
    return d


def main(n=200, seed=20260824):
    rows, seen = [], set()
    for f in sorted(glob.glob(str(POOL / "x_capture_*.ndjson"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            rows.append(d)
    keep = [r for r in rows if r.get("lang") in ("en", "fr") and len(r.get("full_text") or "") > 15]
    random.Random(seed).shuffle(keep)
    # stratify: half en, half fr (pool is fr-heavy; report per-language anyway)
    en = [r for r in keep if r["lang"] == "en"][: n // 2]
    fr = [r for r in keep if r["lang"] == "fr"][: n // 2]
    sample = en + fr
    print(f"pool {len(rows)} unique, screened sample {len(sample)} (en {len(en)} fr {len(fr)})",
          flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(call, r["full_text"]): r for r in sample}
        for i, fut in enumerate(futs):
            pass
        done = 0
        for fut, r in futs.items():
            try:
                j = fut.result()
            except Exception as e:  # noqa: BLE001
                j = {"qualifies": None, "category": "error", "reason": str(e)[:120]}
            results.append({"id": r["id"], "lang": r["lang"],
                            "screen_name": r.get("screen_name"),
                            "operation": r.get("operation"),
                            "text": r["full_text"], **j})
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(sample)}", flush=True)

    with open(OUT / "screen_smoke.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if r["qualifies"] is not None]
    q = [r for r in ok if r["qualifies"]]
    print(f"qualify: {len(q)}/{len(ok)} = {len(q)/max(1,len(ok)):.0%}")
    for lang in ("en", "fr"):
        o = [r for r in ok if r["lang"] == lang]
        ql = [r for r in o if r["qualifies"]]
        print(f"  {lang}: {len(ql)}/{len(o)} = {len(ql)/max(1,len(o)):.0%}")
    from collections import Counter
    print("categories:", Counter(r["category"] for r in ok).most_common(12))
    toks = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)
    print("total tokens:", toks)


if __name__ == "__main__":
    main()
