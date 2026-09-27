"""Extract fact-checkable claim(s) from harvested tweets (pass 1 of the extraction chain).

Production config: DeepSeek-V4-Flash, TEXT ONLY (--no-images is the default) — the config every
urn chain runs, invoked by cn_false_urn.chain and ingest_timeline_capture. Pass --images --model
Qwen/Qwen3-VL-30B-A3B-Instruct for the multimodal variant this script started as.

Reads the canonical posts parquet, sends each post's text (+ image(s) when multimodal) to the model
with the tweet extraction prompt, and writes one row per extracted claim (claim, type=assertion|
attribution, group) with full provenance back to the post. Topic-agnostic, multilingual, and
inference-permissive (a reasonable-reader inference from the framing may be part of the claim).

  cd src && uv run python eval/scripts/claim_sourcing/extract_tweet_claims.py \
      [-i eval/data/survey_claims/outlet_tweets.parquet] [-o eval/data/survey_claims/outlet_claims.parquet] \
      [--per-handle N] [--limit N] [--concurrency 6] [--print]
"""
import argparse, base64, json, re, sys, time, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.prompt_hash import prompt_hash  # noqa: E402

ENV = SRC / ".env"
BASE = "https://api.deepinfra.com/v1/openai"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
TEXT_ONLY = True   # --no-images (default); --images sends the post images to a multimodal model
VOICE = "outlet"   # --voice: outlet | user | quote (selects the system prompt + post framing)
USAGE = {"in": 0, "out": 0, "cost": 0.0}  # accumulated token usage + DeepInfra cost across calls

def _key():
    for line in ENV.read_text().splitlines():
        if line.startswith("DEEPINFRA_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("no DEEPINFRA_API_KEY in src/.env")
KEY = _key()

SYSTEM = """You extract claims from a news outlet's social-media post. A later pass filters them and a verifier checks them; your job is to find every claim and structure it. A claim states something about reality — an event, action, ruling, number, saying, or judgment. Extract every claim the post makes or reports, including hedged, charged, or opinion-shaded ones; when unsure whether something is a claim, extract it and let the filter decide. Promotional and broadcast-teaser posts still get their embedded claims extracted — an announcement can carry real reported events. Do not manufacture claims from promotion, captions, questions, or bare links that state nothing, and never emit the same proposition twice.

Two kinds of claim:
- assertion: something the post states in its own voice. Anonymous or unnamed sourcing ("sources say", "reportedly", "a new study found") counts as the outlet's own voice.
- attribution: the post reports that an identifiable person or organization said, wrote, or found something. Phrase it "X said Y", with the speaker named in the claim. One statement by one speaker is ONE attribution claim, however many sentences it runs.

Pairs. Reported speech from an identifiable person or organization raises two questions: did X say it, and is Y true. Always extract both: the attribution, and Y on its own as a bare assertion — same "group" number for both. If Y contains several distinct propositions — two figures, two effects, two events — each becomes its own assertion in the same group; never weld independently checkable propositions into one claim. Never write "Y, according to X" hybrids: the attribution keeps its "X said" form and the content claims stay bare. If the source is anonymous, unnamed, or an unnamed study or report, extract ONLY the content as an assertion. "group" links only an attribution to its content claims — never group plain assertions.

Wording. Resolve references using the post: name people and things (full name and role on first mention), keep the meaning exact, never invert who does what to whom, add nothing the post does not support. Claims are judged and verified together with the post, so they need not repeat every detail — but a claim should not mislead when read on its own.

Return only JSON:
{"claims": [{"claim": "...", "type": "assertion" or "attribution", "group": integer or null}]}
An empty list is a valid answer for a post that states nothing."""

# --voice user: ordinary-user posts (C2 CN corpus). Same contract as SYSTEM; differences:
# the author is any account, the post may be a reply/rant/anecdote fragment, and claims
# arrive wrapped in tone. Each added sentence answers a measured C2-audit miss class.
SYSTEM_USER = """You extract claims from a social-media post. A later pass filters them and a verifier checks them; your job is to find every claim and structure it. A claim states something about reality — an event, action, ruling, number, saying, or judgment. The post can be anything a person posts: an original post, a reply, a quote, a rant, a joke, or a personal story — often a fragment reacting to something not shown. Judge what this post itself asserts. Extract every claim the post makes or reports, including hedged, charged, or opinion-shaded ones; when unsure whether something is a claim, extract it and let the filter decide. Claims are often wrapped in sarcasm, insults, or outrage — extract the factual assertion beneath the tone, never the sentiment itself. A first-person story can carry checkable public claims — what a company, product, or institution did or charged, what happened at a public event — extract those. A post narrating an event as fact is asserting it, however partisan its framing. Do not manufacture claims from captions, questions, greetings, or bare links that state nothing, and never emit the same proposition twice.

Two kinds of claim:
- assertion: something the post states in its own voice. Anonymous or unnamed sourcing ("sources say", "reportedly", "a new study found") counts as the author's own voice.
- attribution: the post reports that an identifiable person or organization said, wrote, or found something. Phrase it "X said Y", with the speaker named in the claim. One statement by one speaker is ONE attribution claim, however many sentences it runs.

Pairs. Reported speech from an identifiable person or organization raises two questions: did X say it, and is Y true. Always extract both: the attribution, and Y on its own as a bare assertion — same "group" number for both. If Y contains several distinct propositions — two figures, two effects, two events — each becomes its own assertion in the same group; never weld independently checkable propositions into one claim. Never write "Y, according to X" hybrids: the attribution keeps its "X said" form and the content claims stay bare. If the source is anonymous, unnamed, or an unnamed study or report, extract ONLY the content as an assertion. "group" links only an attribution to its content claims — never group plain assertions.

Wording. Resolve references using the post: name people and things (full name and role on first mention), keep the meaning exact, never invert who does what to whom, add nothing the post does not support. A reply or quote fragment may leave a referent unnamed — extract the claim with the best identification the post supports rather than dropping it. Claims are judged and verified together with the post, so they need not repeat every detail — but a claim should not mislead when read on its own.

Return only JSON:
{"claims": [{"claim": "...", "type": "assertion" or "attribution", "group": integer or null}]}
An empty list is a valid answer for a post that states nothing."""


# --voice quote: an ordinary user's QUOTE post (Daniel 2026-08-25). Built on SYSTEM_USER —
# same contract, same "two kinds" / pairs / wording blocks verbatim — with the framing
# changed to a quote post and ONE added paragraph: the author's text is the subject, the
# quoted post is context, and the claims to find are the author's own assertions
# including their assessment of the quoted post (endorse / dispute / correct / add).
# The quoted post's own claims are never extracted here: it is captured separately.
SYSTEM_QUOTE = SYSTEM_USER.replace(
    "The post can be anything a person posts: an original post, a reply, a quote, a rant, "
    "a joke, or a personal story — often a fragment reacting to something not shown. "
    "Judge what this post itself asserts.",
    "This is a QUOTE post: the author reposted another account's post (the QUOTED post, "
    "shown after the author's text) and wrote their own text on top of it — often a "
    "fragment reacting to it. Judge what the author's own text asserts."
).replace(
    "Two kinds of claim:",
    "The quoted post is context, not a source of claims. Use it to resolve what \"this\", "
    "\"he\", \"that number\" refer to and to spell out the proposition the author is "
    "reacting to — but never extract a claim the quoted post makes on its own that the "
    "author takes no position on; the quoted post is handled separately under its own "
    "author. What matters is how the author assesses the quoted post. An author who "
    "endorses it (\"this\", \"exactly\", \"true\", restating it approvingly) is asserting its "
    "content: extract those propositions as the author's assertions, spelled out from the "
    "quoted text. A reaction that presupposes the quoted post is true — outrage, alarm, "
    "\"look at this\", \"unbelievable\", \"do not ignore this\" — counts as endorsement: extract "
    "the quoted propositions as the author's assertions. An author who disputes or corrects it is asserting the correction, or "
    "that the quoted statement is false: extract what the author claims instead, and the "
    "author's claim that the quoted statement is false, each spelled out so it stands on "
    "its own. An author who adds to it — a further fact, a consequence, a cause — asserts "
    "that addition. Mockery, insult, or a bare reaction with no factual position yields "
    "nothing. The quoted account is an identifiable speaker, so the author's text may "
    "attribute a saying to them (\"X said Y\") — extract that as an attribution only when "
    "the author's own text turns on who said it.\n\n"
    "Two kinds of claim:"
)
assert SYSTEM_QUOTE != SYSTEM_USER and SYSTEM_QUOTE.count("QUOTE post") == 1

PROMPT_HASH = prompt_hash(SYSTEM)   # reset in main() once --voice picks the prompt


def _obj(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _fetch_b64(url, timeout=20):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            ct = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
        return f"data:{ct};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return None


def _chat(content, max_tokens=0, retries=4):
    thinking = "Thinking" in MODEL
    mt = max_tokens or (8000 if thinking else 1000)  # thinking needs room for reasoning + JSON
    payload = {"model": MODEL, "temperature": 0, "max_tokens": mt,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": content}]}
    if not thinking:  # forced-JSON mode suppresses thinking models' output; they emit clean JSON in content
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read().decode())
                u = d.get("usage") or {}
                USAGE["in"] += u.get("prompt_tokens", 0); USAGE["out"] += u.get("completion_tokens", 0)
                USAGE["cost"] += u.get("estimated_cost") or 0
                return _obj(d["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(3 * 2 ** a); continue
            print(f"  HTTP {e.code}: {e.read().decode()[:150]}"); return {}
        except Exception:
            time.sleep(3 * 2 ** a)
    return {}


def extract_post(post):
    """One post -> list of claim dicts with provenance. Never raises (a bad post yields [])."""
    try:
        text = post.get("text") or ""
        n_img = int(post.get("n_images") or 0)
        body = f"Post by @{post['handle']}:\n{text}"
        if VOICE == "quote" and post.get("quoted_text"):
            body += (f"\n\nQuoted post by @{post.get('quoted_handle') or 'unknown'} "
                     f"(context only — the author is reacting to this):\n{post['quoted_text']}")
        content = [{"type": "text", "text": body}]
        if not TEXT_ONLY:
            imgs = post.get("image_urls")
            imgs = list(imgs) if imgs is not None else []   # parquet list-cols come back as numpy arrays
            for u in imgs[:4]:
                b = _fetch_b64(u)
                if b:
                    content.append({"type": "image_url", "image_url": {"url": b}})
        obj = _chat(content)
        claims = obj.get("claims") or []
        post["no_claim_reason"] = None   # v4: pass 1 never filters; pass 2 is the judge
        rows = []
        for c in claims:
            if not isinstance(c, dict) or not c.get("claim"):
                continue
            rows.append({"post_id": post["post_id"], "cell": post["cell"], "domain": post["domain"],
                         "handle": post["handle"], "url": post["url"], "created_at": post["created_at"],
                         "lang": post.get("lang"), "post_text": text, "n_images": n_img,
                         "claim": c.get("claim"), "type": c.get("type"),
                         "prompt_hash": PROMPT_HASH,
                         "group": int(c["group"]) if isinstance(c.get("group"), (int, float)) else None})
        return rows
    except Exception as e:
        print(f"  extract error on post {post.get('post_id')}: {type(e).__name__}: {e}")
        return []


def balanced_sample(df, per_handle):
    """Per handle: half image-bearing, half text-only, to exercise the from_image path."""
    out = []
    for _, g in df.groupby("handle"):
        wi, ni = g[g.n_images > 0], g[g.n_images == 0]
        take_i = wi.head(per_handle // 2)
        take_n = ni.head(per_handle - len(take_i))
        out.append(pd.concat([take_i, take_n]))
    return pd.concat(out).reset_index(drop=True)


_CLAIMS_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Tweet claim extraction review</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a19;--muted:#6b6b68;--card:#fff;--line:#e5e4e1;--accent:#3b5bdb;
 --L:#2f6feb;--R:#c0392b;--assert:#2f7d4f;--attrib:#7048b6;}
@media(prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e8e6;--muted:#9a9a97;--card:#202225;--line:#33353a;
 --L:#5a8def;--R:#e05a4d;--assert:#5aa574;--attrib:#9a78d0;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:22px 18px 80px}
h1{font-size:21px;margin:0 0 4px}.sub{color:var(--muted);font-size:14px;margin:0 0 16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 15px;margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:4px 12px 4px 0}th{color:var(--muted)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:5;border-bottom:1px solid var(--line)}
.controls input,.controls select{font:inherit;padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--fg)}
.controls input[type=search]{flex:1;min-width:160px}
label.chk{color:var(--muted);font-size:13px;display:flex;align-items:center;gap:5px}
.count{color:var(--muted);font-size:13px;margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px;font-size:12px}
.src{padding:2px 8px;border-radius:20px;font-weight:600;color:#fff}.src.L{background:var(--L)}.src.R{background:var(--R)}
.date{color:var(--muted)}.meta a{color:var(--accent);text-decoration:none}
.post{white-space:pre-wrap;font-size:14px;margin:2px 0 8px}
.imgs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}.imgs img{max-height:130px;max-width:100%;border-radius:6px;border:1px solid var(--line)}
.claim{border-left:3px solid var(--line);padding:3px 0 3px 10px;margin:6px 0}
.claim.assertion{border-color:var(--assert)}.claim.attribution{border-color:var(--attrib)}
.chip{font-size:11px;font-weight:600;padding:1px 6px;border-radius:5px;margin-right:6px}
.chip.assertion{color:var(--assert);border:1px solid var(--assert)}.chip.attribution{color:var(--attrib);border:1px solid var(--attrib)}
.chip.img{color:var(--muted);border:1px solid var(--line)}
.chip.flag{color:var(--R);border:1px solid var(--R)}
.chip.reason{color:var(--accent);border:1px solid var(--accent);font-size:11px;font-weight:600;padding:1px 6px;border-radius:5px;margin-left:4px}
.filters{position:sticky;top:0;background:var(--bg);padding:9px 0;z-index:5;border-bottom:1px solid var(--line);margin-bottom:12px}
.filters input[type=search]{font:inherit;padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--fg);width:100%;margin-bottom:8px}
.frow{display:flex;flex-wrap:wrap;gap:5px;align-items:center;margin:5px 0}
.flabel{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin-right:2px;min-width:56px}
.fgroup{display:flex;flex-wrap:wrap;gap:4px}
.tog{font:inherit;font-size:12px;padding:3px 9px;border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--fg);cursor:pointer}
.tog.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tog.src.L.on{background:var(--L);border-color:var(--L)}.tog.src.R.on{background:var(--R);border-color:var(--R)}
.ctext{font-weight:600}.basis{color:var(--muted);font-size:12.5px;margin-top:2px}
.nodrop{color:var(--muted);font-style:italic;font-size:13px}
</style></head><body><div class="wrap">
<h1 id="title"></h1><p class="sub" id="sub"></p>
<div class="panel" id="stats"></div>
<div class="filters">
 <input type="search" id="q" placeholder="Search claim / post…">
 <div class="frow"><span class="flabel">sources</span><div class="fgroup" data-group="src" id="fsrc"></div></div>
 <div class="frow">
  <span class="flabel">type</span><div class="fgroup" data-group="type"><button class="tog" data-v="assertion">assertion</button><button class="tog" data-v="attribution">attribution</button></div>
  <span class="flabel">checkworthy</span><div class="fgroup" data-group="cw"><button class="tog" data-v="1">yes</button><button class="tog" data-v="0">no</button></div>
  <span class="flabel">no-claim</span><div class="fgroup" data-group="reason"><button class="tog" data-v="promo">promo</button><button class="tog" data-v="opinion">opinion</button><button class="tog" data-v="none">none</button></div>
 </div>
</div>
<p class="count" id="count"></p><div id="list"></div>
</div>
<script type="application/json" id="data">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const esc=s=>(s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('title').textContent=D.title;
document.getElementById('sub').textContent=`${D.n_posts} posts sampled · ${D.n_with} with a claim (${Math.round(100*D.n_with/D.n_posts)}%) · ${D.n_claims} claims`;
const bySrc={};D.posts.forEach(p=>{const s=bySrc[p.handle]||(bySrc[p.handle]={h:p.handle,cell:p.cell,posts:0,claims:0});s.posts++;s.claims+=p.claims.length;});
let sh='<table><tr><th>source</th><th>lean</th><th>posts</th><th>claims</th><th>yield</th></tr>';
Object.values(bySrc).forEach(s=>{sh+=`<tr><td>@${esc(s.h)}</td><td>${esc(s.cell)}</td><td>${s.posts}</td><td>${s.claims}</td><td>${(s.claims/s.posts).toFixed(1)}/post</td></tr>`;});
document.getElementById('stats').innerHTML=sh+'</table>';
const cls=cell=>cell&&cell.toLowerCase().startsWith('l')?'L':'R';
const srcWrap=document.getElementById('fsrc');
Object.values(bySrc).sort((a,b)=>a.h.localeCompare(b.h)).forEach(s=>{const b=document.createElement('button');b.className='tog src '+cls(s.cell);b.dataset.v=s.h;b.textContent='@'+s.h;srcWrap.appendChild(b);});
const groups={src:new Set(),type:new Set(),cw:new Set(),reason:new Set()};
document.querySelectorAll('.tog').forEach(b=>b.addEventListener('click',()=>{const g=b.parentElement.dataset.group,v=b.dataset.v;if(groups[g].has(v)){groups[g].delete(v);b.classList.remove('on');}else{groups[g].add(v);b.classList.add('on');}render();}));
const q=document.getElementById('q'),list=document.getElementById('list'),count=document.getElementById('count');
q.addEventListener('input',render);
const unit=c=>(!groups.type.size||groups.type.has(c.t))&&(!groups.cw.size||groups.cw.has(c.cw?'1':'0'));
const claimFilt=()=>groups.type.size||groups.cw.size;
function render(){
 const t=q.value.toLowerCase();let shown=0;
 list.innerHTML=D.posts.filter(p=>{
   if(groups.src.size&&!groups.src.has(p.handle))return false;
   const u=p.claims.filter(unit);
   const claimPass=!groups.reason.size&&u.length>0;
   const emptyPass=p.claims.length===0&&!claimFilt()&&(!groups.reason.size||groups.reason.has(p.nr||'none'));
   if(!(claimPass||emptyPass))return false;
   if(t&&!((p.text+' '+p.claims.map(c=>c.c).join(' ')).toLowerCase().includes(t)))return false;
   p._u=u;return true;
 }).map(p=>{shown++;
  const imgs=p.images.map(u=>`<img src="${esc(u)}" loading="lazy">`).join('');
  const body=p.claims.length?p._u.map(c=>`<div class="claim ${c.t}"><span class="chip ${c.t}">${esc(c.t)}</span>${c.cw?'':'<span class="chip flag">not checkworthy</span>'}<span class="ctext">${esc(c.c)}</span></div>`).join('')
    :`<div class="nodrop">no claim <span class="chip reason">${esc(p.nr||'—')}</span></div>`;
  return `<div class="card"><div class="meta"><span class="src ${cls(p.cell)}">@${esc(p.handle)}</span><span class="date">${esc(p.date)}</span><span>${esc(p.domain)}</span><a href="${esc(p.url)}" target="_blank" rel="noopener">tweet ↗</a></div>${imgs?`<div class="imgs">${imgs}</div>`:''}<div class="post">${esc(p.text)}</div>${body}</div>`;
 }).join('');
 count.textContent=`${shown} of ${D.n_posts} posts`;}
render();
</script></body></html>"""


def write_claims_html(results, path, title="Tweet claim extraction review"):
    import json as _json
    posts = []
    for post, rows in results:
        imgs = post.get("image_urls")
        imgs = list(imgs) if imgs is not None else []
        posts.append({"handle": post.get("handle"), "cell": post.get("cell"), "domain": post.get("domain"),
                      "date": str(post.get("created_at") or "")[:10], "url": post.get("url"),
                      "text": post.get("text") or "", "images": [str(u) for u in imgs][:4],
                      "quoted": ({"handle": post.get("quoted_handle"), "text": post["quoted_text"]}
                                 if post.get("quoted_text") else None),
                      "nr": post.get("no_claim_reason"),
                      "claims": [{"c": r.get("claim"), "t": r.get("type"),
                                  "g": r.get("group")}
                                 for r in rows]})
    data = {"title": title, "posts": posts, "n_posts": len(posts),
            "n_claims": sum(len(p["claims"]) for p in posts),
            "n_with": sum(1 for p in posts if p["claims"])}
    payload = _json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    open(path, "w").write(_CLAIMS_HTML.replace("__PAYLOAD__", payload))


def main():
    global TEXT_ONLY, MODEL, VOICE, PROMPT_HASH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=str(SRC / "eval/data/survey_claims/outlet_tweets.parquet"))
    ap.add_argument("-o", "--output", default=str(SRC / "eval/data/survey_claims/outlet_claims.parquet"))
    ap.add_argument("--per-handle", type=int, default=0, help="balanced sample size per handle (0 = all posts)")
    ap.add_argument("--limit", type=int, default=0, help="hard cap on posts processed")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--print", action="store_true", help="print each post + its extracted claims")
    ap.add_argument("--html", default="", help="also write a self-contained claims-review HTML (post -> claims, incl. drops + images)")
    ap.add_argument("--no-images", action="store_true", default=True,
                    help="text-only: do not send images to the model (the default)")
    ap.add_argument("--images", dest="no_images", action="store_false",
                    help="send the post images too (needs a multimodal --model)")
    ap.add_argument("--model", default=MODEL, help="extraction model id (DeepInfra)")
    ap.add_argument("--voice", default="outlet", choices=["outlet", "user", "quote"],
                    help="system-prompt variant: outlet (default, E2/production), "
                         "user (ordinary-user posts, C2 CN corpus), or quote (a user's "
                         "quote post: claims from the author's text, quoted post as context)")
    ap.add_argument("--resume", default="",
                    help="checkpoint JSONL (default <output>.partial.jsonl). This stage was "
                         "single-write-at-end: a kill before the last post lost every result "
                         "(Daniel 2026-08-04). Posts are now appended as they complete and "
                         "skipped on restart.")
    args = ap.parse_args()
    TEXT_ONLY = args.no_images
    MODEL = args.model
    VOICE = args.voice
    if args.voice != "outlet":
        global SYSTEM
        SYSTEM = {"user": SYSTEM_USER, "quote": SYSTEM_QUOTE}[args.voice]
    PROMPT_HASH = prompt_hash(SYSTEM)

    df = pd.read_parquet(args.input)
    posts = balanced_sample(df, args.per_handle) if args.per_handle else df
    if args.limit:
        posts = posts.head(args.limit)
    recs = posts.to_dict("records")
    print(f"extracting from {len(recs)} posts ({MODEL})...", flush=True)

    # ---- resume ----------------------------------------------------------
    ckpt = Path(args.resume or (args.output + ".partial.jsonl"))
    prev = {}
    if ckpt.exists():
        for line in ckpt.open():
            try:
                d = json.loads(line)
                prev[str(d["post_id"])] = d
            except Exception:
                pass                      # torn final line after a kill
    if prev:
        print(f"resume: {len(prev)} posts already extracted", flush=True)
    all_rows, results, done = [], [], 0
    for r_ in recs:                       # replay checkpointed posts, in order
        d = prev.get(str(r_.get("post_id")))
        if d is not None:
            r_["no_claim_reason"] = d.get("no_claim_reason")
            all_rows.extend(d["rows"])
            results.append((r_, d["rows"]))
    todo = [r_ for r_ in recs if str(r_.get("post_id")) not in prev]
    ck = ckpt.open("a")
    # ----------------------------------------------------------------------
    _t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for post, rows in zip(todo, ex.map(extract_post, todo)):
            all_rows.extend(rows)
            results.append((post, rows))
            ck.write(json.dumps({"post_id": post["post_id"], "rows": rows,
                                 "no_claim_reason": post.get("no_claim_reason")},
                                default=str) + "\n")
            ck.flush()
            done += 1
            # Periodic progress: a multi-thousand-post run was otherwise SILENT from
            # start to finish, so a wedged job and a working one looked identical
            # (Daniel 2026-08-04, caught 15 min into a 4,372-post run).
            if done % 100 == 0 or done == len(todo):
                el = time.time() - _t0
                rate = done / max(el, 1e-9)
                print(f"  {done}/{len(todo)} | {len(all_rows)} claims | "
                      f"{rate*60:.0f} posts/min | ${USAGE['cost']:.3f} | "
                      f"{el/60:.1f}m elapsed | ETA {(len(todo)-done)/max(rate,1e-9)/60:.0f}m",
                      flush=True)
            if args.print:
                print(f"\n[{done}/{len(recs)}] @{post['handle']} ({post['cell']}, imgs={post.get('n_images')}) {post['url']}")
                print(f"    TEXT: {(post.get('text') or '')[:200]!r}")
                if not rows:
                    print(f"    -> (no claim: {post.get('no_claim_reason')})")
                for r in rows:
                    cw = "" if r["checkworthy"] else " /not-checkworthy"
                    print(f"    -> [{r['type']}{cw}] {r['claim']}")

    ck.close()
    out = pd.DataFrame(all_rows)
    out.to_parquet(args.output, index=False)
    n_posts = len(recs)
    print(f"\nDONE. {len(out)} claims from {n_posts} posts ({len(out)/max(1,n_posts):.2f} claims/post) -> {args.output}")
    ci, co = USAGE["in"], USAGE["out"]
    x = 20800 / max(1, n_posts)
    print(f"  tokens: {ci:,} in / {co:,} out ({co/max(1,n_posts):.0f} out/post); "
          f"cost ${USAGE['cost']:.3f} (DeepInfra) -> ~${USAGE['cost']*x:.2f} for full {int(x)}x corpus")
    if len(out):
        print(f"  types: {out['type'].value_counts().to_dict()}")

        print(f"  posts with >=1 claim: {out['post_id'].nunique()}/{n_posts} ({100*out['post_id'].nunique()/n_posts:.0f}%)")
        print(f"  per cell:\n{out.groupby('cell').size().to_string()}")
        reasons = pd.Series([p.get("no_claim_reason") for p in recs if p.get("no_claim_reason")]).value_counts()
        if len(reasons):
            print(f"  no_claim_reason:\n{reasons.to_string()}")
    if args.html:
        write_claims_html(results, args.html, title=f"Tweet claim extraction — {len(recs)} posts, {len(out)} claims")
        print(f"  wrote HTML review -> {args.html}")


if __name__ == "__main__":
    main()
