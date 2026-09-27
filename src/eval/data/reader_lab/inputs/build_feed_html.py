import json, html, datetime, random, unicodedata, base64, re, urllib.request
from pathlib import Path
import sys
JSON, TAGS, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]   # MODE timeline | outlet
d = json.load(open(JSON))
tags = json.load(open(TAGS)) if Path(TAGS).exists() else {}
rng = random.Random(20260909)
PROJECT = d.get("numbering") == "project"
def R(r): return int(r) if PROJECT else 6 - int(r)
SHOW = (1, 2, 5) if MODE == "outlet" else (1, 2)
posts = []
for rt in SHOW:
    pool = [p for p in d["posts"] if R(p["post_rating"]) == rt]
    posts += rng.sample(pool, min(25, len(pool)))
def _ts(p):
    try:
        try: return datetime.datetime.strptime(p["created_at"], "%a %b %d %H:%M:%S %z %Y")
        except ValueError: return datetime.datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
    except Exception: return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
posts.sort(key=_ts, reverse=True)

IMG = Path("img"); IMG.mkdir(exist_ok=True)
def fetch(u):
    fn = IMG / (re.sub(r"[^A-Za-z0-9]", "_", u.split("/")[-1].split("?")[0]) + ".jpg")
    if not fn.exists():
        try:
            req = urllib.request.Request(u.split("?")[0] + "?name=small", headers={"User-Agent": "Mozilla/5.0"})
            fn.write_bytes(urllib.request.urlopen(req, timeout=20).read())
        except Exception as e:
            print("image failed", u, e); return None
    b = fn.read_bytes()
    mime = "image/png" if b[:4] == b"\x89PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(b).decode()

def clean_text(t):
    t = unicodedata.normalize("NFKC", t or "")
    return t.replace("\xa0", " ").replace("Â ", " ")
def esc(s): return html.escape(clean_text(s), quote=True)
def kfmt(v):
    if v is None: return ""
    v = float(v)
    if v >= 1e6: return f"{v/1e6:.1f}M".replace(".0M","M")
    if v >= 1e3: return f"{v/1e3:.1f}K".replace(".0K","K")
    return f"{int(v)}"
def dfmt(s):
    try:
        try: t = datetime.datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        except ValueError: t = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return t.strftime("%b %-d") if t.year == 2026 else t.strftime("%b %-d, %Y")
    except Exception: return s or ""
FLAGNAME = {"5":"states it","4":"points toward","X":"on topic, no direction","I":"irrelevant or silent","3":"contested","2":"points against","1":"contradicts it"}
def flagcells(fl):
    if not fl: return ""
    return "".join(f'<i class="f f{c}" title="{FLAGNAME.get(c,c)}"></i>' for c in fl.split(","))
TAGNAME = {"off_claim_evidence":"evidence is off the claim","time_sensitive":"time sensitive","predictive_or_opinion":"prediction or opinion","plausibly_false":"plausibly false","confirming_evidence":"documents confirm the claim","attribution_pair":"reported speech, bare content claim"}
def rlabel(r): return {1:"certainly false",2:"likely false",3:"no verdict",4:"leans true",5:"true"}.get(r,"")
ICON = {
 "reply":'<svg viewBox="0 0 24 24"><path d="M1.75 12.5a10.25 10.25 0 1 1 5.1 8.87L2 22.5l1.3-4.4A10.2 10.2 0 0 1 1.75 12.5z" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
 "rt":'<svg viewBox="0 0 24 24"><path d="M4.5 8.5l3-3 3 3M7.5 5.5v10a2 2 0 0 0 2 2h4M19.5 15.5l-3 3-3-3M16.5 18.5v-10a2 2 0 0 0-2-2h-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
 "like":'<svg viewBox="0 0 24 24"><path d="M12 20.5s-8-5.2-8-11a4.3 4.3 0 0 1 8-2.2 4.3 4.3 0 0 1 8 2.2c0 5.8-8 11-8 11z" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
 "views":'<svg viewBox="0 0 24 24"><path d="M5 20V11M10 20V4M15 20v-8M20 20v-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
}

def card(p):
    h = p["handle"] or "unknown"
    init = h[0].upper()
    hue = sum(ord(c) for c in h) % 360
    quote = ""
    if p.get("is_quote") and (p.get("quoted_text") or p.get("quoted_handle")):
        quote = f'<div class="quote"><div class="qh"><b>{esc(p.get("quoted_handle") or "")}</b> <span>@{esc(p.get("quoted_handle") or "")}</span></div><div class="qt">{esc(p.get("quoted_text") or "")}</div></div>'
    media = ""
    n = int(p.get("n_images") or 0)
    if n:
        urls = p.get("image_urls") or []
        imgs = [(u, fetch(u)) for u in urls]
        got = [f'<a href="{esc(u)}" target="_blank" rel="noopener"><img src="{du}" alt="post image" loading="lazy"></a>' for u, du in imgs if du]
        miss = [f'<a href="{esc(u)}" target="_blank" rel="noopener">image {i+1}</a>' for i, (u, du) in enumerate(imgs) if not du]
        if got: media = f'<div class="pics n{min(len(got),4)}">{"".join(got)}</div>'
        if miss: media += f'<div class="media"><span>not loaded</span>{" ".join(miss)}</div>'
    if int(p.get("n_videos") or 0):
        media += '<div class="media"><span>video attached</span></div>'
    r = R(p["post_rating"])
    alert = f'<div class="alert" data-r="{r}"><svg viewBox="0 0 24 24"><path d="M12 2.5l8 3.5v6c0 5-3.4 8.6-8 9.5-4.6-.9-8-4.5-8-9.5V6l8-3.5z" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M12 8v5M12 15.5v.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg><span>This content was reviewed by an AI tool. The tool rated this content as <b>{"TRUE" if r == 5 else "FALSE"}</b>.</span></div>'
    return f'''<article class="tw">
<div class="av" style="--h:{hue}">{init}</div>
<div class="body">
<div class="hd"><b>{esc(h)}</b><span>@{esc(h)} · {dfmt(p.get("created_at"))}</span>{(f'<span class="ng">{esc(p.get("lean"))} · NewsGuard {p.get("ng_score"):g}</span>' if p.get("ng_score") is not None else "")}</div>
<div class="tx">{esc(p.get("post_text"))}</div>
{quote}{media}
<div class="mx"><span>{ICON["reply"]}{kfmt(p.get("reply_count"))}</span><span>{ICON["rt"]}{kfmt(p.get("retweet_count"))}</span><span>{ICON["like"]}{kfmt(p.get("like_count"))}</span><span>{ICON["views"]}{kfmt(p.get("view_count"))}</span></div>
{alert}
</div></article>'''

def inspector(p):
    r = R(p["post_rating"])
    scored = [c for c in p["claims"] if c.get("score") is not None]
    low = min(scored, key=lambda c: c["score"])
    others = [c for c in scored if c is not low]
    unscored = len(p["claims"]) - len(scored)
    ev = ""
    if low.get("evidence"):
        items = "".join(f'<li><a href="{esc(e.get("url"))}" target="_blank" rel="noopener">{esc(e.get("domain"))}</a>{(" · "+esc(e["date"])) if e.get("date") else ""}<div class="sn">{esc(e.get("snippet"))}</div></li>' for e in low["evidence"][:3])
        ev = f'<div class="lab">Documents read as {"supporting" if r == 5 else "contradicting"} it</div><ul class="ev">{items}</ul>'
    oth = ""
    if others or unscored:
        rows = "".join(f'<li><span class="chip r{R(c["rating"])}">{R(c["rating"])}</span><span class="m">{c["score"]:+.2f}</span><span class="oc">{esc(c["claim"])}</span></li>' for c in sorted(others, key=lambda c: c["score"]))
        note = f'<li class="mut">{unscored} more claim{"s" if unscored!=1 else ""} not verify eligible, no score</li>' if unscored else ""
        oth = f'<div class="lab">Other claims in the post</div><ul class="oth">{rows}{note}</ul>'
    multi = '<p class="cav">The post takes the rating of its lowest claim. Check whether that claim is asserted by the post or only quoted or attacked.</p>' if len(scored) > 1 else ""
    tg = tags.get(p["post_id"]) or {}
    chips = []
    if tg.get("stance") in ("negated", "attributed"):
        chips.append(f'<span class="tag warn">post {"negates" if tg["stance"]=="negated" else "only attributes"} this claim</span>')
    for t in tg.get("tags") or []:
        chips.append(f'<span class="tag{" ok" if t=="plausibly_false" else (" warn" if t in ("confirming_evidence","attribution_pair") else "")}">{TAGNAME.get(t, t)}</span>')
    audit = ""
    if chips or tg.get("note"):
        audit = f'<div class="lab">Hand audit</div><div class="tags">{"".join(chips)}</div>' + (f'<p class="note">{esc(tg.get("note"))}</p>' if tg.get("note") else "")
    return f'''<aside class="ins">
<div class="top"><span class="badge r{r}">{r}</span><div><div class="rl">{rlabel(r)}</div><div class="sc">score <span class="m">{p["post_score"]:+.2f}</span> · <a href="{esc(p["url"])}" target="_blank" rel="noopener">open on X</a></div></div></div>
<div class="lab">Claim that set the rating</div>
<p class="cl">{esc(low["claim"])}</p>
<div class="fl">{flagcells(low.get("flags"))}<span class="m">{low["score"]:+.2f}</span></div>
{ev}{oth}{multi}{audit}
</aside>'''

rows = "".join(f'<section class="row" data-r="{R(p["post_rating"])}">{card(p)}{inspector(p)}</section>' for p in posts)
if PROJECT:
    n5, n4, c5, c4, ns = d["n_posts"]["1"], d["n_posts"]["2"], d["n_claims"]["1"], d["n_claims"]["2"], d["n_claims_scored"]
    n5t, c5t = d["n_posts"]["5"], d["n_claims"]["5"]
    share = f'{d["rating_share_pct"]["1"]:.1f} / {d["rating_share_pct"]["2"]:.1f} / {d["rating_share_pct"]["5"]:.1f}%'
else:
    n5, n4, c5, c4, ns = d["n_posts_r5"], d["n_posts_r4"], d["n_claims_r5"], d["n_claims_r4"], d["n_claims_scored"]
    share = f'{d["rating_share_pct"]["5"]:.1f} / {d["rating_share_pct"]["4"]:.1f}%'
s1 = sum(1 for p in posts if R(p["post_rating"]) == 1); s2 = sum(1 for p in posts if R(p["post_rating"]) == 2); s5 = len(posts) - s1 - s2

TITLE = "Outlet Ratings" if MODE == "outlet" else "Ratings One and Two"
if MODE == "outlet":
    SUB = f"Posts from mid and low tier NewsGuard outlets rated 1, 2 or 5 by truth odds, drawn as the survey respondent would see them, with the tool's reasons beside each · E2 run rescored 9 September 2026"
    INTRO = f"The pool is the E2 evidence run over the 50 outlet roster (2,000 posts, context on, post date as search ceiling), rescored with the frozen 7 flag weights and pad to ten, restricted to outlets scoring below 70 on NewsGuard. That leaves {ns:,} claims from {d['n_posts_scored']:,} posts. A post takes the rating of its lowest scoring claim, and for rating 5 the inspector shows the highest scoring claim. The scale is the project convention, 1 is certainly false and 5 is true. Rating 1 is the 2 percent false alarm cut, score below -4.08. Rating 2 sits between that and the 5 percent cut at -2.08. Rating 5 is the 2 percent miss cut, score 5.56 and above. The ten squares are the ten documents read for the claim that set the rating."
    NOTE = f"Twenty five posts per rating, a seeded random draw from the {n5}, {n4} and {n5t} available, newest first. Images are embedded, clicking one opens the post on X. The outlet lean and NewsGuard score sit at the top right of each card. Hand audit chips, where present, come from a read of every rating 1 and 2 post in this pool."
else:
    SUB = "Twenty five timeline posts rated 1 and twenty five rated 2, drawn as the survey respondent would see it, with the tool's reasons beside it · scored 9 September 2026"
    INTRO = "The left column is the post as it would appear in Phase 1 of the survey, with the PI's alert wording under it. The right column is what the tool knows. A post takes the rating of its lowest scoring claim. The scale is the project convention, 1 is certainly false and 5 is true. Rating 1 is the 2 percent false alarm cut, score below -4.08. Rating 2 sits between that and the 5 percent cut at -2.08. The ten squares are the ten documents read for the claim that set the rating, darkest for a direct contradiction, empty for silence. Weights are the 7 flag fit on the frozen fc-gold population, pad to ten silent."
    NOTE = "The fifty posts are a seeded random draw from the 104 and 44 in the two bands, newest first. Images are embedded, clicking one opens it on X. The hand audit chips at the bottom of each inspector come from a read of all 148 posts, they mark whether the post asserts, attributes or negates the rated claim and whether the contradicting documents are about the claim at all. Under T1 only rating 1 posts carry the alert, under T2 both do. Watch for posts that quote or attack the claim they are rated on, the extraction pulls the quoted claim out and the post inherits its score."
page = f'''<title>{TITLE}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg:#fbfaf7; --ink:#1c1b19; --mut:#6e6a63; --line:#e2dfd8; --panel:#ffffff;
  --accent:#7a3020; --barfill:#c9c4ba;
  --xbg:#ffffff; --xink:#0f1419; --xmut:#536471; --xline:#cfd9de; --xhov:#f7f9f9; --xblue:#1d9bf0;
  --f1:#7a3020; --f2:#b8735f; --f3:#6e6a63; --fX:#c9c4ba; --fI:transparent; --f4:#8d8a83; --f5:#1c1b19;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#161513; --ink:#e7e4dd; --mut:#98938a; --line:#302d28; --panel:#1e1c19;
  --accent:#d08363; --barfill:#4a463f;
  --xbg:#000000; --xink:#e7e9ea; --xmut:#71767b; --xline:#2f3336; --xhov:#080808; --xblue:#1d9bf0;
  --f1:#d08363; --f2:#9a6a58; --f3:#98938a; --fX:#4a463f; --f4:#7a766f; --f5:#e7e4dd;
}} }}
:root[data-theme="dark"] {{
  --bg:#161513; --ink:#e7e4dd; --mut:#98938a; --line:#302d28; --panel:#1e1c19;
  --accent:#d08363; --barfill:#4a463f;
  --xbg:#000000; --xink:#e7e9ea; --xmut:#71767b; --xline:#2f3336; --xhov:#080808; --xblue:#1d9bf0;
  --f1:#d08363; --f2:#9a6a58; --f3:#98938a; --fX:#4a463f; --f4:#7a766f; --f5:#e7e4dd;
}}
body {{ background:var(--bg); color:var(--ink); font:16px/1.6 "IBM Plex Sans",system-ui,sans-serif; margin:0; padding:3rem 2rem 5rem; }}
main {{ max-width:1080px; margin:0 auto; }}
h1 {{ font:600 2.3rem/1.2 "IBM Plex Serif",Georgia,serif; margin:0 0 .4rem; text-wrap:balance; }}
.sub {{ color:var(--mut); margin:0 0 1.6rem; }}
p {{ margin:.7rem 0; }}
.m {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; }}
a {{ color:var(--accent); }}
.stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.4rem 0; }}
.stat {{ border-top:2px solid var(--line); padding-top:.5rem; }}
.stat b {{ display:block; font:500 1.5rem/1.2 "IBM Plex Mono",monospace; }}
.stat span {{ color:var(--mut); font-size:.85rem; }}
.ctl {{ position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:.8rem 0; margin:1.2rem 0 1.6rem; display:flex; flex-wrap:wrap; gap:1.2rem 2.4rem; align-items:center; }}
.seg {{ display:inline-flex; border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.seg button {{ background:transparent; color:var(--ink); border:0; padding:.4rem .9rem; font:inherit; font-size:.9rem; cursor:pointer; }}
.seg button + button {{ border-left:1px solid var(--line); }}
.seg button[aria-pressed="true"] {{ background:var(--ink); color:var(--bg); }}
.seg button:focus-visible {{ outline:2px solid var(--accent); outline-offset:-2px; }}
.ctl .lab {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); margin-right:.6rem; }}
.legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1rem; font-size:.8rem; color:var(--mut); align-items:center; }}
.legend span {{ display:inline-flex; align-items:center; gap:.3rem; }}
.row {{ display:grid; grid-template-columns:600px minmax(0,1fr); gap:1.6rem; align-items:start; padding:1.4rem 0; border-bottom:1px solid var(--line); }}
.row[hidden] {{ display:none; }}
@media (max-width:1000px) {{ .row {{ grid-template-columns:1fr; }} }}
/* the tweet */
.tw {{ background:var(--xbg); color:var(--xink); border:1px solid var(--xline); border-radius:16px; padding:14px 16px; display:grid; grid-template-columns:44px 1fr; gap:12px; font:15px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
.av {{ width:42px; height:42px; border-radius:50%; background:hsl(var(--h) 35% 45%); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:18px; }}
.hd {{ display:flex; gap:6px; align-items:baseline; flex-wrap:wrap; }}
.hd b {{ font-weight:700; }}
.hd span {{ color:var(--xmut); }}
.hd .ng {{ margin-left:auto; font-size:12px; color:var(--mut); font-family:"IBM Plex Sans",system-ui,sans-serif; }}
.badge.r5 {{ background:var(--ink); color:var(--bg); }} .chip.r5 {{ background:var(--ink); color:var(--bg); border-color:var(--ink); }}
.tx {{ white-space:pre-wrap; overflow-wrap:anywhere; margin-top:4px; }}
.quote {{ border:1px solid var(--xline); border-radius:12px; padding:10px 12px; margin-top:10px; font-size:14px; }}
.qh b {{ font-weight:700; }} .qh span {{ color:var(--xmut); margin-left:4px; }}
.qt {{ white-space:pre-wrap; overflow-wrap:anywhere; margin-top:2px; }}
.media {{ margin-top:10px; border:1px dashed var(--xline); border-radius:12px; padding:22px 12px; text-align:center; color:var(--xmut); font-size:13px; display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }}
.media a {{ color:var(--xblue); text-decoration:none; }}
.pics {{ margin-top:10px; display:grid; gap:4px; border-radius:12px; overflow:hidden; border:1px solid var(--xline); }}
.pics.n2, .pics.n3, .pics.n4 {{ grid-template-columns:1fr 1fr; }}
.pics img {{ display:block; width:100%; height:100%; object-fit:cover; max-height:520px; }}
.pics.n2 img, .pics.n3 img, .pics.n4 img {{ max-height:260px; }}
.pics.n3 a:first-child {{ grid-row:span 2; }} .pics.n3 a:first-child img {{ max-height:none; }}
.tags {{ display:flex; flex-wrap:wrap; gap:.35rem; }}
.tag {{ font-size:.75rem; border:1px solid var(--line); border-radius:3px; padding:.05rem .45rem; color:var(--mut); }}
.tag.warn {{ border-color:var(--accent); color:var(--accent); }}
.tag.ok {{ border-color:var(--ink); color:var(--ink); }}
.note {{ color:var(--mut); font-size:.82rem; margin:.35rem 0 0; }}
.mx {{ display:flex; justify-content:space-between; max-width:420px; margin-top:12px; color:var(--xmut); font-size:13px; }}
.mx span {{ display:inline-flex; align-items:center; gap:5px; }}
.mx svg {{ width:17px; height:17px; }}
.alert {{ margin-top:12px; border:1px solid var(--xline); border-radius:12px; padding:10px 12px; display:flex; gap:10px; align-items:flex-start; font-size:14px; }}
.alert svg {{ width:20px; height:20px; flex:0 0 auto; margin-top:1px; }}
.alert b {{ font-weight:700; }}
.t1 .row[data-r="2"] .alert {{ display:none; }}
/* the inspector */
.ins {{ font-size:.9rem; min-width:0; }}
.top {{ display:flex; gap:.8rem; align-items:center; margin-bottom:.9rem; }}
.badge {{ width:2.4rem; height:2.4rem; border-radius:6px; display:flex; align-items:center; justify-content:center; font:500 1.3rem "IBM Plex Mono",monospace; flex:0 0 auto; }}
.badge.r1 {{ background:var(--accent); color:var(--bg); }}
.badge.r2 {{ border:2px solid var(--accent); color:var(--accent); }}
.rl {{ font:600 1.05rem/1.2 "IBM Plex Serif",Georgia,serif; }}
.sc {{ color:var(--mut); font-size:.85rem; }}
.ins .lab {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); margin:.9rem 0 .25rem; }}
.cl {{ margin:0; font-weight:600; }}
.fl {{ display:flex; align-items:center; gap:3px; margin-top:.4rem; }}
.fl .m {{ margin-left:.5rem; color:var(--mut); font-size:.85rem; }}
.f {{ width:12px; height:12px; border:1px solid var(--mut); border-radius:2px; display:inline-block; }}
.f1 {{ background:var(--f1); border-color:var(--f1); }} .f2 {{ background:var(--f2); border-color:var(--f2); }}
.f3 {{ background:repeating-linear-gradient(45deg,var(--f3) 0 2px,transparent 2px 4px); }}
.fX {{ background:var(--fX); }} .fI {{ background:var(--fI); }}
.f4 {{ background:var(--f4); border-color:var(--f4); }} .f5 {{ background:var(--f5); border-color:var(--f5); }}
.ev, .oth {{ list-style:none; padding:0; margin:0; }}
.ev li {{ margin:.35rem 0; }}
.sn {{ color:var(--mut); font-size:.82rem; line-height:1.45; }}
.oth li {{ display:grid; grid-template-columns:1.4rem 3.6rem minmax(0,1fr); gap:.5rem; align-items:baseline; margin:.3rem 0; }}
.oth .mut {{ display:block; color:var(--mut); font-size:.82rem; }}
.chip {{ font:500 .75rem "IBM Plex Mono",monospace; border:1px solid var(--line); border-radius:3px; text-align:center; }}
.chip.r1 {{ background:var(--accent); color:var(--bg); border-color:var(--accent); }} .chip.r2 {{ color:var(--accent); border-color:var(--accent); }}
.oc {{ overflow-wrap:anywhere; }}
.cav {{ color:var(--mut); font-size:.82rem; border-left:3px solid var(--line); padding-left:.7rem; margin:.9rem 0 0; }}
.count {{ color:var(--mut); font-size:.85rem; }}
@media (prefers-reduced-motion: no-preference) {{ .seg button {{ transition:background .12s; }} }}
</style>
<main>
<h1>{TITLE}</h1>
<p class="sub">{SUB}</p>
<p>{INTRO}</p>
<p>{NOTE}</p>
<div class="stats">
<div class="stat"><b>{s1} of {n5}</b><span>posts rated 1 shown, {c5} claims in the band</span></div>
<div class="stat"><b>{s2} of {n4}</b><span>posts rated 2 shown, {c4} claims in the band</span></div>
<div class="stat"><b>{ns:,}</b><span>claims scored</span></div>
<div class="stat"><b>{share}</b><span>share of scored claims in {"1 / 2 / 5" if MODE == "outlet" else "1 / 2"}</span></div>
</div>
<div class="ctl">
<div><span class="lab">Show</span><span class="seg" role="group" aria-label="rating filter"><button data-f="1" aria-pressed="true">Rating 1 <span class="count">({s1})</span></button><button data-f="2" aria-pressed="false">Rating 2 <span class="count">({s2})</span></button>{(f'<button data-f="5" aria-pressed="false">Rating 5 <span class="count">({s5})</span></button>' if MODE == "outlet" else "")}<button data-f="all" aria-pressed="false">All</button></span></div>
<div><span class="lab">Treatment</span><span class="seg" role="group" aria-label="treatment"><button data-t="t1" aria-pressed="false">T1, alert on 1</button><button data-t="t2" aria-pressed="true">T2, alert on 1 and 2</button></span></div>
<div class="legend"><span><i class="f f1"></i>contradicts</span><span><i class="f f2"></i>points against</span><span><i class="f f3"></i>contested</span><span><i class="f fX"></i>on topic, no direction</span><span><i class="f fI"></i>silent</span><span><i class="f f4"></i>points toward</span><span><i class="f f5"></i>states it</span></div>
</div>
<div id="feed" class="t2">{rows}</div>
</main>
<script>
(function(){{
  var feed=document.getElementById('feed');
  var rows=Array.prototype.slice.call(feed.querySelectorAll('.row'));
  function setFilter(f){{
    rows.forEach(function(r){{ r.hidden = !(f==='all' || r.dataset.r===f); }});
    document.querySelectorAll('[data-f]').forEach(function(b){{ b.setAttribute('aria-pressed', String(b.dataset.f===f)); }});
  }}
  function setTreat(t){{
    feed.className=t;
    document.querySelectorAll('[data-t]').forEach(function(b){{ b.setAttribute('aria-pressed', String(b.dataset.t===t)); }});
  }}
  document.querySelectorAll('[data-f]').forEach(function(b){{ b.addEventListener('click', function(){{ setFilter(b.dataset.f); }}); }});
  document.querySelectorAll('[data-t]').forEach(function(b){{ b.addEventListener('click', function(){{ setTreat(b.dataset.t); }}); }});
  setFilter('1');
}})();
</script>
'''
open(OUT,"w").write(page)
print(len(page)//1024, "KB")
